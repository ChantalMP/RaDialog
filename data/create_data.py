import argparse
import dataclasses
import json
import os
from enum import auto, Enum
from pathlib import Path
from typing import List, Any
import random

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data.sampler import Sampler

from data.instruct_tasks import create_direct_task_data, create_cp_task_data, create_correction_task_data, create_nle_task_data
from local_config import VIS_ROOT, PATH_TO_MIMIC_CXR
from model.lavis.models.blip2_models.modeling_llama_imgemb import LlamaForCausalLM


class MyReportProcessor():
    def __init__(self, prompt="", max_words=50, prompt_neg=""):
        self.prompt = prompt
        self.max_words = max_words
        self.prompt_neg = prompt_neg

    def __call__(self, findings, no_labels=False):
        prompt = self.prompt

        if no_labels:
            findings = "no common findings"  # cannot write which findings as we don't no them
        prompt = prompt.format(findings=findings)

        return prompt

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    # Used for gradio server
    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += self.sep + " " + role + ": " + message
                else:
                    ret += self.sep + " " + role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


def create_conv():
    conv = Conversation(
        system="A chat between a curious user and an artificial intelligence assistant acting as an experienced radiologist. "
               "The assistant gives professional, detailed, and polite answers to the user's questions.",
        roles=["USER", "ASSISTANT"],
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    )
    return conv


class MIMIC_Text_Dataset(Dataset):
    def __init__(self, split, truncate=None, prompt_type="basic", use_indication=False):
        super().__init__()

        # load csv file
        self.split = pd.read_csv(
            f'{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv')
        self.reports = pd.read_csv('mimic-cxr/reports_processed/mimic_cxr_sectioned.csv')
        # drop reports where findings are nan
        self.reports = self.reports.dropna(subset=['findings'])

        self.img_ids = {img_id: i for i, img_id in enumerate(self.reports['dicom_id'])}
        self.chexpert = pd.read_csv(f'data/data_files/finding_chexbert_labels.csv')
        self.chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]

        self.use_indication = use_indication

        self.vis_root = VIS_ROOT

        self.prompt_type = prompt_type

        self.split_ids = set(self.split.loc[self.split['split'] == split]['dicom_id'])
        self.train_ids = set(self.split.loc[self.split['split'] == 'train']['dicom_id'])

        # get all dicom_ids where "split" is split
        self.annotation = self.reports.loc[self.reports['dicom_id'].isin(self.split_ids)]
        if truncate is not None:
            self.annotation = self.annotation[:truncate]

        self.annotation['findings'] = self.annotation['findings'].apply(lambda x: x.replace('\n', ''))

        # Extract patient_id from Img_Folder (3rd part) and study_id is the name of the notefile without the pre-pending 's'
        self.annotation['subject_id'] = self.annotation['Img_Folder'].apply(lambda x: int(x.split('/')[2].lstrip('p')))
        self.annotation['study_id'] = self.annotation['Note_file'].apply(lambda x: int(x.lstrip('s').rstrip('.txt')))

        # Merge chexpert labels with annotation dataframe
        self.annotation = pd.merge(self.annotation, self.chexpert, how='left', left_on=['dicom_id'],
                                   right_on=['dicom_id'])

        # for every row add a string of comma-separated positive labels
        self.annotation['positive_labels'] = self.annotation.apply(lambda x: self.convert_to_finding_labels(x[self.chexpert_cols].values,
                                                                                                            self.chexpert_cols), axis=1)

        # maybe use transforms from here: ResNet50_Weights.IMAGENET1K_V2.transforms
        # read prompt from json
        prompts = json.loads(Path(f"vicuna_prompts.json").read_text(encoding="UTF-8"))
        self.text_processor = MyReportProcessor(
            prompt=prompts[prompt_type], max_words=1000,
            prompt_neg=prompts[prompt_type.replace("matching_examples", "neg_matching_examples")])

    def convert_to_finding_labels(self, chexpert_labels, columns, label=1):
        # Get indices where value is 1
        indices = np.where(chexpert_labels == label)
        # Get the corresponding column names and join them into a string
        labels = ", ".join([columns[i] for i in indices[0]])
        return labels

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        # if self.use_indication:
        #     indication = self.indications[study_id]
        #     if indication == "":
        #         indication = "Indication not given."
        caption = ann["findings"].strip()
        chexpert_labels = ann[self.chexpert_cols].astype(float).values
        chexpert_label_str = ann["positive_labels"]
        dicom_id = ann["dicom_id"]

        # check if all columns are in (nan, 0) -> no labels
        no_labels = np.all((np.isnan(chexpert_labels)) | (chexpert_labels == 0) | (chexpert_labels == -1.))
        finding_string = chexpert_label_str.lower().strip()

        input_text = self.text_processor(findings=finding_string, no_labels=no_labels)

        # if self.use_indication:
        #     input_text = "Indication: " + indication + " " + input_text

        # template for vicuna v1.3
        conv = Conversation(
            system="A chat between a curious user and an artificial intelligence assistant acting as an experienced radiologist. "
                   "The assistant gives professional, detailed, and polite answers to the user's questions.",
            roles=["USER", "ASSISTANT"],
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        return {
            "text_input": prompt,
            "text_target": caption,
            "ig_label_string": finding_string,
            "chexpert_labels": chexpert_labels,
            "chexpert_cols": self.chexpert_cols,
            "dicom": dicom_id,
            "img_path": ann["Img_Folder"] + "/" + ann["Img_Filename"],
        }

    def __len__(self):
        return len(self.annotation)


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def stratified_sample(df, simulated_epochs=1):
    # We want to reduce the number of examples with no finding to 1/14th of the dataset. We achieve this easily by first seperating the dataset into 2 groups: no finding and finding.
    # either no finding, or nothing is considered a no finding
    no_findings_indices = df.annotation[((df.annotation['No Finding'] == 1) | ((df.annotation[df.chexpert_cols] == 1).sum(1) == 0) == 1)].index
    finding_indices = df.annotation.index.difference(no_findings_indices)
    no_findings_indices = no_findings_indices.tolist()
    finding_indices = finding_indices.tolist()

    # we are striving to lose as little no_finding data as possible. So instead of just reducing the number of no_finding examples, we will increase the number of finding examples. Just clone and extend dataset
    finding_indices = finding_indices * simulated_epochs
    # subsample the no finding examples to be 1/14th of the new dataset
    new_dataset_size = len(finding_indices) * 14 / 13
    new_no_finding_count = int(new_dataset_size / 14)
    # merge considering the new dataset size
    all_indices = finding_indices + random.sample(no_findings_indices, new_no_finding_count)
    return all_indices


def create_report_data_vicuna_specific_stratified(prompt_type):
    val_dataset = MIMIC_Text_Dataset(split="train", truncate=None, prompt_type=prompt_type)
    stratified_indices = stratified_sample(val_dataset, simulated_epochs=2)
    sampler = SubsetSampler(stratified_indices)
    data_loader = DataLoader(val_dataset, batch_size=200, num_workers=200, sampler=sampler)

    report_jsons = []
    for _, batch in tqdm(enumerate(data_loader)):
        # iterate over batch elements
        for i in range(len(batch["text_input"])):
            text_input = batch["text_input"][i]
            text_target = batch["text_target"][i]
            dicom = batch["dicom"][i]

            # sample random prompt for every report
            reports_json = {
                "instruction": text_input,
                "input": "",
                "output": text_target,
                "dicom": dicom,
            }
            report_jsons.append(reports_json)

    # Save the JSON data to a file
    with open("data/data_files/mimic_cxr_reports_stratified.json", "w") as f:
        json.dump(report_jsons, f, ensure_ascii=False, indent=4)


'''
this method saves instruct data jsons for all the different tasks we defined:
- easy language: EL DONE
- correction: CO DONE
- summerization: SU DONE
- reasoning: RE (based on MIMIC-NLE) DONE
- region QA: RQA DONE
- CP binary QA: CPbQA DONE
- CP all QA: CPaQA DONE

for every report we sample one task and one prompt and save the report, the question (task) and the answer generated by vicuna (or from dataset groundtruth)
'''


def create_report_data_vicuna_instruct_large():
    lang_model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.3", torch_dtype=torch.float16, device_map='auto', load_in_8bit=False)
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3", use_fast=False, truncation_side="left", padding_side="left")
    tokenizer.pad_token = tokenizer.unk_token

    val_dataset = MIMIC_Text_Dataset(split="train", truncate=None, prompt_type="img_matching_examples_ig2_noexamples")
    # split in 6 portions of 1/6th each, randomly
    split_size = len(val_dataset) // 6
    remainder = len(val_dataset) % 6

    val_dataset_EL, _, val_dataset_SU, val_dataset_EX, val_dataset_RQA, val_dataset_CPQA = torch.utils.data.random_split(val_dataset,
                                                                                                                         [split_size + (i < remainder)
                                                                                                                          for i in range(
                                                                                                                             6)])  # correction is samples somewhere else

    # split val_dataset_CPQA in 2
    split_size = len(val_dataset_CPQA) // 2
    remainder = len(val_dataset_CPQA) % 2
    val_dataset_CPbQA, val_dataset_CPaQA = torch.utils.data.random_split(val_dataset_CPQA, [split_size + (i < remainder) for i in range(2)])

    # create directory
    if not os.path.exists("data/large_instruct_data"):
        os.makedirs("data/large_instruct_data")

    # create data
    create_direct_task_data(lang_model, tokenizer, val_dataset_EL, task_name="EL")
    create_direct_task_data(lang_model, tokenizer, val_dataset_SU, task_name="SU")
    create_direct_task_data(lang_model, tokenizer, val_dataset_RQA, task_name="RQA")
    create_cp_task_data(val_dataset_CPbQA, task_name="CPbQA")
    create_cp_task_data(val_dataset_CPaQA, task_name="CPaQA")

    create_correction_task_data(lang_model, tokenizer)
    create_nle_task_data()


'''
fuse instruct data with report generation task into one dataset json
'''


def fuse_instruct_dataset(prompt_type="img_matching_examples_ig2_noexamples_IMG_findings"):
    # get report generation data
    val_dataset = MIMIC_Text_Dataset(split="train", truncate=None, prompt_type=prompt_type)
    stratified_indices = stratified_sample(val_dataset, simulated_epochs=2)
    sampler = SubsetSampler(stratified_indices)
    data_loader = DataLoader(val_dataset, batch_size=200, sampler=sampler, num_workers=200)
    report_jsons = []
    for _, batch in tqdm(enumerate(data_loader)):
        # iterate over batch elements
        for i in range(len(batch["text_input"])):
            text_input = batch["text_input"][i]
            text_target = batch["text_target"][i]
            dicom = batch["dicom"][i]

            # sample random prompt for every report
            reports_json = {
                "instruction": text_input,
                "input": "",
                "output": text_target,
                "dicom": dicom,
            }
            report_jsons.append(reports_json)

    task_jsons = []
    with open(f"vicuna_prompts.json", "r") as f:
        prompts = json.load(f)
    report_prompt = prompts[prompt_type]

    # get instruct data
    for task in ["EL", "RE", "CO", "SU", "RQA", "CPbQA", "CPaQA"]:
        print("Creating data for " + task)
        with open(f"data/large_instruct_data/instruct_large_{task}.json", "r") as f:
            task_data = json.load(f)

        for elem in tqdm(task_data):
            report = elem["gt_report"] if task != "CO" else elem["incorrect_report"]

            conv = create_conv()
            conv.append_message(conv.roles[0], report_prompt)
            conv.append_message(conv.roles[1], report)
            conv.append_message(conv.roles[0], elem["task"])
            conv.append_message(conv.roles[1], None)

            instruction = conv.get_prompt()

            # get elem directly from val_dataset.train_annotation with same dicom
            orig_elem = val_dataset.annotation[val_dataset.annotation["dicom_id"] == elem["dicom"]].iloc[0]

            if type(orig_elem['positive_labels']) == float and np.isnan(orig_elem['positive_labels']):
                finding_str = "no common findings"
            else:
                finding_str = orig_elem['positive_labels'].lower().strip()
            instruction = instruction.format(findings=finding_str)

            task_json = {
                "instruction": instruction,
                "input": "",
                "output": elem["output"].lower().strip() if task == "CPaQA" else elem["output"].strip(),
                "dicom": elem["dicom"],
            }
            task_jsons.append(task_json)

    # combine and shuffle report and task jsons
    combined_jsons = report_jsons + task_jsons
    random.shuffle(combined_jsons)

    # save to json
    with open(f"data/data_files/mimic_cxr_instruct_stratified.json", "w") as f:
        json.dump(combined_jsons, f, indent=4)


if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='RG', help='RG or INS')
    args = parser.parse_args()

    ''' Create data to train RaDialog-RG model'''
    if args.mode == 'RG':
        create_report_data_vicuna_specific_stratified(prompt_type="img_matching_examples_ig2_noexamples_IMG_findings")

    ''' Create data to train RaDialog-INS model'''
    if args.mode == 'INS':
        create_report_data_vicuna_instruct_large()
        fuse_instruct_dataset()

    # This code is meant for understanding how our instruct dataset is created.
    # Due to randomness in the sampling and model predictions, a newly generated dataset could be slightly different.
    # To exactly reproduce our results, please use the instruct dataset we published and use 'fuse_instruct_dataset' to merge with your MIMIC data.