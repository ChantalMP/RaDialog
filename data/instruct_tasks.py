import dataclasses
import json
import random
from enum import Enum, auto
from pathlib import Path
from typing import List, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from local_config import PATH_TO_MIMIC_NLE


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


def create_direct_task_data(lang_model, tokenizer, val_dataset, task_name):
    prompts = pd.read_csv(f"data/instruct_prompts/{task_name}_prompts.csv")["instruction"].tolist()
    data_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=0)
    report_jsons = []
    print("Dataloader len: ", len(data_loader))
    for _, batch in tqdm(enumerate(data_loader)):

        # Create prompts for every report
        # sample batchsize questions from EL_prompts
        batch_prompts = random.choices(prompts, k=len(batch["text_input"]))
        batch_instructions = []
        for text_target, prompt in zip(batch["text_target"], batch_prompts):
            conv = create_conv()
            conv.append_message(conv.roles[0], "Report: " + text_target + "\n" + prompt)
            conv.append_message(conv.roles[1], None)
            batch_instructions.append(conv.get_prompt())

        inputs = tokenizer.batch_encode_plus(batch_instructions, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(torch.device("cuda"))

        # generate answers with no-lora vicuna
        generation_output = lang_model.generate(
            input_ids=input_ids,
            dicom=None,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256
        )
        preds = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        preds = [p.split("ASSISTANT:")[1] for idx, p in enumerate(preds)]

        # iterate over batch elements
        for i in range(len(batch["text_input"])):
            text_target = batch["text_target"][i]  # GT report
            task_prompt = batch_prompts[i]
            task_instruction = batch_instructions[i]
            answer = preds[i]
            dicom = batch["dicom"][i]

            # sample random prompt for every report
            reports_json = {
                "gt_report": text_target,
                "task": task_prompt,
                "instruction": task_instruction,
                "input": "",
                "output": answer,
                "dicom": dicom,
                "task_type": task_name
            }
            report_jsons.append(reports_json)

    # save
    with open(f"data/large_instruct_data/instruct_large_{task_name}.json", "w") as f:
        json.dump(report_jsons, f, ensure_ascii=False, indent=4)


def create_cp_task_data(val_dataset, task_name):
    prompts = pd.read_csv(f"data/instruct_prompts/{task_name}_prompts.csv")["instruction"].tolist()
    data_loader = DataLoader(val_dataset, batch_size=200, shuffle=False, num_workers=200)
    report_jsons = []
    for _, batch in tqdm(enumerate(data_loader)):

        # Create prompts for every report
        # sample batchsize questions from EL_prompts
        batch_prompts = random.choices(prompts, k=len(batch["text_input"]))

        # iterate over batch elements
        for i in range(len(batch["text_input"])):
            text_target = batch["text_target"][i]  # GT report
            task_prompt = batch_prompts[i]
            cp_indices = np.where(batch["chexpert_labels"][i] == 1.)
            cp_findings = [val_dataset.dataset.dataset.chexpert_cols[i] for i in cp_indices[0]]

            if task_name == "CPbQA":  # binary QA
                if "No Finding" in cp_findings:
                    cp_findings.remove("No Finding")
                # 50% sample finding from cp_findings, 50% sample finding from val_dataset.dataset.dataset.chexpert_cols - cp_findings
                if random.random() < 0.6 and len(cp_findings) > 0:
                    finding = random.choice(cp_findings)  # answer: yes
                    answer = 'yes'
                else:
                    finding = random.choice(list(set(val_dataset.dataset.dataset.chexpert_cols[1:]) - set(cp_findings)))  # answer: no
                    answer = 'no'
                task_prompt = task_prompt.replace("<X>", finding)

            elif task_name == "CPaQA":  # give all findings
                answer = ', '.join(cp_findings)

            dicom = batch["dicom"][i]

            # sample random prompt for every report
            reports_json = {
                "gt_report": text_target,
                "task": task_prompt,
                "input": "",
                "output": answer,
                "dicom": dicom,
                "task_type": task_name
            }
            report_jsons.append(reports_json)

    # save
    with open(f"data/large_instruct_data/instruct_large_{task_name}.json", "w") as f:
        json.dump(report_jsons, f, ensure_ascii=False, indent=4)


class CorrectionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        fp = sample["fp"]
        fn = sample["fn"]
        fp_str = ', '.join(fp)
        fp_str = fp_str.rsplit(', ', 1)
        fp_str = ' and '.join(fp_str)
        fn_str = ', '.join(fn)
        fn_str = fn_str.rsplit(', ', 1)
        fn_str = ' and '.join(fn_str)

        gt_report = sample["gt_report"]
        pred_report = sample["pred_report"]
        dicom = sample["dicom"]
        return {'gt_report': gt_report, 'pred_report': pred_report, 'fp': fp_str, 'fn': fn_str, 'dicom': dicom}


def create_correction_task_data(lang_model, tokenizer):
    # load correction json
    with open("data/instruct_prompts/instruct_task_correction_preds.json") as f:
        correction_preds = json.load(f)

    # create pytorch dataset from json
    correction_dataset = CorrectionDataset(correction_preds)
    data_loader = DataLoader(correction_dataset, batch_size=12, shuffle=False, num_workers=12)

    prompts_both = pd.read_csv(f"data/instruct_prompts/CO_both_prompts.csv")["instruction"].tolist()
    prompts_add = pd.read_csv(f"data/instruct_prompts/CO_add_prompts.csv")["instruction"].tolist()
    prompts_rem = pd.read_csv(f"data/instruct_prompts/CO_rem_prompts.csv")["instruction"].tolist()
    report_jsons = []
    for _, batch in tqdm(enumerate(data_loader)):
        # use very clear, fixed prompt for data generation -> in training use random prompts

        fixed_batch_prompts = []
        for fp, fn in zip(batch["fp"], batch["fn"]):
            fixed_corr_prompt = "Please provide an adapted report. "
            if fp != "":
                fixed_corr_prompt += f"Do not mention {fp}. "
            if fn != "":
                fixed_corr_prompt += f"Mention {fn}. "

            if fp == "" and fn == "":
                fixed_corr_prompt = "NOCHANGE"
            fixed_batch_prompts.append(fixed_corr_prompt.strip())

        batch_prompts = []
        for fp, fn in zip(batch["fp"], batch["fn"]):
            if fp == "" and fn == "":
                batch_prompts.append("NOCHANGE")
            elif fp == "":
                batch_prompts.append(random.choice(prompts_add).replace("<add>", fn))
            elif fn == "":
                batch_prompts.append(random.choice(prompts_rem).replace("<rem>", fp))
            else:
                batch_prompts.append(random.choice(prompts_both).replace("<add>", fn).replace("<rem>", fp))

        batch_instructions = []
        for pred_report, prompt in zip(batch["pred_report"], fixed_batch_prompts):
            conv = create_conv()
            conv.append_message(conv.roles[0], "Please write a radiology report for the given x-ray.")
            conv.append_message(conv.roles[1], pred_report)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            batch_instructions.append(conv.get_prompt())

        inputs = tokenizer.batch_encode_plus(batch_instructions, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(torch.device("cuda"))

        # generate answers with no-lora vicuna
        generation_output = lang_model.generate(
            input_ids=input_ids,
            dicom=None,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256
        )
        preds = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        preds = [p.split("ASSISTANT:")[-1].strip() for idx, p in enumerate(preds)]

        # iterate over batch elements
        for i in range(len(batch["pred_report"])):
            gt_report = batch["gt_report"][i]  # GT report
            incorrect_report = batch["pred_report"][i]  # predicted report that will be corrected
            task_prompt = batch_prompts[i]
            task_instruction = batch_instructions[i]
            answer = preds[i]
            dicom = batch["dicom"][i]

            if task_prompt == "NOCHANGE":
                continue  # we don't want to train for correction on already correct reports
            # sample random prompt for every report
            reports_json = {
                "gt_report": gt_report,
                "incorrect_report": incorrect_report,
                "task": task_prompt,
                "instruction": task_instruction,
                "input": "",
                "output": answer,
                "dicom": dicom,
                "task_type": 'CO'
            }
            report_jsons.append(reports_json)

    # save
    with open(f"data/large_instruct_data/instruct_large_CO.json", "w") as f:
        json.dump(report_jsons, f, ensure_ascii=False, indent=4)


def create_nle_task_data():
    MIMIC_DIAGNOSISLIST = ['Atelectasis', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion',
                           'Pleural Other', 'Pneumonia', 'Pneumothorax']
    # load mimic_nle json
    mimic_nle = []
    with open(f'{PATH_TO_MIMIC_NLE}/mimic-nle/mimic-nle-train.json', 'r') as f:
        for line in f:
            obj = json.loads(line)
            mimic_nle.append(obj)

    prompts = pd.read_csv(f"data/instruct_prompts/RE_prompts.csv")["instruction"].tolist()
    report_jsons = []
    reports = pd.read_csv('mimic-cxr/reports_processed/mimic_cxr_sectioned.csv')
    reports = reports.dropna(subset=['findings'])
    reports['findings'] = reports['findings'].apply(lambda x: x.replace('\n', ''))

    for sample in tqdm(mimic_nle):
        report_id = sample["report_ID"]
        gt_report = reports[reports["Note_file"] == f"{report_id}.txt"]["findings"].tolist()
        if len(gt_report) == 0:  # report did have no findings section
            continue
        gt_report = gt_report[0]

        nle = sample['nle']
        if nle not in gt_report:  # sort out samples that reference the impression instead of the findings section
            continue

        dicom = reports[reports["Note_file"] == f"{report_id}.txt"]["dicom_id"].tolist()[0]
        task_prompt = random.choice(prompts)

        diagnoses = [d for idx, d in enumerate(MIMIC_DIAGNOSISLIST) if sample["diagnosis_label"][idx] == 1]
        diagnoses_string = ", ".join(diagnoses)
        diagnoses_string = diagnoses_string.rsplit(', ', 1)
        diagnoses_string = ' and '.join(diagnoses_string)
        task_prompt = task_prompt.replace("<X>", diagnoses_string)

        # sample random prompt for every report
        reports_json = {
            "gt_report": gt_report,
            "task": task_prompt,
            "input": "",
            "output": sample['nle'],
            "dicom": dicom,
            "task_type": 'RE'
        }
        report_jsons.append(reports_json)

    # save
    print(len(report_jsons))
    with open(f"data/large_instruct_data/instruct_large_RE.json", "w") as f:
        json.dump(report_jsons, f, ensure_ascii=False, indent=4)
