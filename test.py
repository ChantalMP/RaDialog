import argparse
import dataclasses
import json
import os

from local_config import PATH_TO_MIMIC_CXR, VIS_ROOT, JAVA_HOME, JAVA_PATH

# set java path
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_PATH + os.environ["PATH"]
from enum import auto, Enum
from pathlib import Path
import random
from typing import List, Any

import numpy as np
import pandas as pd
import torch
from peft import PeftModelForCausalLM
from torch import nn
from torch.backends import cudnn

from downstream_tasks.automated_correction import get_correction_prompts
from downstream_tasks.chexpert_classification_downstream import get_chexpert_prompts_bin, get_chexpert_prompts_all
from model.lavis.data.ReportDataset import MIMICEvalCap
from model.lavis.models.blip2_models.modeling_llama_imgemb import LlamaForCausalLM
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from transformers import LlamaTokenizer
from data.create_data import MyReportProcessor

from chexbert.run_chexbert import run_chexbert_labeler

torch.multiprocessing.set_sharing_strategy('file_system')


class MIMIC_Text_Dataset(Dataset):
    def __init__(self, split, truncate=None, prompt_type="basic"):
        super().__init__()

        # load csv file
        self.split = pd.read_csv(f'{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv')
        self.reports = pd.read_csv('mimic-cxr/reports_processed/mimic_cxr_sectioned.csv')

        # drop reports where findings are nan
        self.reports = self.reports.dropna(subset=['findings'])
        self.chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]

        self.img_ids = {img_id: i for i, img_id in enumerate(self.reports['dicom_id'])}
        self.chexpert = pd.read_csv(f'data/data_files/finding_chexbert_labels.csv')

        if split == 'validate':
            self.pred_chexpert_labels = json.load(open('findings_classifier/predictions/structured_preds_chexpert_log_weighting_val_macro.json', 'r'))
        elif split == 'test':
            self.pred_chexpert_labels = json.load(open('findings_classifier/predictions/structured_preds_chexpert_log_weighting_test_macro.json', 'r'))

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

        # read prompt from json
        prompts = json.loads(Path("vicuna_prompts.json").read_text(encoding="UTF-8"))
        self.text_processor = MyReportProcessor(
            prompt=prompts[prompt_type], max_words=1000,
            prompt_neg=prompts[prompt_type.replace("matching_examples", "neg_matching_examples")])

    def create_structured_chexpert_findings(self, ann):
        pred_chexpert_labels = self.pred_chexpert_labels[str(ann['dicom_id'])]
        no_labels = len(pred_chexpert_labels) == 0
        counter = 0
        no_findings = "No Finding" in pred_chexpert_labels
        if no_findings:
            counter += 1
        supp_devices = "Support Devices" in pred_chexpert_labels
        if supp_devices:
            counter += 1
        # We check if there are any findings except no findings and support devices
        if len(pred_chexpert_labels) > counter and no_findings:
            pred_chexpert_labels.remove("No Finding")
            no_findings = False
        finding_string = ', '.join(pred_chexpert_labels).lower().strip()
        return no_labels, finding_string
    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        caption = ann['findings'].strip()
        dicom_id = ann["dicom_id"]

        no_labels, finding_string = self.create_structured_chexpert_findings(ann)

        input_text = self.text_processor(finding_string, no_labels=no_labels)

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
            "chexpert_labels": ann[self.chexpert_cols].astype(float).values,
            "dicom": dicom_id,
            "img_path": ann["Img_Folder"] + "/" + ann["Img_Filename"]
        }

    def __len__(self):
        return len(self.annotation)


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


def compute_metrics(all_preds, evaluator):
    scores, _ = evaluator.evaluate(all_preds)
    b1, b2, b3, b4, meteor, rouge = scores["Bleu_1"], scores["Bleu_2"], scores["Bleu_3"], scores["Bleu_4"], scores["METEOR"], scores["ROUGE_L"]
    return b1, b2, b3, b4, meteor, rouge


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def extract_report(pred):
    pred = pred.split("ASSISTANT:")[1]
    if 'report:' in pred:
        return pred.split("report:")[1]
    elif 'Report:' in pred:
        return pred.split("Report:")[1]
    elif 'REPORT:' in pred:
        return pred.split("REPORT:")[1]
    else:
        return pred


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
    all_indices = finding_indices + no_findings_indices[:new_no_finding_count]
    return all_indices


if __name__ == '__main__':
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="examples",
                        help="prompt type")  # options=["basic", "advanced", "gen_examples", "matching_examples"]
    parser.add_argument("--lora_model", type=str, default=None, help="lora model name")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for dataloader")
    parser.add_argument("--use_embs", action="store_true", help="use img embs as input", default=False)
    parser.add_argument("--do_sample", action="store_true", help="", default=False)
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--num_beams", type=int, default=1, help="beam size for generation")
    parser.add_argument("--do_corr", action="store_true", help="", default=False)
    parser.add_argument("--do_cp_bin_qa", action="store_true", help="", default=False)
    parser.add_argument("--do_cp_all_qa", action="store_true", help="", default=False)
    parser.add_argument("--strat_eval", action="store_true", help="", default=False)

    args = parser.parse_args()
    prompt_type = args.prompt

    # set all seeds to make code deterministic
    setup_seeds(42)
    val_dataset = MIMIC_Text_Dataset(split="test", truncate=None, prompt_type=prompt_type)
    batchsize = 12  # 12
    if args.strat_eval:
        stratified_indices = stratified_sample(val_dataset, simulated_epochs=1)
        sampler = SubsetSampler(stratified_indices)
        data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers, sampler=sampler)
    else:
        data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers)

    if "13b" in args.lora_model:
        vicuna_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3", use_fast=False, truncation_side="right", padding_side="left")
        lang_model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.3", torch_dtype=torch.float16, device_map='auto')
    else:
        vicuna_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3", use_fast=False, truncation_side="right", padding_side="left")
        lang_model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3", torch_dtype=torch.float16, device_map='auto')

    if args.use_embs:
        lang_model.base_model.img_proj_layer = nn.Linear(768, lang_model.base_model.config.hidden_size).to(lang_model.base_model.device)
        vicuna_tokenizer.add_special_tokens({"additional_special_tokens": ["<IMG>"]})
        lang_model.resize_token_embeddings(len(vicuna_tokenizer))

    lang_model = lang_model.cuda()
    if args.lora_model is not None:
        lang_model = PeftModelForCausalLM.from_pretrained(lang_model, f"{args.lora_model}", torch_dtype=torch.float16, use_ram_optimized_load=False).half()
    lang_model.eval()

    vicuna_tokenizer.pad_token = vicuna_tokenizer.unk_token  # unk token is ignored in attention mask
    evaluator = MIMICEvalCap(val_dataset.annotation, val_dataset.img_ids)

    '''Report Generation'''
    exp_name = f"{'_'.join(args.lora_model.split('/'))}"
    # exp_name = f"debug"
    if args.do_corr:
        exp_name += "_before_corr"
    if args.do_cp_bin_qa:
        exp_name += "_before_cp_bin_qa"
    if args.do_cp_all_qa:
        exp_name += "_before_cp_all_qa"

    text_targets = []
    text_inputs = []
    all_preds = []
    all_chexpert_labels = []
    dicom_ids = []
    eval_preds = []
    preds_history = []
    finding_strings = []
    all_study_ids = []

    for _, batch in tqdm(enumerate(data_loader)):
        text_input = batch["text_input"]

        text_target = batch["text_target"]
        chexpert_labels = batch["chexpert_labels"]
        dicom_id = batch["dicom"]

        all_chexpert_labels.extend(chexpert_labels.numpy())

        inputs = vicuna_tokenizer.batch_encode_plus(text_input, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].cuda()

        generation_output = lang_model.generate(
            input_ids=input_ids,
            dicom=dicom_id if args.use_embs else None,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=300,
            # num_beams=args.num_beams
            # do_sample=args.do_sample,
            # temperature=args.temperature,
        )

        if args.do_corr or args.do_cp_bin_qa or args.do_cp_all_qa:  # downstream tasks also need img tokens
            preds = vicuna_tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=False)
            # The special token you want to keep
            special_tokens_to_keep = ["<IMG>"]

            # Get all special tokens and remove the one you want to keep
            all_special_tokens = vicuna_tokenizer.all_special_tokens
            all_special_tokens = [token for token in all_special_tokens if token not in special_tokens_to_keep]

            # Replace all other special tokens
            for idx, output in enumerate(preds):
                for token in all_special_tokens:
                    output = output.replace(token, "")
                preds[idx] = output
        else:
            preds = vicuna_tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)

        text_targets.extend(text_target)
        text_inputs.extend(text_input)
        dicom_ids.extend(dicom_id)

        all_preds.extend([p.split("ASSISTANT:")[1] for idx, p in enumerate(preds)])

        preds_history.extend(preds)

    # save predictions
    pred_dir = Path("chexbert").absolute() / "outputs" / "predictions"
    with open(pred_dir / "predictions_{}.csv".format(exp_name), "w") as f:
        for i in range(len(all_preds)):
            f.write('"' + all_preds[i].replace('"', '') + '"\n')

    eval_preds = [{"image": None, "caption": pred, "image_id": val_dataset.img_ids[dicom]} for pred, dicom in zip(all_preds, dicom_ids)]
    bleu1_score, bleu2_score, bleu3_score, bleu4_score, meteor_score, rouge_score = compute_metrics(eval_preds, evaluator)

    # chexpert score
    # save results to txt file
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    # run chexpert labeler
    torch.cuda.empty_cache()

    run_chexbert_labeler(reports_path=pred_dir / "predictions_{}.csv".format(exp_name), output_path=pred_dir / "labels_{}.csv".format(exp_name))

    # read chexpert labels from file
    cp_pred = pd.read_csv(pred_dir / "labels_{}.csv".format(exp_name))
    pred_labels = np.array(cp_pred[val_dataset.chexpert_cols].values)
    all_chexpert_labels = np.array(all_chexpert_labels)

    # Map present (1) cases to 1 and absent (0, was NaN) and uncertain (-1) cases to 0
    all_chexpert_labels = np.nan_to_num(all_chexpert_labels, nan=0)
    pred_labels = np.nan_to_num(pred_labels, nan=0)
    all_chexpert_labels[all_chexpert_labels == -1] = 0
    pred_labels[pred_labels == -1] = 0

    # Calculate F1 score
    mean_f1 = f1_score(all_chexpert_labels, pred_labels, average="macro")
    mean_prec = precision_score(all_chexpert_labels, pred_labels, average="macro")
    mean_rec = recall_score(all_chexpert_labels, pred_labels, average="macro")
    sample_f1 = f1_score(all_chexpert_labels, pred_labels, average="samples")

    print("Macro F1 Score:", mean_f1)
    print("Sample F1 Score:", sample_f1)

    # Calculate Accuracy
    acc_scores = []
    for i in range(all_chexpert_labels.shape[1]):
        acc = accuracy_score(all_chexpert_labels[:, i], pred_labels[:, i])
        acc_scores.append(acc)

    mean_acc = np.mean(acc_scores)

    # save results to file
    with open(f'vicuna_results/results_{exp_name}.txt', 'w') as f:
        f.write(f"Prompt: {text_input[0]}\n")
        f.write(f"Avg Bleu 1: {bleu1_score}\n")
        f.write(f"Avg Bleu 2: {bleu2_score}\n")
        f.write(f"Avg Bleu 3: {bleu3_score}\n")
        f.write(f"Avg Bleu 4: {bleu4_score}\n")
        f.write(f"Avg Meteor: {meteor_score}\n")
        f.write(f"Avg Rouge: {rouge_score}\n")
        f.write(f"Mean Chexpert F1: {mean_f1}\n")
        f.write(f"Mean Chexpert Precision: {mean_prec}\n")
        f.write(f"Mean Chexpert Recall: {mean_rec}\n")
        f.write(f"Sample Chexpert F1: {sample_f1}\n")
        f.write(f"Mean Chexpert Accuracy: {mean_acc}\n")

    '''
    Automatic Prompt Correction
    '''
    if args.do_corr:
        batchsize = 1
        data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers)
        correction_prompts = get_correction_prompts(preds_history, val_dataset.chexpert_cols, pred_labels, all_chexpert_labels)
        # rerun vicuna with correction prompts
        text_targets_corr = []
        text_inputs_corr = []
        all_preds_corr = []
        all_chexpert_labels_corr = []
        dicom_ids_corr = []
        eval_preds_corr = []
        for idx, batch in tqdm(enumerate(data_loader)):
            # use the corrected prompts
            text_input = [correction_prompts[i] for i in range(batchsize * idx, min(batchsize * (idx + 1), len(correction_prompts)))]
            text_target = batch["text_target"]
            chexpert_labels = batch["chexpert_labels"]
            dicom_id = batch["dicom"]
            all_chexpert_labels_corr.extend(chexpert_labels.numpy())

            inputs = vicuna_tokenizer.batch_encode_plus(text_input, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].cuda()
            generation_output = lang_model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256,
                dicom=dicom_id if args.use_embs else None,
                num_beams=args.num_beams,
            )

            preds = vicuna_tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)

            text_targets_corr.extend(text_target)
            text_inputs_corr.extend(text_input)
            dicom_ids_corr.extend(dicom_id)
            all_preds_corr.extend([p.split("ASSISTANT:")[-1].strip() if "KEEP_OLD" not in text_input[idx] else
                                   text_input[idx].split("</s>USER: KEEP_OLD")[0].split("ASSISTANT:")[-1].strip() for idx, p in enumerate(preds)])

        # save predictions
        pred_dir = Path("chexbert").absolute() / "outputs" / "predictions"
        with open(pred_dir / "predictions_{}_after_corrections.csv".format(exp_name), "w") as f:
            for i in range(len(all_preds_corr)):
                f.write('"' + all_preds_corr[i].replace('"', '') + '"\n')

        eval_preds_corr = [{"image": None, "caption": pred, "image_id": val_dataset.img_ids[dicom]} for pred, dicom in
                           zip(all_preds_corr, dicom_ids_corr)]
        bleu1_score, bleu2_score, bleu3_score, bleu4_score, meteor_score, rouge_score = compute_metrics(eval_preds_corr, evaluator)

        # chexpert score
        # save results to txt file
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        # run chexpert labeler
        # del lang_model
        torch.cuda.empty_cache()

        run_chexbert_labeler(reports_path=pred_dir / "predictions_{}_after_corrections.csv".format(exp_name),
                             output_path=pred_dir / "labels_{}_after_corrections.csv".format(exp_name))

        # read chexpert labels from file
        cp_pred = pd.read_csv(pred_dir / "labels_{}_after_corrections.csv".format(exp_name))
        pred_labels = np.array(cp_pred[val_dataset.chexpert_cols].values)
        all_chexpert_labels = np.array(all_chexpert_labels_corr)

        # Map present (1) cases to 1 and absent (0, was NaN) and uncertain (-1) cases to 0
        all_chexpert_labels = np.nan_to_num(all_chexpert_labels, nan=0)
        pred_labels = np.nan_to_num(pred_labels, nan=0)
        all_chexpert_labels[all_chexpert_labels == -1] = 0
        pred_labels[pred_labels == -1] = 0

        # Calculate F1 score
        mean_f1 = f1_score(all_chexpert_labels, pred_labels, average="macro")
        mean_prec = precision_score(all_chexpert_labels, pred_labels, average="macro")
        mean_rec = recall_score(all_chexpert_labels, pred_labels, average="macro")
        sample_f1 = f1_score(all_chexpert_labels, pred_labels, average="samples")

        print("Macro F1 Score:", mean_f1)
        print("Sample F1 Score:", sample_f1)

        # Calculate Accuracy
        acc_scores = []
        for i in range(all_chexpert_labels.shape[1]):
            acc = accuracy_score(all_chexpert_labels[:, i], pred_labels[:, i])
            acc_scores.append(acc)

        mean_acc = np.mean(acc_scores)
        # print(acc_scores)
        print("Mean Accuracy:", mean_acc)

        # save results to file
        with open(f'vicuna_results/results_{exp_name}_after_corrections.txt', 'w') as f:
            f.write(f"Prompt: {text_input[0]}\n")
            f.write(f"Avg Bleu 1: {bleu1_score}\n")
            f.write(f"Avg Bleu 2: {bleu2_score}\n")
            f.write(f"Avg Bleu 3: {bleu3_score}\n")
            f.write(f"Avg Bleu 4: {bleu4_score}\n")
            f.write(f"Avg Meteor: {meteor_score}\n")
            f.write(f"Avg Rouge: {rouge_score}\n")
            f.write(f"Mean Chexpert F1: {mean_f1}\n")
            f.write(f"Mean Chexpert Precision: {mean_prec}\n")
            f.write(f"Mean Chexpert Recall: {mean_rec}\n")
            f.write(f"Sample Chexpert F1: {sample_f1}\n")
            f.write(f"Mean Chexpert Accuracy: {mean_acc}\n")

    '''
    CheXpert Label Prediction
    '''
    if args.do_cp_bin_qa:
        chexpert_prompts = get_chexpert_prompts_bin(preds_history, val_dataset.chexpert_cols)
        batchsize = 1
        data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers)

        chexpert_preds = []
        for idx, batch in tqdm(enumerate(data_loader)):
            text_input = chexpert_prompts[idx]
            chexpert_labels = batch["chexpert_labels"]
            dicom_id = batch["dicom"]
            inputs = vicuna_tokenizer.batch_encode_plus(text_input, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].cuda()
            generation_output = lang_model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=10,
                dicom=dicom_id if args.use_embs else None,
            )

            preds = vicuna_tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)

            chexpert_preds.append([1 if "yes" in p.split("ASSISTANT:")[-1].lower() else 0 for idx, p in enumerate(preds)])

        relevant_cols = [c for c in val_dataset.chexpert_cols if c not in ["No Finding"]]
        relevant_cols_idx = [val_dataset.chexpert_cols.index(c) for c in relevant_cols]
        no_findings_idx = val_dataset.chexpert_cols.index("No Finding")
        any_findings = np.array(chexpert_preds)[:, relevant_cols_idx].sum(axis=1)
        any_findings[any_findings > 0] = 1
        # invert
        no_findings = 1 - any_findings
        # compare to ground truth
        chexpert_preds = np.array(chexpert_preds)
        chexpert_preds[:, no_findings_idx] = no_findings
        chexpert_preds = np.nan_to_num(chexpert_preds, nan=0)
        all_chexpert_labels[all_chexpert_labels == -1] = 0

        # Calculate F1 score
        mean_f1 = f1_score(all_chexpert_labels, chexpert_preds, average="macro")
        mean_prec = precision_score(all_chexpert_labels, chexpert_preds, average="macro")
        mean_rec = recall_score(all_chexpert_labels, chexpert_preds, average="macro")
        try:
            auc = roc_auc_score(all_chexpert_labels, chexpert_preds, average="macro")
        except ValueError:
            auc = -1
        acc = accuracy_score(all_chexpert_labels.flatten(), chexpert_preds.flatten())

        print("Macro F1 Score:", mean_f1)
        print("Macro AUC Score:", auc)
        print("Macro Precision Score:", mean_prec)
        print("Macro Recall Score:", mean_rec)
        print("Accuracy Score:", acc)

        # save results to file
        with open(f'vicuna_results/results_{exp_name}_after_cp_bin_qa.txt', 'w') as f:
            f.write(f"Prompt: {text_input[0]}\n")
            f.write(f"Mean Chexpert F1: {mean_f1}\n")
            f.write(f"Mean Chexpert Precision: {mean_prec}\n")
            f.write(f"Mean Chexpert Recall: {mean_rec}\n")
            f.write(f"Mean Chexpert Accuracy: {acc}\n")
            f.write(f"Mean Chexpert AUC: {auc}\n")

    if args.do_cp_all_qa:
        chexpert_prompts = get_chexpert_prompts_all(preds_history, val_dataset.chexpert_cols)
        batchsize = 5
        data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers)

        chexpert_preds = []
        for idx, batch in tqdm(enumerate(data_loader)):
            text_input = [chexpert_prompts[i] for i in range(batchsize * idx, min(batchsize * (idx + 1), len(chexpert_prompts)))]
            text_target = batch["text_target"]
            chexpert_labels = batch["chexpert_labels"]
            dicom_id = batch["dicom"]
            inputs = vicuna_tokenizer.batch_encode_plus(text_input, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].cuda()
            generation_output = lang_model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=30,
                dicom=dicom_id if args.use_embs else None,
                num_beams=args.num_beams
            )

            preds = vicuna_tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
            preds = [p.split("ASSISTANT:")[-1].lower() for p in preds]

            # iterate through all chexpert labels and check if they are in finding preds
            finding_preds_cleaned = []
            for finding_pred in preds:
                finding_pred_cleaned = []
                for label in val_dataset.chexpert_cols:
                    if label.lower() in finding_pred:
                        finding_pred_cleaned.append(label.lower())
                # convert to one-hot
                finding_pred_cleaned = [1 if c.lower() in finding_pred_cleaned else 0 for c in val_dataset.chexpert_cols]
                finding_preds_cleaned.append(finding_pred_cleaned)
            chexpert_preds.extend(finding_preds_cleaned)

        # compare to ground truth
        chexpert_preds = np.array(chexpert_preds)
        chexpert_preds = np.nan_to_num(chexpert_preds, nan=0)
        all_chexpert_labels[all_chexpert_labels == -1] = 0

        # Calculate F1 score
        mean_f1 = f1_score(all_chexpert_labels, chexpert_preds, average="macro")
        mean_prec = precision_score(all_chexpert_labels, chexpert_preds, average="macro")
        mean_rec = recall_score(all_chexpert_labels, chexpert_preds, average="macro")
        try:
            auc = roc_auc_score(all_chexpert_labels, chexpert_preds, average="macro")
        except ValueError:
            auc = -1
        acc = accuracy_score(all_chexpert_labels.flatten(), chexpert_preds.flatten())

        print("Macro F1 Score:", mean_f1)
        print("Macro AUC Score:", auc)
        print("Macro Precision Score:", mean_prec)
        print("Macro Recall Score:", mean_rec)
        print("Accuracy Score:", acc)

        with open(f'vicuna_results/results_{exp_name}_after_cp_all_qa.txt', 'w') as f:
            f.write(f"Prompt: {text_input[0]}\n")
            f.write(f"Mean Chexpert F1: {mean_f1}\n")
            f.write(f"Mean Chexpert Precision: {mean_prec}\n")
            f.write(f"Mean Chexpert Recall: {mean_rec}\n")
            f.write(f"Mean Chexpert Accuracy: {acc}\n")
            f.write(f"Mean Chexpert AUC: {auc}\n")
