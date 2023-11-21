import dataclasses
import json
import os
import re
import time
from enum import Enum, auto
from pathlib import Path
from typing import List, Any

from local_config import PATH_TO_MIMIC_CXR

# set java path
os.environ["JAVA_HOME"] = "/home/guests/chantal_pellegrini/java/jre1.8.0_361"
os.environ["PATH"] = "/home/guests/chantal_pellegrini/java/jre1.8.0_361/bin:" + os.environ["PATH"]
os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.getcwd(), "gradio_tmp")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from nltk import word_tokenize
from omegaconf import OmegaConf
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from skimage import io
from sklearn.metrics import classification_report, accuracy_score
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, transforms

from model.lavis.processors import BaseProcessor
from model.lavis.common.registry import registry
from model.lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from model.lavis.datasets.datasets.base_dataset import BaseDataset
from model.lavis.datasets.datasets.caption_datasets import __DisplMixin


@registry.register_processor("my_blip_caption")
class MyBlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([!\"()*#;~])",
            " ",
            caption,
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


class ExpandChannels:
    """
    Transforms an image with one channel to an image with three channels by copying
    pixel intensities of the image along the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Tensor of shape [1, H, W].
        :return: Tensor with channel copied three times, shape [3, H, W].
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)


def create_chest_xray_transform_for_inference(resize: int, center_crop_size: int) -> Compose:
    """
    Defines the image transformation pipeline for Chest-Xray datasets.

    :param resize: The size to resize the image to. Linear resampling is used.
                   Resizing is applied on the axis with smaller shape.
    :param center_crop_size: The size to center crop the image to. Square crop is applied.
    """

    transforms = [Resize(resize), CenterCrop(center_crop_size), ToTensor(), ExpandChannels()]
    return Compose(transforms)

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

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

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


class MIMIC_CXR_Dataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, split, cfg, ann_paths=[], truncate=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # load csv file
        self.split = pd.read_csv(f'{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv')
        self.cur_split = split
        self.reports = pd.read_csv('mimic-cxr/reports_processed/mimic_cxr_sectioned.csv')
        # drop reports where findings are nan
        self.reports = self.reports.dropna(subset=['findings'])

        self.use_pred_labels = True

        self.chexpert = pd.read_csv(f'data/data_files/finding_chexbert_labels.csv')
        self.chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]

        self.custom_epochs_per_epoch = 2 if split == 'train' and cfg.run_cfg.task != "image_text_pretrain_eval" and truncate==None else 1
        self.current_custom_epoch = 0
        self.vit_model = cfg.model_cfg['vit_model']
        self.img_size = cfg.datasets_cfg.mimic_cxr.vis_processor.train.image_size  # should be 224 for coco models, 448 for biovil models
        if self.vit_model == 'biovil':
            self.vis_transforms = create_chest_xray_transform_for_inference(512, center_crop_size=self.img_size)
            if split == 'train':
                self.vis_augs = transforms.Compose([transforms.RandomAffine(degrees=30, shear=15),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2)])

        self.img_ids = {img_id: i for i, img_id in enumerate(self.reports['dicom_id'])}
        self.id_to_dicom = {v: k for k, v in self.img_ids.items()}
        self.split_ids = set(self.split.loc[self.split['split'] == split]['dicom_id'])

        # get all dicom_ids where "split" is split
        self.annotation = self.reports.loc[self.reports['dicom_id'].isin(self.split_ids)]
        if truncate is not None:
            self.annotation = self.annotation[:truncate]

        self.annotation['findings'] = self.annotation['findings'].apply(lambda x: x.replace('\n', ''))

        # Extract patient_id from Img_Folder (3rd part) and study_id is the name of the notefile without the pre-pending 's'
        self.annotation['subject_id'] = self.annotation['Img_Folder'].apply(lambda x: int(x.split('/')[2].lstrip('p')))
        self.annotation['study_id'] = self.annotation['Note_file'].apply(lambda x: int(x.lstrip('s').rstrip('.txt')))

        # Merge chexpert labels with annotation dataframe
        self.annotation = pd.merge(self.annotation, self.chexpert, how='left', left_on=['dicom_id'], right_on=['dicom_id'])


        add_findings_in_prompt = cfg.run_cfg.get("add_findings_in_prompt", False)
        self.prompt = cfg.datasets_cfg.mimic_cxr.text_processor.train.prompt if split == 'train' \
            else cfg.datasets_cfg.mimic_cxr.text_processor.eval.prompt

        self.text_processor = MyReportProcessor(
            prompt=self.prompt, max_words=1000)

        self.evaluator = MIMICEvalCap(self.annotation, self.img_ids)

    def set_custom_epoch(self, custom_epoch):
        self.current_custom_epoch = custom_epoch

    def remap_to_uint8(self, array: np.ndarray, percentiles=None) -> np.ndarray:
        """Remap values in input so the output range is :math:`[0, 255]`.

        Percentiles can be used to specify the range of values to remap.
        This is useful to discard outliers in the input data.

        :param array: Input array.
        :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
            Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
        :returns: Array with ``0`` and ``255`` as minimum and maximum values.
        """
        array = array.astype(float)
        if percentiles is not None:
            len_percentiles = len(percentiles)
            if len_percentiles != 2:
                message = (
                    'The value for percentiles should be a sequence of length 2,'
                    f' but has length {len_percentiles}'
                )
                raise ValueError(message)
            a, b = percentiles
            if a >= b:
                raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
            if a < 0 or b > 100:
                raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
            cutoff: np.ndarray = np.percentile(array, percentiles)
            array = np.clip(array, *cutoff)
        array -= array.min()
        array /= array.max()
        array *= 255
        return array.astype(np.uint8)

    def load_image(self, path) -> Image.Image:
        """Load an image from disk.

        The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

        :param path: Path to image.
        :returns: Image as ``Pillow`` ``Image``.
        """
        # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
        if path.suffix in [".jpg", ".jpeg", ".png"]:
            image = io.imread(path)
        else:
            raise ValueError(f"Image type not supported, filename was: {path}")

        image = self.remap_to_uint8(image)
        return Image.fromarray(image).convert("L")


    def __getitem__(self, index):
        subset_size = len(self.annotation) // self.custom_epochs_per_epoch
        start_index = self.current_custom_epoch * subset_size
        actual_index = start_index + index

        ann = self.annotation.iloc[actual_index]

        image_path = os.path.join(self.vis_root, ann["Img_Folder"], ann["Img_Filename"])
        if self.vit_model == "biovil":  # old version worked with smaller img and without biovil img processing
            image = self.load_image(Path(image_path))
            image = self.vis_transforms(image)

        caption = ann["findings"].strip()
        input_text = self.text_processor(findings=None, no_findings=False)

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

        if "<IMG>" in prompt:
            before_img, after_img = prompt.split("<IMG>")
            prompt = (before_img, after_img)


        return {
            "image": image,
            "text_input": prompt,
            "text_output": caption,
            "image_id": self.img_ids[ann["dicom_id"]],
            # "index": index,
            # "string_index": str(index)
        }

    def __len__(self):
        return len(self.annotation) // self.custom_epochs_per_epoch


@registry.register_builder("mimic_cxr")
class MIMIC_CXR_Builder(BaseDatasetBuilder):
    train_dataset_cls = MIMIC_CXR_Dataset
    eval_dataset_cls = MIMIC_CXR_Dataset

    DATASET_CONFIG_DICT = {
        "default": "defaults_report.yaml"
    }


class MIMICEvalCap:
    def __init__(self, gts, img_id_map):

        self.gts = gts

        # invert img_id_map
        self.dicom_to_id = img_id_map
        self.id_to_dicom = {v: k for k, v in img_id_map.items()}

        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L")
        ]

        # if mimic_nle:
        #     self.scorers.extend([(Cider(), "CIDEr"),(Spice(), "SPICE"),])

    def preprocess(self, s):
        s = s.replace('\n', '')
        s = s.replace('<s>', '')
        s = s.replace('</s>', '')
        # s = s.translate(str.maketrans('', '', '0123456789'))
        # s = s.translate(str.maketrans('', '', string.punctuation))
        return s

    def evaluate(self, res):

        res = {self.id_to_dicom[elem["image_id"]]: elem["caption"] for elem in res}
        res_keys_set = set(res.keys())
        gts = {}
        gts_img_id = {}
        for _, elem in self.gts.iterrows():
            dicom_id = elem["dicom_id"]
            if dicom_id in res_keys_set:
                gts[dicom_id] = [elem["findings"]]
                gts_img_id[self.dicom_to_id[dicom_id]] = [elem["findings"]]

        # gts = {elem["dicom_id"]: [elem["findings"]] for _, elem in self.gts.iterrows() if elem["dicom_id"] in res.keys()}
        # gts_img_id = {self.dicom_to_id[elem["findings"]]: [elem["Note"]] for _, elem in self.gts.iterrows() if elem["dicom_id"] in res.keys()}
        assert res.keys() == gts.keys()
        # =================================================
        # Pre-process sentences
        # =================================================
        print('tokenization...')
        for dicom in res.keys():
            pred_text = ' '.join(word_tokenize(self.preprocess(res[dicom]))).lower()
            true_text = ' '.join(word_tokenize(self.preprocess(gts[dicom][0]))).lower()

            res[dicom] = [pred_text]
            gts[dicom] = [true_text]

        # =================================================
        # Compute scores
        # =================================================
        final_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    final_scores[m] = sc
                    #final_scores["elem_wise_" + str(m)] = scs
                    print("%s: %0.3f" % (m, sc))
            else:
                print("%s: %0.3f" % (method, score))
                #final_scores["elem_wise_" + str(method)] = scores
                final_scores[method] = score

        final_scores['agg_metrics'] = np.mean(list({k: v for k, v in final_scores.items() if "elem_wise" not in k}.values()))

        return final_scores, gts_img_id
