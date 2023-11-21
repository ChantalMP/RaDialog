import collections
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, transforms

from model.lavis.data.ReportDataset import ExpandChannels

from local_config import VIS_ROOT, PATH_TO_MIMIC_CXR

class Chexpert_Dataset(Dataset):
    def __init__(self, split='train', truncate=None, loss_weighting="none", use_augs=False):

        super().__init__()

        # load csv file
        self.split = pd.read_csv(f'{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv')
        self.reports = pd.read_csv('mimic-cxr/reports_processed/mimic_cxr_sectioned.csv')
        self.reports = self.reports.dropna(subset=['findings'])

        self.vis_root = VIS_ROOT
        self.img_ids = {img_id: i for i, img_id in enumerate(self.reports['dicom_id'])}
        self.split_ids = set(self.split.loc[self.split['split'] == split]['dicom_id'])
        self.chexpert = pd.read_csv(f'data/data_files/finding_chexbert_labels.csv')
        self.chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]

        # get all dicom_ids where "split" is split
        self.annotation = self.reports.loc[self.reports['dicom_id'].isin(self.split_ids)]
        self.annotation['study_id'] = self.annotation['Note_file'].apply(lambda x: int(x.lstrip('s').rstrip('.txt')))
        # merge chexpert labels
        self.annotation = pd.merge(self.annotation, self.chexpert, how='left', left_on=['dicom_id'], right_on=['dicom_id'])
        if truncate is not None:
            self.annotation = self.annotation[:truncate]

        self.vis_transforms = Compose([Resize(512), CenterCrop(488), ToTensor(), ExpandChannels()])
        if use_augs:
            aug_tfm = transforms.Compose([transforms.RandomAffine(degrees=30, shear=15),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2)])

            self.vis_transforms = transforms.Compose([self.vis_transforms, aug_tfm])
        self.loss_weighting = loss_weighting

    def get_class_weights(self):
        """Compute class weights based on the inverse of class frequencies.

        Returns:
            Dict[str, float]: Class weights.
        """
        if self.loss_weighting == "none":
            return torch.ones(len(self.chexpert_cols), dtype=torch.float32)

        label_counts = torch.zeros(len(self.chexpert_cols), dtype=torch.float32)
        # iterate over dataframe getting rows
        for _, ann in self.annotation.iterrows():
            chexpert_labels = self._extract_chexpert_labels_from_row(ann)
            label_counts += chexpert_labels

        # Compute class weights
        if self.loss_weighting == "lin":
            class_weights = len(self.annotation) / label_counts
        elif self.loss_weighting == "log":
            class_weights = torch.log(len(self.annotation) / label_counts)

        return class_weights

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

    def _extract_chexpert_labels_from_row(self, row: pd.Series) -> torch.Tensor:
        labels = torch.zeros(len(self.chexpert_cols), dtype=torch.float32)
        for i, col in enumerate(self.chexpert_cols):
            if row[col] == 1:
                labels[i] = 1
        return labels

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        image_path = os.path.join(self.vis_root, ann["Img_Folder"], ann["Img_Filename"])
        image = self.load_image(Path(image_path))
        image = self.vis_transforms(image)
        chexpert_labels = self._extract_chexpert_labels_from_row(ann)

        return {
            "image": image,
            "labels": chexpert_labels,
            "image_id": self.img_ids[ann["dicom_id"]],
            "report": ann["findings"],
            "study_id": ann["study_id"],
            "dicom_id": ann["dicom_id"],
        }

    def __len__(self):
        return len(self.annotation)


if __name__ == '__main__':
    dataset = Chexpert_Dataset()
    print(dataset[0])
