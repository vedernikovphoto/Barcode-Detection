from pathlib import Path
from typing import Optional, Tuple, List

import torch
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.augmentations import get_transforms, TransformFlags
from src.config import Config
from src.dataset import DatasetBarcode, DatasetConfig
from src.dataset_splitter import stratify_shuffle_split_subsets


class BarcodeDM(LightningDataModule):
    def __init__(self, config: Config) -> None:
        """
        Initialize the BarcodeDM data module.

        Args:
            config (Config): Configuration object containing parameters for the data module.
        """
        super().__init__()
        self._config = config
        self._augmentation_params = config.augmentation_params
        self._images_folder = Path(self._config.data_config.data_path)

        self.train_dataset: Optional[DatasetBarcode] = None
        self.valid_dataset: Optional[DatasetBarcode] = None
        self.test_dataset: Optional[DatasetBarcode] = None
        self.batch_size = self._config.data_config.batch_size
        self.num_workers = self._config.data_config.n_workers

    def prepare_data(self) -> None:
        """
        Prepare data by splitting and saving train, validation, and test datasets.
        """
        split_and_save_datasets(
            self._config.data_config.data_path,
            self._config.seed,
            self._config.data_config.train_size,
        )

    def build_image_path(self, folder, file_name) -> str:
        """
        Constructs the full path to an image file.

        Args:
            folder (str): The folder containing the image file.
            file_name (str): The name of the image file.

        Returns:
            str: Full path to the image file.
        """
        return str(Path(folder) / file_name)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training, validation, and testing.

        Args:
            stage (Optional[str]): Stage of the setup ('fit', 'test', etc.).
        """
        if stage == 'fit' or stage is None:
            df_train = read_df(self._config.data_config.data_path, 'train')
            df_valid = read_df(self._config.data_config.data_path, 'valid')

            image_paths_train = [self.build_image_path(self._images_folder, x) for x in df_train['filename']]
            labels_train = np.zeros(len(df_train), dtype=np.int64)
            boxes_train = df_train[['x_from', 'y_from', 'width', 'height']].values

            image_paths_valid = [self.build_image_path(self._images_folder, x) for x in df_valid['filename']]
            labels_valid = np.zeros(len(df_valid), dtype=np.int64)
            boxes_valid = df_valid[['x_from', 'y_from', 'width', 'height']].values

            train_transforms = get_transforms(
                aug_config=self._augmentation_params,
                width=self._config.data_config.width,
                height=self._config.data_config.height,
                flags=TransformFlags(preprocessing=True, augmentations=True, postprocessing=True),
            )
            val_transforms = get_transforms(
                aug_config=self._augmentation_params,
                width=self._config.data_config.width,
                height=self._config.data_config.height,
                flags=TransformFlags(preprocessing=True, augmentations=False, postprocessing=True),
            )

            dataset_config_train = DatasetConfig(
                image_folder=str(self._images_folder),
                transforms=train_transforms,
            )
            dataset_config_val = DatasetConfig(
                image_folder=str(self._images_folder),
                transforms=val_transforms,
            )

            self.train_dataset = DatasetBarcode(image_paths_train, boxes_train, labels_train, dataset_config_train)
            self.valid_dataset = DatasetBarcode(image_paths_valid, boxes_valid, labels_valid, dataset_config_val)

        if stage == 'test' or stage is None:
            df_test = read_df(self._config.data_config.data_path, 'test')

            image_paths_test = [self.build_image_path(self._images_folder, x) for x in df_test['filename']]
            labels_test = np.zeros(len(df_test), dtype=np.int64)
            boxes_test = df_test[['x_from', 'y_from', 'width', 'height']].values

            test_transforms = get_transforms(
                aug_config=self._augmentation_params,
                width=self._config.data_config.width,
                height=self._config.data_config.height,
                flags=TransformFlags(preprocessing=True, augmentations=False, postprocessing=True),
            )

            dataset_config_test = DatasetConfig(
                image_folder=str(self._images_folder),
                transforms=test_transforms,
            )

            self.test_dataset = DatasetBarcode(image_paths_test, boxes_test, labels_test, dataset_config_test)

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, dict]]) -> Tuple[torch.Tensor, torch.Tensor]:   # noqa: WPS602, WPS221, E501
        """
        Custom collate function to process and batch images and targets.

        Args:
            batch (List[Tuple[torch.Tensor, dict]]): A batch of tuples containing images
                and corresponding target dictionaries.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Batched images and processed targets.
        """
        images = []
        targets = []
        for i, (image, target) in enumerate(batch):
            images.append(image)
            boxes = target['boxes']
            labels = target['labels']

            # Get image dimensions. Assuming image is (C, H, W)
            _, h, w = image.shape

            # Convert boxes to (x_center, y_center, width, height)
            x_min = boxes[:, 0]
            y_min = boxes[:, 1]
            x_max = boxes[:, 2]
            y_max = boxes[:, 3]
            x_center = (x_min + x_max) / 2.0    # noqa: WPS432
            y_center = (y_min + y_max) / 2.0    # noqa: WPS432
            width = x_max - x_min
            height = y_max - y_min

            # Convert boxes to (x_center, y_center, width, height)
            boxes_xywh = boxes.clone()
            boxes_xywh[:, 0] = x_center
            boxes_xywh[:, 1] = y_center
            boxes_xywh[:, 2] = width
            boxes_xywh[:, 3] = height

            # Normalize coordinates
            boxes_xywh[:, 0] /= w
            boxes_xywh[:, 1] /= h
            boxes_xywh[:, 2] /= w
            boxes_xywh[:, 3] /= h

            # Create target tensor
            num_boxes = boxes.shape[0]
            target_tensor = torch.zeros((num_boxes, 6))
            target_tensor[:, 0] = i  # Image index in batch
            target_tensor[:, 1] = labels
            target_tensor[:, 2:] = boxes_xywh

            targets.append(target_tensor)

        images = torch.stack(images)
        targets = torch.cat(targets, dim=0)

        return images, targets


def split_and_save_datasets(data_path: str, seed: int, train_fraction: float = 0.8) -> None:
    """
    Splits the dataset into train, validation, and test subsets, and saves them.

    Args:
        data_path (str): Path to the data directory.
        seed (int): Random seed for reproducibility.
        train_fraction (float): Proportion of data to allocate to training. Default is 0.8.
    """

    df = pd.read_csv(Path(data_path) / 'annotations.tsv', sep='\t')
    df = df.drop_duplicates()
    train_df, valid_df, test_df = stratify_shuffle_split_subsets(df, seed, train_fraction=train_fraction)

    train_df.to_csv(Path(data_path) / 'df_train.tsv', sep='\t', index=False)
    valid_df.to_csv(Path(data_path) / 'df_valid.tsv', sep='\t', index=False)
    test_df.to_csv(Path(data_path) / 'df_test.tsv', sep='\t', index=False)


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    """
    Reads a DataFrame from a TSV file.

    Args:
        data_path (str): Path to the data directory.
        mode (str): Mode of the data ('train', 'valid', 'test').

    Returns:
        pd.DataFrame: DataFrame read from the TSV file.
    """
    file_name = f'df_{mode}.tsv'
    file_path = Path(data_path) / file_name
    return pd.read_csv(file_path, sep='\t')
