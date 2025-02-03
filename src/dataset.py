import torch
import cv2
import numpy as np
import albumentations as albu

from typing import Optional, List, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    """
    Configuration class for dataset parameters.

    Attributes:
        image_folder (str): Path to the folder containing images.
        transforms (Optional[albu.BaseCompose]): Augmentation and preprocessing transforms.
    """
    image_folder: str
    transforms: Optional[albu.BaseCompose] = None


class DatasetBarcode(Dataset):
    """
    Custom PyTorch Dataset for barcode detection.

    This dataset loads images, applies optional augmentations, and provides
    bounding boxes and labels as targets for training or evaluation.

    Attributes:
        image_folder (str): Path to the folder containing images.
        transforms (Optional[albu.BaseCompose]): Augmentation and preprocessing transforms.
        image_paths (List[str]): List of image file paths.
        boxes (np.ndarray): Array of bounding box coordinates for each image.
        labels (np.ndarray): Array of labels for each image.
    """

    def __init__(self, image_paths: List[str], boxes: np.ndarray, labels: np.ndarray, config: DatasetConfig) -> None:
        """
        Initializes the DatasetBarcode instance.

        Args:
            image_paths (List[str]): List of image file paths.
            boxes (np.ndarray): Bounding box coordinates in the format (x_min, y_min, width, height).
            labels (np.ndarray): Labels corresponding to each image.
            config (DatasetConfig): Configuration object containing dataset parameters.
        """
        self.image_folder = config.image_folder
        self.transforms = config.transforms
        self.image_paths = image_paths
        self.labels = labels
        self.boxes = boxes

    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict]:
        """
        Retrieves an image and its corresponding target by index.

        Args:
            idx (int): Index of the image and target to retrieve.

        Returns:
            tuple:
                - image (torch.Tensor): Transformed image tensor.
                - target (dict): Dictionary containing:
                    - 'boxes' (torch.Tensor): Tensor of bounding box coordinates.
                    - 'labels' (torch.Tensor): Tensor of labels.
        """
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x_min = self.boxes[idx, 0]
        y_min = self.boxes[idx, 1]
        width = self.boxes[idx, 2]
        height = self.boxes[idx, 3]
        x_max = x_min + width
        y_max = y_min + height
        boxes = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)

        labels = np.array([self.labels[idx]], dtype=np.int64)  # Get the label for this index

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = transformed['labels']

        image = torch.tensor(image, dtype=torch.float32)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
        }

        return image, target

    def __len__(self) -> int:
        """
        Gets the total number of images in the dataset.

        Returns:
            int: The total number of images.
        """
        return len(self.image_paths)
