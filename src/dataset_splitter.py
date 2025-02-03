from typing import List, Tuple

import numpy as np
import pandas as pd


def _prepare_data(annotation: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract image paths and bounding box annotations from the dataset.

    Args:
        annotation (pd.DataFrame): DataFrame containing image file names and bounding box coordinates.

    Returns:
        Tuple[np.ndarray, pd.DataFrame]:
        - Array of image file paths.
        - DataFrame of bounding box coordinates.
    """
    bbox_columns = ['x_from', 'y_from', 'width', 'height']

    all_image_paths = annotation['filename'].to_numpy()
    all_bbox = annotation[bbox_columns]

    return all_image_paths, all_bbox


def _create_subset(path_data, bbox_data, indices_data) -> pd.DataFrame:
    """
    Create a subset of the dataset using selected indices.

    Args:
        path_data (np.ndarray): Array of image file paths.
        bbox_data (pd.DataFrame): DataFrame containing bounding box coordinates.
        indices_data (np.ndarray): Array of indices to select the subset.

    Returns:
        pd.DataFrame: Subset DataFrame containing image file paths and bounding box coordinates.
    """
    path_data_selected = path_data[indices_data]
    bbox_data_selected = bbox_data.iloc[indices_data]

    dictionary_pass = {
        'filename': path_data_selected,
        'x_from': bbox_data_selected['x_from'].values,
        'y_from': bbox_data_selected['y_from'].values,
        'width': bbox_data_selected['width'].values,
        'height': bbox_data_selected['height'].values,
    }

    return pd.DataFrame(dictionary_pass)


def _split(
    paths: np.array,
    seed: int,
    distribution: List[float],
) -> Tuple[np.array, np.array]:
    """
    Split data into two subsets based on the specified distribution.

    Args:
        paths (np.ndarray): Array of image file paths.
        seed (int): Seed for reproducibility of the split.
        distribution (List[float]): Proportions for the two subsets (e.g., [0.8, 0.2]).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Indices for the first and second subsets.
    """
    np.random.seed(seed)
    indeces_array = np.arange(len(paths))
    np.random.shuffle(indeces_array)

    train_indices_size = int(distribution[0] * len(indeces_array))

    train_indices = indeces_array[:train_indices_size]
    val_indices = indeces_array[train_indices_size:]

    return train_indices, val_indices


def stratify_shuffle_split_subsets(
    annotation: pd.DataFrame,
    seed: int,
    train_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into stratified train, validation, and test subsets.

    Args:
        annotation (pd.DataFrame): DataFrame containing image paths and bounding boxes.
        seed (int): Seed for reproducibility of the splits.
        train_fraction (float): Fraction of data to allocate to the training set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        - Train subset as a DataFrame.
        - Validation subset as a DataFrame.
        - Test subset as a DataFrame.
    """
    all_image_paths, all_bbox = _prepare_data(annotation)
    train_indexes, else_indexes = _split(all_image_paths, seed, distribution=[train_fraction, 1 - train_fraction])  # noqa: WPS221, E501

    paths_else = all_image_paths[else_indexes]
    test_indexes, valid_indexes = _split(paths_else, seed, distribution=[0.5, 0.5])

    train_subset = _create_subset(all_image_paths, all_bbox, train_indexes)
    valid_subset = _create_subset(all_image_paths, all_bbox, valid_indexes)
    test_subset = _create_subset(all_image_paths, all_bbox, test_indexes)

    return train_subset, valid_subset, test_subset
