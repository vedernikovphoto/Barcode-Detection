from omegaconf import OmegaConf
from pydantic import BaseModel


class DataConfig(BaseModel):
    """
    Configuration for the dataset.

    Attributes:
        data_path (str): Path to the data directory.
        batch_size (int): Number of samples per batch.
        n_workers (int): Number of worker threads for data loading.
        train_size (float): Proportion of the data to be used for training.
        width (int): Width to resize the images to.
        height (int): Height to resize the images to.
        input_size (tuple): Tuple representing the input size of the model (channels, height, width).
        data_config_path (str): Path to the YOLOv5 `data.yaml` configuration file for model export.
    """
    data_path: str
    batch_size: int
    n_workers: int
    train_size: float
    width: int
    height: int
    input_size: tuple
    data_config_path: str


class AugmentationConfig(BaseModel):
    """
    Configuration for data augmentation.

    Attributes:
        hue_shift_limit (int): Limit for hue shift.
        sat_shift_limit (int): Limit for saturation shift.
        val_shift_limit (int): Limit for value shift.
        brightness_limit (float): Limit for brightness adjustment.
        contrast_limit (float): Limit for contrast adjustment.
    """
    hue_shift_limit: int
    sat_shift_limit: int
    val_shift_limit: int
    brightness_limit: float
    contrast_limit: float


class Config(BaseModel):
    """
    Main configuration class for the project.

    Attributes:
        project_name (str): The name of the project.
        experiment_name (str): The name of the experiment.
        data_config (DataConfig): Configuration for the dataset.
        augmentation_params (AugmentationConfig): Configuration for data augmentation.
        n_epochs (int): Number of training epochs.
        accelerator (str): Type of accelerator to use (e.g., 'gpu').
        device (int): Device identifier.
        seed (int): Random seed for reproducibility.
        log_every_n_steps (int): Logging frequency in steps.
        monitor_metric (str): Metric to monitor for early stopping.
        monitor_mode (str): Mode for monitoring ('min' or 'max').
        model_kwargs (dict): Additional keyword arguments for the model.
        optimizer (str): Name of the optimizer.
        optimizer_kwargs (dict): Additional keyword arguments for the optimizer.
        scheduler (str): Name of the learning rate scheduler.
        scheduler_kwargs (dict): Additional keyword arguments for the scheduler.
        iou_thres (float): IoU threshold for Non-Maximum Suppression to filter overlapping detections.
        conf_thres (float): Confidence threshold to filter out low-confidence predictions.

    Methods:
        from_yaml(cls, path: str) -> 'Config': Load configuration from a YAML file.
    """
    project_name: str
    experiment_name: str
    data_config: DataConfig
    augmentation_params: AugmentationConfig
    n_epochs: int
    accelerator: str
    device: int
    seed: int
    log_every_n_steps: int
    monitor_metric: str
    monitor_mode: str
    model_kwargs: dict
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    iou_thres: float
    conf_thres: float

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """
        Load configuration from a YAML file.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            Config: Loaded configuration object.
        """
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
