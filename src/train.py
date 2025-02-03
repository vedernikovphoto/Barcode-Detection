import argparse
import torch
import logging
from pathlib import Path
import pytorch_lightning as pl
from clearml import Task
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from yolov5.export import run
from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import BarcodeDM
from src.lightning_module import BarcodeModule


logging.basicConfig(level=logging.INFO)


def save_yolo_compatible_checkpoint(model, save_path: str) -> None:
    """
    Saves the given PyTorch model in a format compatible with YOLOv5.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        save_path (str): The file path to save the YOLOv5-compatible checkpoint.

    Returns:
        None
    """
    torch.save({'model': model}, save_path)
    logging.info(f'Model saved in YOLOv5-compatible format at {save_path}')


def export_model_to_onnx(weights_path: str, config: Config, export_path: Path) -> None:
    """
    Exports a trained model to ONNX format using the YOLOv5 export script.

    Args:
        weights_path (str): Path to the YOLOv5-compatible model checkpoint.
        config (Config): Configuration object with model dimensions and data path.
        export_path (Path): The target file path for the ONNX model.

    Returns:
        None
    """
    run(
        weights=weights_path,
        imgsz=(config.data_config.width, config.data_config.height),
        device='cpu',
        simplify=False,
        data=config.data_config.data_config_path,
        include=['onnx'],
    )
    logging.info(f'Model exported to ONNX format at {export_path}')


def arg_parse() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments, including the configuration file path.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config_file')
    return parser.parse_args()


def prepare_task(config: Config) -> Task:
    """
    Initializes and connects a ClearML task for experiment tracking.

    Args:
        config (Config): Configuration object containing experiment settings.

    Returns:
        Task: A ClearML Task object initialized for the given configuration.
    """
    task = Task.init(
        project_name=config.project_name,
        task_name=f'{config.experiment_name}',
        auto_connect_frameworks=True,
    )
    task.connect(config.dict())
    return task


def train(config: Config) -> None:
    """
    Trains and evaluates the Barcode detection model.

    Args:
        config (Config): Configuration object containing all training parameters.
    """
    datamodule = BarcodeDM(config)
    model = BarcodeModule(config)
    task = prepare_task(config)   # noqa: F841

    experiment_save_path = Path(EXPERIMENTS_PATH) / config.experiment_name
    experiment_save_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}',
    )

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config.monitor_metric, patience=1000, mode=config.monitor_mode),
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)

    # Load best checkpoint to get the weights path for export
    best_model_path = checkpoint_callback.best_model_path

    # Save model in YOLOv5-compatible format
    yolo_compatible_checkpoint_path = experiment_save_path / 'best_model_yolov5.pt'
    best_model = BarcodeModule.load_from_checkpoint(best_model_path, config=config)
    save_yolo_compatible_checkpoint(best_model.get_core_model(), yolo_compatible_checkpoint_path)

    # Define export path and export to ONNX using YOLOv5 export script
    onnx_export_path = experiment_save_path / 'model.onnx'
    export_model_to_onnx(weights_path=str(yolo_compatible_checkpoint_path), config=config, export_path=onnx_export_path)


if __name__ == '__main__':
    args = arg_parse()
    config = Config.from_yaml(args.config_file)
    pl.seed_everything(config.seed, workers=True)
    train(config)
