import pytorch_lightning as pl
import torch
import torch.nn as nn   # noqa: WPS301
from yolov5.models.yolo import Model
from yolov5.utils.loss import ComputeLoss
from pathlib import Path
from src.config import Config
from src.metrics import get_metrics
from src.train_utils import load_object
from src.processing import process_targets, process_outputs

BOXES_KEY = 'boxes'


class BarcodeModule(pl.LightningModule):
    """
    PyTorch Lightning module for the Barcode dataset.

    Attributes:
        config (Config): Configuration object.
        model (torch.nn.Module): YOLOv5 model for detection tasks.
        metrics (object): Object for computing evaluation metrics.
        loss_function (ComputeLoss): List of loss functions.
    """

    def __init__(self, config: Config):
        """
        Initializes the BarcodeModule.

        Args:
            config (Config): Configuration object containing model, optimizer, and scheduler settings.
        """
        super().__init__()
        self._config = config

        # Set up model paths
        current_dir = Path(__file__).resolve().parent.parent
        model_cfg = str(current_dir / 'yolov5' / 'models' / 'yolov5s.yaml')
        weights_path = str(current_dir / 'yolov5s_pre_trained_weights' / 'yolov5s.pt')

        # # Initialize YOLOv5 model
        self._model = Model(cfg=model_cfg, ch=3, nc=1)
        self._model.hyp = {
            'cls_pw': 1.0,
            'obj_pw': 1.0,
            'fl_gamma': 0.0,  # noqa: WPS358
            'anchor_t': 4.0,
            'label_smoothing': 0.0,  # noqa: WPS358
            'iou_t': 0.2,
            'box': 0.05,
            'obj': 1.0,
            'cls': 0.5,
        }

        # Load pre-trained weights if available
        if config.model_kwargs.get('pretrained_weights'):
            self._model.load_state_dict(torch.load(weights_path)['model'])

        # Initialize Mean Average Precision metric
        self.metrics = get_metrics()
        self._model.to('cuda')
        self.loss_function = ComputeLoss(self._model)

        self.save_hyperparameters(config.dict())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor representing images.

        Returns:
            torch.Tensor: Model outputs (predicted bounding boxes).
        """
        return self._model(x)

    def get_core_model(self) -> nn.Module:
        """
        Returns the core PyTorch model.

        Returns:
            nn.Module: The YOLOv5 model.
        """
        return self._model

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizers and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and scheduler configurations.
        """
        optimizer_class = load_object(self._config.optimizer)
        optimizer = optimizer_class(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )

        scheduler_class = load_object(self._config.scheduler)
        scheduler = scheduler_class(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Executes a single training step.

        Args:
            batch (tuple): A batch containing images and targets.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        images, targets = batch
        outputs = self._model(images)
        loss, loss_items = self.loss_function(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Executes a single validation step, including loss computation, metric updates, and visualization.

        Args:
            batch (tuple): A tuple containing images and targets.
            batch_idx (int): Index of the current validation batch.

        Returns:
            torch.Tensor: Validation loss for the batch.
        """
        images, targets = batch

        # Compute loss with model in train mode
        self._model.train()
        outputs = self._model(images)
        loss, loss_items = self.loss_function(outputs, targets)

        # Compute metrics with model in eval mode
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(images)

            # Process targets and outputs
            target_list = process_targets(targets, images)
            output_list = process_outputs(outputs, conf_thres=self._config.conf_thres, iou_thres=self._config.iou_thres)

            # Update metrics
            self.metrics.update(output_list, target_list)

        self.log('val_loss', loss, on_step=False, on_epoch=True)

        # Clear cache at the end of each validation batch
        torch.cuda.empty_cache()
        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Computes and logs validation metrics at the end of the validation epoch.

        Returns:
            None
        """
        # Compute and log the metrics
        metrics = self.metrics.compute()
        self.log('val_map', metrics['map'], on_step=False, on_epoch=True)
        self.metrics.reset()

        # Clear cache after logging metrics
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Executes a single test step, including loss computation and metric updates.

        Args:
            batch (tuple): A tuple containing images and targets.
            batch_idx (int): Index of the current test batch.

        Returns:
            torch.Tensor: Test loss for the batch.
        """
        images, targets = batch

        # Compute loss in train mode
        self._model.train()
        outputs = self._model(images)
        loss, _ = self.loss_function(outputs, targets)

        # Update metrics in eval mode
        self._model.eval()
        target_list = process_targets(targets, images)
        output_list = process_outputs(outputs, conf_thres=self._config.conf_thres, iou_thres=self._config.iou_thres)
        self.metrics.update(output_list, target_list)

        # Log test loss
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        Computes and logs test metrics at the end of the test epoch.

        Returns:
            None
        """
        metrics = self.metrics.compute()
        self.log('test_map', metrics['map'], on_step=False, on_epoch=True)
        self.metrics.reset()
