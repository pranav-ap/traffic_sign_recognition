import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from config import config
from .model import TrafficSignRecognitionModel

torch.set_float32_matmul_precision('medium')


class Light(pl.LightningModule):
    def __init__(self, neptune_logger=None, tensorboard_logger=None):
        super().__init__()

        self.neptune_logger = neptune_logger
        self.tensorboard_logger = tensorboard_logger

        self.model = TrafficSignRecognitionModel(num_classes=config.model.num_classes)

        self.learning_rate = config.train.learning_rate

        self.save_hyperparameters({
            'learning_rate': self.learning_rate,
        },
            ignore=[
                'model',
                'neptune_logger',
                'tensorboard_logger'
            ]
        )

    def forward(self, images):
        return self.model(images)

    @staticmethod
    def _compute_loss(logits, labels):
        return F.cross_entropy(logits, labels)

    @staticmethod
    @torch.no_grad()
    def _compute_accuracy(logits, labels):
        preds = torch.argmax(logits, dim=1)
        return (preds == labels).float().mean()

    def _shared_step(self, batch):
        images, labels = batch
        logits = self.model(images)

        loss = self._compute_loss(logits, labels)
        accuracy = self._compute_accuracy(logits, labels)

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._shared_step(batch)

        metrics = {
            "train/loss": loss,
            "train/accuracy": accuracy,
        }

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._shared_step(batch)

        metrics = {
            "val/loss": loss,
            "val/accuracy": accuracy,
        }

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        return metrics

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)

        loss = self._compute_loss(logits, labels)
        accuracy = self._compute_accuracy(logits, labels)

        metrics = {
            "test/loss": loss,
            "test/accuracy": accuracy,
        }

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=2,
                factor=0.5
            ),
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            patience=config.train.patience,
            mode="min",
            verbose=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val/loss',
            mode='min',
            dirpath=config.paths.output.checkpoints,
            filename="best_checkpoint",
            save_top_k=1,
            save_last=True,
        )

        progress_bar_callback = TQDMProgressBar(refresh_rate=2)
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

        return [
            early_stop_callback,
            checkpoint_callback,
            progress_bar_callback,
            lr_monitor_callback
        ]
