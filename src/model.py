import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics


class SegmentationModel(L.LightningModule):
    """Lightning Module for Segmentation Models using Segmentation Models PyTorch (SMP)"""

    def __init__(self, trainer_config):
        super().__init__()
        self.save_hyperparameters(trainer_config)

        self.init_model()
        self.init_loss()
        self.init_metrics()

    def init_model(self) -> None:
        """Initialize the model based on the provided hyperparameters."""
        self.model = smp.create_model(**self.hparams.model)

    def init_loss(self) -> None:
        """Initialize the loss function based on the provided hyperparameters."""
        loss_name = self.hparams["loss"]["loss_name"]
        if loss_name == "dice":
            loss_class = smp.losses.DiceLoss
        elif loss_name == "jaccard":
            loss_class = smp.losses.JaccardLoss
        elif loss_name == "ce":
            loss_class = nn.CrossEntropyLoss
        elif loss_name == "focal":
            loss_class = smp.losses.FocalLoss
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

        self.loss_fn = loss_class(**self.hparams["loss"]["loss_params"])

    def init_metrics(self) -> None:
        """Initialize metrics based on the provided hyperparameters."""
        metric_params = self.hparams["metrics"]["metric_params"]

        self.train_iou = torchmetrics.JaccardIndex(**metric_params)
        self.val_iou = torchmetrics.JaccardIndex(**metric_params)
        self.test_iou = torchmetrics.JaccardIndex(**metric_params)

        self.train_acc = torchmetrics.Accuracy(**metric_params, average="macro")
        self.val_acc = torchmetrics.Accuracy(**metric_params, average="macro")
        self.test_acc = torchmetrics.Accuracy(**metric_params, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Same as torch.nn.Module.forward"""
        return self.model(x)

    def _common_step(self, batch: tuple, batch_idx: int, stage: str) -> torch.Tensor:
        """Common step for training, validation, and testing."""
        images, masks = batch
        logits = self(images)

        loss = self.loss_fn(logits, masks)

        preds = torch.argmax(logits, dim=1)  # (N, H, W)

        if stage == "train":
            iou = self.train_iou(preds, masks)
            acc = self.train_acc(preds, masks)
            self.log(
                f"{stage}_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{stage}_iou",
                iou,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                f"{stage}_acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        elif stage == "val":
            iou = self.val_iou(preds, masks)
            acc = self.val_acc(preds, masks)
            self.log(
                f"{stage}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                f"{stage}_iou",
                iou,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                f"{stage}_acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        elif stage == "test":
            iou = self.test_iou(preds, masks)
            acc = self.test_acc(preds, masks)
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, logger=True)
            self.log(f"{stage}_iou", iou, on_step=False, on_epoch=True, logger=True)
            self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, logger=True)

        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step for the model."""
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step for the model."""
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Test step for the model."""
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self) -> optim.Optimizer | tuple:
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        optimizer_name = self.hparams["optimizer"]["optimizer_name"]
        if optimizer_name == "adam":
            optimizer_class = optim.Adam
        elif optimizer_name == "adamw":
            optimizer_class = optim.AdamW
        elif optimizer_name == "sgd":
            optimizer_class = optim.SGD
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")

        optimizer = optimizer_class(
            self.parameters(), **self.hparams["optimizer"]["optimizer_params"]
        )

        if self.hparams["scheduler"]["use_scheduler"]:
            scheduler_name = self.hparams["scheduler"]["scheduler_name"]
            if scheduler_name == "reducelronplateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **self.hparams["scheduler"]["scheduler_params"]
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": scheduler,
                    "monitor": self.hparams["scheduler"]["monitor"],
                }
            elif scheduler_name == "cosineannealinglr":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams["common"]["epochs"],
                    **self.hparams["scheduler"]["scheduler_params"],
                )
                return [optimizer], [scheduler]
            else:
                raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        else:
            return optimizer
