from typing import Any, List

import torch
from pytorch_lightning import LightningModule

from src.models.components.sa_eend import SAEEND
from src.utils.loss import batch_pit_loss, report_diarization_error


class SAEENDModule(LightningModule):
    def __init__(self, net: torch.nn.Module, lr: float = 1e-3, weight_decay: float = 0.0005):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.net = net

    def foward(self, ys: torch.Tensor):
        return self.net(ys)

    def step(self, batch: Any):
        ys, ts, ilens = batch
        preds = self.foward(ys)
        loss, labels, sigmas = batch_pit_loss(preds, ts, ilens)
        return loss, preds, ts, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, labels = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, labels = self.step(batch)
        stats, der = report_diarization_error(preds, labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/der", der, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "der": der, "stats": stats}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, labels = self.step(batch)
        stats, der = report_diarization_error(preds, labels)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/der", der, on_step=False, on_epoch=True)

        return {"loss": loss, "der": der, "stats": stats}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
