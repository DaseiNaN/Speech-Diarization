import os
from typing import Any, List

import h5py
import numpy as np
import torch
from pytorch_lightning import LightningModule

from src.datamodules.components.diarization_dataset import _generate_chunk_indices
from src.models.components.sa_eend import SAEEND
from src.utils.loss import batch_pit_loss, report_diarization_error


class SAEENDModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        infer_dir: str,
        lr: float = 1e-3,
        weight_decay: float = 0.0005,
        chunk_size: int = 500,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.net = net

    def foward(self, ys: torch.Tensor, activation=None):
        return self.net(ys, activation=activation)

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
        recid, Y = batch
        out_preds = []
        for start, end in _generate_chunk_indices(len(Y), self.hparams.chunk_size):
            ys = torch.from_numpy(np.array(Y[start:end])).unsqueeze(dim=0)
            preds = self.foward(ys, activation=torch.sigmoid)
            out_preds.append(preds[0].numpy())
        out_file_name = recid + ".h5"
        out_data = np.vstack(out_preds)
        out_path = os.path.join(self.hparams.infer_dir, out_file_name)
        with h5py.File(out_path, "w") as wf:
            wf.create_dataset("T_hat", data=out_data)

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
