import os
from typing import Any, List

import h5py
import numpy as np
import torch
from pytorch_lightning import LightningModule

from src.datamodules.components.diarization_dataset import _generate_chunk_indices
from src.models.components.sa_eend import SAEEND
from src.utils.loss import batch_pit_loss, report_diarization_error
from src.utils.noam_scheduler import NoamScheduler


class SAEENDModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        infer_dir: str,
        lr: float = 1.0,
        optimizer: str = "noam",
        noam_warmup_steps: int = 100000,
        gradient_accumulation_steps: int = 1,
        gradclip: int = 5,
        chunk_size: int = 500,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.net = net

        self.automatic_optimization = False

    def foward(self, ys: torch.Tensor, activation=None):
        return self.net(ys, activation=activation)

    def step(self, batch: Any):
        ys, ts, ilens = batch
        preds = self.foward(ys)
        loss, labels, sigmas = batch_pit_loss(preds, ts, ilens)
        return loss, preds, ts, labels

    def training_step(self, batch: Any, batch_idx: int):
        ys, ts, ilens = batch
        preds = self.foward(ys)

        opt = self.optimizers()
        if self.hparams.optimizer == "noam":
            sch = self.lr_schedulers()

        if (self.global_step + 1) % self.hparams.gradient_accumulation_steps == 0:
            opt.zero_grad()
            loss, labels, sigmas = batch_pit_loss(preds, ts, ilens)
            self.manual_backward(loss)
            opt.step()
            if self.hparams.optimizer == "noam":
                sch.step()
            if self.hparams.gradclip > 0:
                torch.nn.utils.clip_grad_value_(self.net.parameters(), self.hparams.gradclip)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/lr", opt.param_groups[0]["lr"], on_step=False, on_epoch=True, prog_bar=False
        )
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, labels = self.step(batch)
        stats, der = report_diarization_error(preds, labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/der", der, on_step=False, on_epoch=True, prog_bar=True)
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
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
            return optimizer
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.hparams.lr)
            return optimizer
        elif self.hparams.optimizer == "noam":
            # for noam, lr refers to base_lr (i.e. scale), suggest lr=1.0
            optimizer = torch.optim.Adam(
                self.net.parameters(), lr=self.hparams.lr, betas=(0.9, 0.98), eps=1e-9
            )
        else:
            raise ValueError(self.hparams.optimizer)

        if self.hparams.optimizer == "noam":
            scheduler = NoamScheduler(
                optimizer=optimizer,
                d_model=self.net.n_units,
                warmup_steps=self.hparams.noam_warmup_steps,
            )
        return [optimizer], [scheduler]
