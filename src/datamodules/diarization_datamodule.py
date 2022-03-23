from typing import Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.components.diarization_dataset import (
    DiarizationDataset,
    DiarizationDatasetforInfer,
)


def collate_fn(batch):
    ys, ts, ilens = list(zip(*batch))
    ilens = np.array(ilens)
    ys = np.array(
        [
            np.pad(y, [(0, np.max(ilens) - len(y)), (0, 0)], "constant", constant_values=(-1,))
            for y in ys
        ]
    )
    ts = np.array(
        [
            np.pad(t, [(0, np.max(ilens) - len(t)), (0, 0)], "constant", constant_values=(+1,))
            for t in ts
        ]
    )
    ys = torch.from_numpy(np.array(ys)).to(torch.float32)
    ts = torch.from_numpy(np.array(ts)).to(torch.float32)
    ilens = torch.from_numpy(np.array(ilens)).to(torch.int32)
    return ys, ts, ilens


class DiarizationDataModule(LightningDataModule):
    def __init__(
        self,
        data_dirs: Tuple[str, str, str],
        chunk_size: int = 2000,
        context_size: int = 7,
        frame_size: int = 1024,
        frame_shift: int = 256,
        subsampling: int = 10,
        sample_rate: int = 8000,
        input_transform: str = "logmel23_mn",
        n_speakers: int = None,
        batch_sizes: Tuple[int, int, int] = (64, 64, 1),
        num_workers: int = 0,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        if not self.data_train and not self.data_val and not self.data_test:
            train_dir, val_dir, test_dir = self.hparams.data_dirs
            self.data_train = DiarizationDataset(
                data_dir=train_dir,
                chunk_size=self.hparams.chunk_size,
                context_size=self.hparams.context_size,
                frame_size=self.hparams.frame_size,
                frame_shift=self.hparams.frame_shift,
                subsampling=self.hparams.subsampling,
                sample_rate=self.hparams.sample_rate,
                input_transform=self.hparams.input_transform,
                n_speakers=self.hparams.n_speakers,
            )
            self.data_val = DiarizationDataset(
                data_dir=val_dir,
                chunk_size=self.hparams.chunk_size,
                context_size=self.hparams.context_size,
                frame_size=self.hparams.frame_size,
                frame_shift=self.hparams.frame_shift,
                subsampling=self.hparams.subsampling,
                sample_rate=self.hparams.sample_rate,
                input_transform=self.hparams.input_transform,
                n_speakers=self.hparams.n_speakers,
            )
            self.data_test = DiarizationDatasetforInfer(
                data_dir=test_dir,
                chunk_size=self.hparams.chunk_size,
                context_size=self.hparams.context_size,
                frame_size=self.hparams.frame_size,
                frame_shift=self.hparams.frame_shift,
                subsampling=self.hparams.subsampling,
                sample_rate=self.hparams.sample_rate,
                input_transform=self.hparams.input_transform,
                n_speakers=self.hparams.n_speakers,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_sizes[0],
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_sizes[1],
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_sizes[2],
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
