import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import feature, kaldi_data


def _count_frames(data_length: int, size: int, step: int) -> int:
    return int((data_length - size + step) / step)


def _generate_frame_indices(data_length: int, size: int, step: int):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size

    if i * step + size < data_length:
        if data_length - (i + 1) * step > 0:
            if i == -1:
                yield (i + 1) * step, data_length
            else:
                yield data_length - size, data_length


def _generate_chunk_indices(data_len, chunk_size):
    step = chunk_size
    start = 0
    while start < data_len:
        end = min(data_len, start + chunk_size)
        yield start, end
        start += step


class DiarizationDataset(Dataset):
    def __init__(
        self,
        data_dir,
        chunk_size=2000,
        context_size=0,
        frame_size=1024,
        frame_shift=256,
        subsampling=1,
        sample_rate=8000,
        input_transform=None,
        n_speakers=None,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.sample_rate = sample_rate
        self.input_transform = input_transform
        self.n_speakers = n_speakers

        self.chunk_indices = []
        self.data = kaldi_data.KaldiData(self.data_dir)

        for rec in self.data.wavs:
            # data_length is depend on `sample_rate` and `subsampling`
            data_length = int(self.data.reco2dur[rec] * self.sample_rate / self.frame_shift)
            data_length = int(data_length / self.subsampling)

            # make chunk indices: rec_filepath, start_frame, end_frame 3-tuple
            for start, end in _generate_frame_indices(
                data_length=data_length, size=self.chunk_size, step=self.chunk_size
            ):
                self.chunk_indices.append((rec, start * self.subsampling, end * subsampling))

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, index: int):
        rec, start, end = self.chunk_indices[index]
        Y, T = feature.get_labeledSTFT(
            kaldi_obj=self.data,
            rec=rec,
            start=start,
            end=end,
            frame_size=self.frame_size,
            frame_shift=self.frame_shift,
            n_speakers=self.n_speakers,
        )
        # perform feature transforming
        Y = feature.transform(Y, self.input_transform)
        # append context
        Y_spliced = feature.splice(Y, self.context_size)
        # perform subsampling
        Y_ss, T_ss = feature.subsample(Y_spliced, T, self.subsampling)
        ilen = np.array(Y_ss.shape[0], dtype=np.int64)

        if self.n_speakers and T_ss.shape[1] > self.n_speakers:
            selected_speakers = np.argsort(T_ss.sum(axis=0))[::-1][: self.n_speakers]
            T_ss = T_ss[:, selected_speakers]

        return Y_ss, T_ss, ilen


class DiarizationDatasetforInfer(Dataset):
    def __init__(
        self,
        data_dir,
        chunk_size=2000,
        context_size=0,
        frame_size=1024,
        frame_shift=256,
        subsampling=1,
        sample_rate=8000,
        input_transform=None,
        n_speakers=None,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.sample_rate = sample_rate
        self.input_transform = input_transform
        self.n_speakers = n_speakers

        self.data = kaldi_data.KaldiData(self.data_dir)
        self.recs = []

        for rec in self.data.wavs:
            self.recs.append(rec)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, index):
        recid = self.recs[index]
        data, rate = self.data.load_wav(recid=recid)
        Y = feature.stft(data, self.frame_size, self.frame_shift)
        Y = feature.transform(Y, transform_type=self.input_transform)
        Y = feature.splice(Y, context_size=self.context_size)
        Y = Y[:: self.subsampling]
        return recid, Y
