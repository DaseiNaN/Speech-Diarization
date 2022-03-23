# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

from itertools import permutations

import numpy as np
import torch
import torch.nn.functional as F

"""
P: number of permutation
T: number of frames
C: number of speakers (classes)
B: mini-batch size
"""


def pit_loss(pred, label):
    """Permutation-invariant training (PIT) cross entropy loss function.

    Args:
      pred:  (T,C)-shaped pre-activation values
      label: (T,C)-shaped labels in {0,1}

    Returns:
      min_loss: (1,)-shape mean cross entropy
      label_perms[min_index]: permutated labels
      sigma: permutation
    """

    T = len(label)
    C = label.shape[-1]
    label_perms_indices = [list(p) for p in permutations(range(C))]
    P = len(label_perms_indices)
    perm_mat = torch.zeros(P, T, C, C)

    for i, p in enumerate(label_perms_indices):
        perm_mat[i, :, torch.arange(label.shape[-1]), p] = 1

    x = torch.unsqueeze(torch.unsqueeze(label, 0), -1)
    y = torch.arange(P * T * C).view(P, T, C, 1)

    broadcast_label = torch.broadcast_tensors(x, y)[0]
    allperm_label = torch.matmul(perm_mat, broadcast_label).squeeze(-1)

    x = torch.unsqueeze(pred, 0)
    y = torch.arange(P * T).view(P, T, 1)
    broadcast_pred = torch.broadcast_tensors(x, y)[0]

    # broadcast_pred: (P, T, C)
    # allperm_label: (P, T, C)
    losses = F.binary_cross_entropy_with_logits(broadcast_pred, allperm_label, reduction="none")
    mean_losses = torch.mean(torch.mean(losses, dim=1), dim=1)
    min_loss = torch.min(mean_losses) * len(label)
    min_index = torch.argmin(mean_losses)
    sigma = list(permutations(range(label.shape[-1])))[min_index]

    return min_loss, allperm_label[min_index], sigma


def batch_pit_loss(ys, ts, ilens=None):
    """PIT loss over mini-batch.

    Args:
      ys: B-length list of predictions
      ts: B-length list of labels

    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
      sigmas: B-length list of permutation
    """
    if ilens is None:
        ilens = [t.shape[0] for t in ts]

    loss_w_labels_w_sigmas = [
        pit_loss(y[:ilen, :], t[:ilen, :]) for (y, t, ilen) in zip(ys, ts, ilens)
    ]
    losses, labels, sigmas = zip(*loss_w_labels_w_sigmas)
    loss = torch.sum(torch.stack(losses))
    n_frames = np.sum([ilen for ilen in ilens])
    loss = loss / n_frames

    return loss, labels, sigmas


def calc_diarization_error(pred, label, label_delay=0):
    """Calculates diarization error stats for reporting.

    Args:
      pred (torch.FloatTensor): (T,C)-shaped pre-activation values
      label (torch.FloatTensor): (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
           pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      res: dict of diarization error stats
    """
    label = label[: len(label) - label_delay, ...]
    decisions = torch.sigmoid(pred[label_delay:, ...]) > 0.5
    n_ref = label.sum(axis=-1).long()
    n_sys = decisions.sum(axis=-1).long()
    res = {}
    res["speech_scored"] = (n_ref > 0).sum()
    res["speech_miss"] = ((n_ref > 0) & (n_sys == 0)).sum()
    res["speech_falarm"] = ((n_ref == 0) & (n_sys > 0)).sum()
    res["speaker_scored"] = (n_ref).sum()
    res["speaker_miss"] = torch.max((n_ref - n_sys), torch.zeros_like(n_ref)).sum()
    res["speaker_falarm"] = torch.max((n_sys - n_ref), torch.zeros_like(n_ref)).sum()
    n_map = ((label == 1) & (decisions == 1)).sum(axis=-1)
    res["speaker_error"] = (torch.min(n_ref, n_sys) - n_map).sum()
    res["correct"] = (label == decisions).sum() / label.shape[1]
    res["diarization_error"] = res["speaker_miss"] + res["speaker_falarm"] + res["speaker_error"]
    res["frames"] = len(label)
    return res


def report_diarization_error(ys, labels):
    """Reports diarization errors Should be called with torch.no_grad.

    Args:
      ys: B-length list of predictions (torch.FloatTensor)
      labels: B-length list of labels (torch.FloatTensor)
    """
    stats_avg = {}
    cnt = 0
    for y, t in zip(ys, labels):
        stats = calc_diarization_error(y, t)
        for k, v in stats.items():
            stats_avg[k] = stats_avg.get(k, 0) + float(v)
        cnt += 1

    stats_avg = {k: v / cnt for k, v in stats_avg.items()}

    for k in stats_avg.keys():
        stats_avg[k] = round(stats_avg[k], 2)

    der = stats_avg["diarization_error"] / stats_avg["speaker_scored"] * 100
    return stats_avg, der
