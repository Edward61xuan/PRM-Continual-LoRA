"""
Continual learning evaluation metrics.

Standard metrics for class-incremental continual learning:
  - AAA (Average Accuracy Across tasks):  mean of per-task accuracies
  - BWT (Backward Transfer):              mean drop on old tasks after learning new ones
  - FWT (Forward Transfer):               mean performance on future tasks before learning them
  - Forgetting:                            max accuracy minus current accuracy on past tasks
"""

import numpy as np
import torch
from torch import Tensor


def calc_accuracy(preds, target, topk=(1,)):
    """Compute top-k accuracy.

    Args:
        preds:  Logits (N, C) or predicted labels (N,) or (N, k).
        target: Ground truth labels (N,).
        topk:   Tuple of k values.

    Returns:
        List of accuracy values, one per k.
    """
    assert target.ndim == 1
    assert preds.shape[0] == target.shape[0]
    maxk = max(topk)
    res = []

    if preds.dtype == torch.long:
        if preds.ndim == 1:
            _pred = preds.unsqueeze(1)
        else:
            _pred = preds[:, :maxk]
    elif preds.dtype in (torch.float16, torch.float32):
        _, _pred = preds.topk(maxk, 1, True, True)
    else:
        raise TypeError(f"Unsupported dtype: {preds.dtype}")

    _pred = _pred.t()
    correct = _pred.eq(target.view(1, -1).expand_as(_pred))

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
        res.append(correct_k / target.shape[0])
    return res


def calc_forgetting(acc_mat, current_taskid):
    """Compute forgetting: mean of (max_acc - current_acc) over past tasks.

    Args:
        acc_mat:         (T, T) accuracy matrix where acc_mat[i, j] = acc on task j after training task i.
        current_taskid:  Index of the most recently trained task.

    Returns:
        float: Average forgetting over past tasks.
    """
    if current_taskid == 0:
        return 0.0
    max_acc = np.amax(acc_mat, axis=0)[:current_taskid]
    cur_acc = acc_mat[current_taskid][:current_taskid]
    return float(np.mean(max_acc - cur_acc))


def calc_aaa(acc_list, current_taskid):
    """Average Accuracy Across tasks (AAA).

    AAA_t = (1/(t+1)) * sum_{k=0..t} Acc_k

    Args:
        acc_list:        (T,) array of average accuracies after each task.
        current_taskid:  Index of the most recently trained task.

    Returns:
        float: AAA metric.
    """
    return float(np.mean(acc_list[: current_taskid + 1]))


def calc_bwt(acc_mat, current_taskid):
    """Backward Transfer (BWT).

    BWT_t = (1/t) * sum_{i=0..t-1} (A_{t,i} - A_{i,i})

    Positive BWT means learning new tasks improved old task performance.
    Negative BWT indicates forgetting.

    Args:
        acc_mat:         (T, T) accuracy matrix.
        current_taskid:  Index of the most recently trained task.

    Returns:
        float: BWT metric.
    """
    if current_taskid == 0:
        return 0.0
    row_t = acc_mat[current_taskid][:current_taskid]
    diag_prev = np.diag(acc_mat)[:current_taskid]
    return float(np.mean(row_t - diag_prev))


def calc_fwt(acc_mat, current_taskid):
    """Forward Transfer (FWT).

    FWT_t = (1/t) * sum_{i=1..t} A_{i-1,i}

    Measures how well the model performs on task i before training on it
    (using knowledge from previous tasks).

    Args:
        acc_mat:         (T, T) accuracy matrix.
        current_taskid:  Index of the most recently trained task.

    Returns:
        float: FWT metric.
    """
    if current_taskid == 0:
        return 0.0
    vals = [acc_mat[i - 1, i] for i in range(1, current_taskid + 1)]
    return float(np.mean(vals))
