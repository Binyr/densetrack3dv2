# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
import torch.nn.functional as F
from densetrack3d.models.model_utils import reduce_masked_mean
from einops import rearrange, repeat
from jaxtyping import Float, Int64
from torch import Tensor, nn

from densetrack3d.models.geometry_utils import reproject_2d3d

EPS = 1e-6


def rigid_instane_loss(traj_inst_ids, gt_uvd, pred_uvd, intrinsics, weight_offset=0.8, num_sampled_traj=256, use_var=False, use_d_weight=False):
    """
    inst_id: B,N
    gt_uvd: B,T,N,3
    pred_uvd: B,I,T,N,3
    intrinsic: B,T,3,3
    """
    B,I,T,N = pred_uvd.shape[:4]

    if num_sampled_traj is not None:
        sample_index = torch.randperm(N)[:num_sampled_traj]
        traj_inst_ids = traj_inst_ids[:, sample_index]
        gt_uvd = gt_uvd[:, :, sample_index]
        pred_uvd = pred_uvd[:, :, :, sample_index]
    # print(pred_uvd.shape)
    B,I,T,N = pred_uvd.shape[:4]
    # 1. reproject 2d to 3D
    gt_3d = reproject_2d3d(gt_uvd, intrinsics) # B, T, N, 3

    pred_uvd = rearrange(pred_uvd, "b i t n c -> (b i) t n c")
    _intrinsics = intrinsics.repeat(I, 1, 1, 1)
    pred_3d = reproject_2d3d(pred_uvd, _intrinsics) # (B I), T, N, 3
    pred_3d = rearrange(pred_3d, "(b i) t n c -> b i t n c", i=I)

    # 2. inst loss
    loss = 0
    for i in range(B):
        inst_id_set = torch.unique(traj_inst_ids[i])
        for _id in inst_id_set:
            gt_3d_inst = gt_3d[i, :, traj_inst_ids[i]==_id] # T, n, 3
            avg_depth_inst = gt_3d_inst[:, :, 2].mean()
            if avg_depth_inst < 0.1:
                avg_depth_inst = 0.1

            if gt_3d_inst.shape[1] < 4:
                continue

            gt_3d_inst_1 = gt_3d_inst.unsqueeze(2) # T, n, 1, 3
            gt_3d_inst_2 = gt_3d_inst.unsqueeze(1) # T, 1, n, 3
            gt_dist_matrix =  torch.linalg.norm(gt_3d_inst_1 - gt_3d_inst_2, dim=3) # T, n, n

            pred_3d_inst = pred_3d[i, :, :, traj_inst_ids[i]==_id]
            pred_3d_inst_1 = pred_3d_inst.unsqueeze(3) # I, T, n, 1, 3
            pred_3d_inst_2 = pred_3d_inst.unsqueeze(2) # I, T, 1, n, 3
            pred_dist_matrix =  torch.linalg.norm(pred_3d_inst_1 - pred_3d_inst_2, dim=4) # I, T, n, n

            _, n, _ = gt_dist_matrix.shape
            _loss = 0
            for j in range(I):
                i_weight = weight_offset ** (I - j - 1)
                if not use_var:
                    _loss += i_weight * (pred_dist_matrix[j] - gt_dist_matrix).abs().sum() / (T * n * (n - 1))
                else:
                    _loss += i_weight * (pred_dist_matrix[j] - pred_dist_matrix[j].mean(dim=0, keepdim=True)).abs().sum() / (T * n * (n - 1))
            
            if use_d_weight:
                _loss = _loss / avg_depth_inst / 10.
            
            loss += _loss

    return loss



def huber_loss(x: Float[Tensor, "*"], y: Float[Tensor, "*"], delta: float = 1.0) -> Float[Tensor, "*"]:
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).float()
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


def track_loss(
    prediction: Float[Tensor, "b i s n c"],
    gt: Float[Tensor, "b s n c"],
    valid: Float[Tensor, "b s n"] = None,
    has_batch_dim: bool = True,
    is_dense: bool = False,
    use_huber_loss: bool = False,
    weight_offset: float = 0.8,
    divide_n_repeat: bool = False,
    delta_huber_loss: float = 6.0,
) -> Float[Tensor, ""]:

    if not has_batch_dim:
        prediction = prediction.unsqueeze(0)
        gt = gt.unsqueeze(0)
        if valid is not None:
            valid = valid.unsqueeze(0)

    if is_dense:
        prediction = rearrange(prediction, "b i s c h w -> b i s (h w) c")
        gt = rearrange(gt, "b s c h w -> b s (h w) c")
        if valid is not None:
            valid = rearrange(valid, "b s h w -> b s (h w)")

    I = prediction.shape[1]

    track_loss = 0
    for i in range(I):
        i_weight = weight_offset ** (I - i - 1)

        # binyanrui
        if len(gt.shape) == 5: # this means gt is B I T N C
            gt_ = gt[:, i]
        else:
            gt_ = gt
        # binyanrui
        if use_huber_loss:
            i_loss = huber_loss(prediction[:, i], gt_, delta=delta_huber_loss)
        else:
            i_loss = (prediction[:, i] - gt_).abs()  # S, N, 2

        i_loss = torch.mean(i_loss, dim=-1)  # S, N

        if valid is not None:
            track_loss += i_weight * reduce_masked_mean(i_loss, valid)
        else:
            track_loss += i_weight * i_loss.mean()

    if divide_n_repeat:
        track_loss = track_loss / I

    return track_loss


def balanced_bce_loss(
    prediction: Float[Tensor, "b s n c"],
    gt: Float[Tensor, "b s n c"],
    valid: Float[Tensor, "b s n"] = None,
) -> Float[Tensor, ""]:

    pos = (gt > 0.95).float()
    neg = (gt < 0.05).float()

    label = pos * 2.0 - 1.0
    a = -label * prediction
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

    if valid is not None:
        pos = pos * valid
        neg = neg * valid

    pos_loss = reduce_masked_mean(loss, pos)
    neg_loss = reduce_masked_mean(loss, neg)

    balanced_loss = pos_loss + neg_loss

    return balanced_loss

def seq_balanced_bce_loss(
    prediction: Float[Tensor, "b i s n c"],
    gt: Float[Tensor, "b s n c"],
    valid: Float[Tensor, "b s n"] = None,
    weight_offset: float = 0.8,
    divide_n_repeat: bool = False,
) -> Float[Tensor, ""]:

    pos = (gt > 0.95).float()
    neg = (gt < 0.05).float()

    label = pos * 2.0 - 1.0

    I = prediction.shape[1]

    total_loss = 0

    for i in range(I):
        i_weight = weight_offset ** (I - i - 1)

        a = -label * prediction[:, i]
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

        if valid is not None:
            pos = pos * valid
            neg = neg * valid

        pos_loss = reduce_masked_mean(loss, pos)
        neg_loss = reduce_masked_mean(loss, neg)

        balanced_loss = pos_loss + neg_loss

        # total_loss = total_loss + i_weight * balanced_loss
        total_loss = total_loss + balanced_loss # following cotracker3, dont weight the loss

    if divide_n_repeat:
        total_loss = total_loss / I

    return total_loss


def bce_loss(
    prediction: Float[Tensor, "b s n c"],
    gt: Float[Tensor, "b s n c"],
    valid: Float[Tensor, "b s n"] = None,
) -> Float[Tensor, ""]:

    if valid is None:
        loss = F.binary_cross_entropy(prediction, gt)
    else:
        loss = F.binary_cross_entropy(prediction, gt, reduction="none")
        loss = reduce_masked_mean(loss, valid)

    return loss

def seq_bce_loss(
    prediction: Float[Tensor, "b i s n c"],
    gt: Float[Tensor, "b s n c"],
    valid: Float[Tensor, "b s n"] = None,
    divide_n_repeat: bool = False,
    has_batch_dim: bool = True,
) -> Float[Tensor, ""]:
    
    # breakpoint()
    if not has_batch_dim:
        prediction = prediction.unsqueeze(0)
        gt = gt.unsqueeze(0)
        if valid is not None:
            valid = valid.unsqueeze(0)

    I = prediction.shape[0]

    total_loss = 0

    for i in range(I):
        if valid is None:
            loss = F.binary_cross_entropy(prediction[:, i], gt)
        else:
            loss = F.binary_cross_entropy(prediction[:, i], gt, reduction="none")
            loss = reduce_masked_mean(loss, valid)

        total_loss = total_loss + loss

    if divide_n_repeat:
        total_loss = total_loss / I
    
    return total_loss


def confidence_loss(
    tracks: Float[Tensor, "b i s n c"],
    confidence: Float[Tensor, "b s n"],
    target_points: Float[Tensor, "b s n c"],
    visibility: Float[Tensor, "b s n"],
    valid: Float[Tensor, "b s n"] = None,
    expected_dist_thresh: float = 12.0,
    has_batch_dim: bool = True,
    is_dense: bool = False,
) -> Float[Tensor, ""]:
    """Loss for classifying if a point is within pixel threshold of its target."""
    # Points with an error larger than 12 pixels are likely to be useless; marking
    # them as occluded will actually improve Jaccard metrics and give
    # qualitatively better results.

    # if len(tracks.shape) == 5:
    #     B, I, S, N, C = tracks.shape
    # else:
    #     I, S, N, C = tracks.shape
    #     tracks = tracks.unsqueeze(0)
    #     confidence = confidence.unsqueeze(0)
    #     target_points = target_points.unsqueeze(0)
    #     visibility = visibility.unsqueeze(0)

    if not has_batch_dim:
        tracks = tracks.unsqueeze(0)
        confidence = confidence.unsqueeze(0)
        target_points = target_points.unsqueeze(0)
        visibility = visibility.unsqueeze(0)
        if valid is not None:
            valid = valid.unsqueeze(0)

    if is_dense:
        tracks = rearrange(tracks, "b i s c h w -> b i s (h w) c")
        target_points = rearrange(target_points, "b s c h w -> b s (h w) c")
        confidence = rearrange(confidence, "b s h w -> b s (h w)")
        visibility = rearrange(visibility, "b s h w -> b s (h w)")
        if valid is not None:
            valid = rearrange(valid, "b s h w -> b s (h w)")

    if not visibility.dtype == torch.bool:
        visibility = (visibility > 0.9).bool()

    err = torch.sum((tracks[:, -1].detach() - target_points) ** 2, dim=-1)
    conf_gt = (err <= expected_dist_thresh**2).float()
    logprob = F.binary_cross_entropy(confidence, conf_gt, reduction="none")
    logprob *= visibility.float()

    if valid is not None:
        logprob = reduce_masked_mean(logprob, valid)
    else:
        logprob = logprob.mean()

    return logprob

def seq_confidence_loss(
    tracks: Float[Tensor, "b i s n c"],
    confidence: Float[Tensor, "b i s n"],
    target_points: Float[Tensor, "b s n c"],
    visibility: Float[Tensor, "b s n"],
    valid: Float[Tensor, "b s n"] = None,
    expected_dist_thresh: float = 12.0,
    has_batch_dim: bool = True,
    is_dense: bool = False,
    divide_n_repeat: bool = False,
) -> Float[Tensor, ""]:
    """Loss for classifying if a point is within pixel threshold of its target."""
    # Points with an error larger than 12 pixels are likely to be useless; marking
    # them as occluded will actually improve Jaccard metrics and give
    # qualitatively better results.

    # if len(tracks.shape) == 5:
    #     B, I, S, N, C = tracks.shape
    # else:
    #     I, S, N, C = tracks.shape
    #     tracks = tracks.unsqueeze(0)
    #     confidence = confidence.unsqueeze(0)
    #     target_points = target_points.unsqueeze(0)
    #     visibility = visibility.unsqueeze(0)

    if not has_batch_dim:
        tracks = tracks.unsqueeze(0)
        confidence = confidence.unsqueeze(0)
        target_points = target_points.unsqueeze(0)
        visibility = visibility.unsqueeze(0)
        if valid is not None:
            valid = valid.unsqueeze(0)

    if is_dense:
        tracks = rearrange(tracks, "b i s c h w -> b i s (h w) c")
        target_points = rearrange(target_points, "b s c h w -> b s (h w) c")
        confidence = rearrange(confidence, "b i s h w -> b i s (h w)")
        visibility = rearrange(visibility, "b s h w -> b s (h w)")
        if valid is not None:
            valid = rearrange(valid, "b s h w -> b s (h w)")

    if not visibility.dtype == torch.bool:
        visibility = (visibility > 0.9).bool()

    I = confidence.shape[1]
    total_loss = 0

    for i in range(I):
        if I == 1: # only last conf
            err = torch.sum((tracks[:, -1].detach() - target_points) ** 2, dim=-1)
        else:
            err = torch.sum((tracks[:, i].detach() - target_points) ** 2, dim=-1)
            
        conf_gt = (err <= expected_dist_thresh**2).float()
        logprob = F.binary_cross_entropy(confidence[:, i], conf_gt, reduction="none")
        logprob *= visibility.float()

        if valid is not None:
            logprob = reduce_masked_mean(logprob, valid)
        else:
            logprob = logprob.mean()

        total_loss = total_loss + logprob

    if divide_n_repeat:
        total_loss = total_loss / I

    return logprob
