import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from densetrack3d.datasets.cvo_dataset import CVO
from densetrack3d.datasets.kubric_dataset import KubricDataset
from densetrack3d.datasets.dr_dataset2 import DynamicReplicaDataset
from densetrack3d.datasets.mix_dataset import MixDataset
from densetrack3d.datasets.tapvid2d_dataset import TapVid2DDataset
from densetrack3d.datasets.utils import collate_fn, collate_fn_train, dataclass_to_cuda_
from densetrack3d.evaluation.core.evaluator import Evaluator

from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
# from densetrack3d.models.densetrack3d.densetrack3dv2 import DenseTrack3DV2
from densetrack3d.models.densetrack3d.densetrack3dv2_cycle_consist import DenseTrack3DV2
from densetrack3d.datasets.tapvid3d_dataset2 import TapVid3DDataset

from densetrack3d.models.evaluation_predictor.evaluation_predictor import EvaluationPredictor
from densetrack3d.models.loss import balanced_bce_loss, seq_balanced_bce_loss, bce_loss, seq_bce_loss, confidence_loss, seq_confidence_loss, track_loss, rigid_instane_loss
from densetrack3d.models.model_utils import (
    bilinear_sampler,
    dense_to_sparse_tracks_3d_in_3dspace,
    get_grid,
    get_points_on_a_grid,
)
from densetrack3d.models.optimizer import fetch_optimizer
from densetrack3d.utils.logger import Logger
from densetrack3d.utils.visualizer import Visualizer, flow_to_rgb
from einops import rearrange
# from pytorch_lightning.lite import LightningLite

# from torch.cuda.amp import GradScaler
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
import hydra
from omegaconf import OmegaConf
import pathlib
from pathlib import Path

from datetime import timedelta
from collections import defaultdict, OrderedDict

# from densetrack3d.utils.signal_utils import sig_handler, term_handler

TAPVID2D_DIR = None
# KUBRIC3D_MIX_DIR = "datasets/kubric/movif_512x512_dense_3d_processed/"
KUBRIC3D_MIX_DIR = "datasets/kubric_DELTA/movif/kubric_processed_mix_3d_instance/"

logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('nose').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('PIL').setLevel(logging.CRITICAL)
logging.getLogger('trimesh').setLevel(logging.CRITICAL)

def vis_instance_mask(batch):
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy.ndimage import label
    import mediapy as media
    cmap = plt.get_cmap("tab20")
    colors = [tuple(int(255 * c) for c in cmap(i)[:3]) for i in range(20)]

    dense_queries_inst_id = batch.dense_queries_inst_id[0].cpu().numpy()

    
    # no need, is already instance seg
    if False:
        instance_mask = np.zeros_like(dense_queries_inst_id, dtype=np.int32)
        current_id = 1
        for sem_id in np.unique(dense_queries_inst_id):
            if sem_id == 0:  # skip background if 0
                continue
            labeled, num = label(dense_queries_inst_id == sem_id)
            labeled[labeled > 0] += current_id - 1
            instance_mask += labeled
            current_id += num
    
    H, W = dense_queries_inst_id.shape
    # sem mask
    max_sem_id = dense_queries_inst_id.max()
    vis_sem_id = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(max_sem_id + 1):
        vis_sem_id[dense_queries_inst_id == i] = np.array(colors[i])
    
    # instance mask
    videos = batch.video[0].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    media.write_video(
                "vis_video.mp4",
                videos,
                fps=10,
            )
    
    rgb = batch.video[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    vis = np.concatenate([rgb, vis_sem_id], axis=1)
    Image.fromarray(vis).save("vis_inst_id.png")


    

def sample_sparse_queries(trajs_g, trajs_d, vis_g):
    B, T, N, D = trajs_g.shape
    device = trajs_g.device

    # NOTE sample sparse queries
    __, first_positive_inds = torch.max(vis_g, dim=1)
    # We want to make sure that during training the model sees visible points
    # that it does not need to track just yet: they are visible but queried from a later frame
    N_rand = N // 4
    # inds of visible points in the 1st frame
    nonzero_inds = [[torch.nonzero(vis_g[b, :, i]) for i in range(N)] for b in range(B)]

    for b in range(B):
        rand_vis_inds = torch.cat(
            [nonzero_row[torch.randint(len(nonzero_row), size=(1,))] for nonzero_row in nonzero_inds[b]],
            dim=1,
        )
        first_positive_inds[b] = torch.cat([rand_vis_inds[:, :N_rand], first_positive_inds[b : b + 1, N_rand:]], dim=1)

    ind_array_ = torch.arange(T, device=device)
    ind_array_ = ind_array_[None, :, None].repeat(B, 1, N)
    assert torch.allclose(
        vis_g[ind_array_ == first_positive_inds[:, None, :]],
        torch.ones(1, device=device),
    )
    gather = torch.gather(trajs_g, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, D))
    gather_d = torch.gather(trajs_d, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, 1))
    xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)
    xys_d = torch.diagonal(gather_d, dim1=1, dim2=2).permute(0, 2, 1)

    sparse_queries = torch.cat([first_positive_inds[:, :, None], xys[:, :, :2], xys_d], dim=2)

    return sparse_queries


def calc_sparse_loss(args, sparse_train_data_dict, batch, trajs_g, trajs_d, vis_g, valids, n_sparse_queries, W, H, max_depth, use_sparse=True):
    coord_predictions = sparse_train_data_dict["coords"]
    coord_depth_predictions = sparse_train_data_dict["coord_depths"]
    vis_predictions = sparse_train_data_dict["vis"]
    conf_predictions = sparse_train_data_dict["conf"]
    valid_mask = sparse_train_data_dict["mask"]

    S = args.sliding_window_len

    seq_loss = torch.tensor(0.0, requires_grad=True).cuda()
    seq_depth_loss = torch.tensor(0.0, requires_grad=True).cuda()
    vis_loss = torch.tensor(0.0, requires_grad=True).cuda()
    conf_loss = torch.tensor(0.0, requires_grad=True).cuda()
    rigid_loss = torch.tensor(0.0, requires_grad=True).cuda()
    if not use_sparse:
        return seq_loss, seq_depth_loss, vis_loss, conf_loss, rigid_loss
    
    for idx, ind in enumerate(range(0, args.sequence_len - S // 2, S // 2)):

        traj_gt_ = trajs_g[:, ind : ind + S].clone()
        traj_d_gt_ = trajs_d[:, ind : ind + S].clone()
        vis_gt_ = vis_g[:, ind : ind + S].clone()
        valid_gt_ = valids[:, ind : ind + S].clone() * valid_mask[:, ind : ind + S, :n_sparse_queries].clone()

        coord_predictions_ = coord_predictions[idx][:, :, :, :n_sparse_queries, :].clone() # B I T N C
        coord_depth_predictions_ = coord_depth_predictions[idx][:, :, :, :n_sparse_queries, :]



        if len(vis_predictions[idx].shape) == 3: # B T N
            vis_predictions_ = vis_predictions[idx][:, :, :n_sparse_queries]
            conf_predictions_ = conf_predictions[idx][:, :, :n_sparse_queries]
            vis_predictions_ = vis_predictions_.unsqueeze(1)
            conf_predictions_ = conf_predictions_.unsqueeze(1) # B I T N C
            # breakpoint()
        else:# B I T N
            vis_predictions_ = vis_predictions[idx][:, :, :, :n_sparse_queries]
            conf_predictions_ = conf_predictions[idx][:, :, :, :n_sparse_queries]

        coord_predictions_[..., 0] /= W - 1
        coord_predictions_[..., 1] /= H - 1

        traj_gt_[..., 0] /= W - 1
        traj_gt_[..., 1] /= H - 1

        coord_depth_predictions_[coord_depth_predictions_ < 0.01] = 0.01
        traj_d_gt_[traj_d_gt_ < 0.01] = 0.01
        coord_depth_predictions_ /= max_depth
        traj_d_gt_ /= max_depth

        seq_loss += track_loss(coord_predictions_, traj_gt_, valid_gt_, divide_n_repeat=False, use_huber_loss=False, delta_huber_loss=6.0/W)
        seq_depth_loss += track_loss(1 / coord_depth_predictions_, 1 / traj_d_gt_, valid_gt_, divide_n_repeat=False)
        vis_loss += seq_balanced_bce_loss(vis_predictions_, vis_gt_, valid_gt_, divide_n_repeat=False)
        conf_loss += seq_confidence_loss(
            coord_predictions_, conf_predictions_, traj_gt_, vis_gt_, valid_gt_, expected_dist_thresh=12.0 / (W - 1), divide_n_repeat=False
        )
        # 
        if args.lambda_rigid_loss is not None:
            traj_inst_ids = batch.sparse_queries_inst_id.clone()
            intrs = batch.intrs[:, ind:ind+S].clone()
            coord_predictions_uv_space = coord_predictions[idx][:, :, :, :n_sparse_queries, :]
            # import pdb
            # pdb.set_trace()
            pred_uvd = torch.cat([coord_predictions_uv_space, traj_d_gt_.unsqueeze(1).repeat(1, coord_predictions_uv_space.shape[1], 1, 1, 1)], axis=-1)
            gt_uvd = torch.cat([trajs_g[:, ind : ind + S], traj_d_gt_], axis=-1)
            rigid_loss += rigid_instane_loss(traj_inst_ids, gt_uvd, pred_uvd, intrs)

    seq_loss = seq_loss * args.lambda_2d / len(coord_predictions)
    seq_depth_loss = seq_depth_loss * args.lambda_d / len(coord_predictions)
    vis_loss = vis_loss * args.lambda_vis / len(coord_predictions)
    conf_loss = conf_loss * args.lambda_conf / len(coord_predictions)
    if args.lambda_rigid_loss is not None:
        rigid_loss = rigid_loss * args.lambda_rigid_loss / len(coord_predictions)

    return seq_loss, seq_depth_loss, vis_loss, conf_loss, rigid_loss

def calc_cycle_loss(args, sparse_train_data_dict, sparse_train_data_dict_inverse,
                    vis_g, n_sparse_queries, W, H, max_depth):
    coord_predictions = sparse_train_data_dict["coords"]
    coord_depth_predictions = sparse_train_data_dict["coord_depths"]
    valid_mask = sparse_train_data_dict["mask"]
    # avoid calc vis and conf loss. Inaccurate uv may lead to incorrect vis and conf, now vis and conf loss becomes meaningless.
    # vis_predictions = sparse_train_data_dict["vis"]
    # conf_predictions = sparse_train_data_dict["conf"]

    coord_predictions_inverse = sparse_train_data_dict_inverse["coords"]
    coord_depth_predictions_inverse = sparse_train_data_dict_inverse["coord_depths"]
    valid_mask_inverse = sparse_train_data_dict_inverse["mask"]

    valid_mask_from_vis = vis_g[:, -1, :] == 1 # B, N
    valid_mask_from_vis = valid_mask_from_vis.unsqueeze(1).repeat(1, args.sequence_len, 1)
    valid_mask_inverse[:, :, :n_sparse_queries] *= valid_mask_from_vis
    
    S = args.sliding_window_len

    seq_loss = torch.tensor(0.0, requires_grad=True).cuda()
    seq_depth_loss = torch.tensor(0.0, requires_grad=True).cuda()

    for idx, ind in enumerate(range(0, args.sequence_len - S // 2, S // 2)):

        
        valid_gt_ = valid_mask[:, ind : ind + S, :n_sparse_queries].clone() * valid_mask_inverse[:, ind : ind + S, :n_sparse_queries].clone() # this means track before query will not calc the loss
        # 1. 
        coord_predictions_ = coord_predictions[idx][:, :, :, :n_sparse_queries, :].clone() # B I T N C
        coord_depth_predictions_ = coord_depth_predictions[idx][:, :, :, :n_sparse_queries, :]

        coord_predictions_[..., 0] /= W - 1
        coord_predictions_[..., 1] /= H - 1

        coord_depth_predictions_[coord_depth_predictions_ < 0.01] = 0.01
        coord_depth_predictions_ /= max_depth

        # 2. inverse
        coord_predictions_inverse_ = coord_predictions_inverse[idx][:, :, :, :n_sparse_queries, :].clone() # B I T N C
        coord_depth_predictions_inverse_ = coord_depth_predictions_inverse[idx][:, :, :, :n_sparse_queries, :]

        coord_predictions_inverse_[..., 0] /= W - 1
        coord_predictions_inverse_[..., 1] /= H - 1

        coord_depth_predictions_inverse_[coord_depth_predictions_inverse_ < 0.01] = 0.01
        coord_depth_predictions_inverse_ /= max_depth

        # 3. calc loss
        seq_loss += track_loss(coord_predictions_, coord_predictions_inverse_, valid_gt_, divide_n_repeat=False, use_huber_loss=False, delta_huber_loss=6.0/W)
        # seq_depth_loss += track_loss(1 / coord_depth_predictions_, 1 / coord_depth_predictions_inverse_, valid_gt_, divide_n_repeat=False)
        
    seq_loss = seq_loss * args.lambda_2d * args.lambda_cycle / len(coord_predictions)
    seq_depth_loss = seq_depth_loss * args.lambda_d * args.lambda_cycle / len(coord_predictions)

    return seq_loss, seq_depth_loss

def forward_batch(batch, model, args, accelerator=None):
    model_stride = args.model_stride

    video = batch.video
    videodepth = batch.videodepth
    depth_init = batch.depth_init
    depth_init_last = batch.depth_init_last

    max_depth = videodepth[videodepth > 0.01].max()

    trajs_g = batch.trajectory
    trajs_d = batch.trajectory_d
    vis_g = batch.visibility
    valids = batch.valid

    dense_trajectory_d = batch.dense_trajectory_d

    flow = batch.flow
    flow_alpha = batch.flow_alpha

    # breakpoint()
    B, T, C, H, W = video.shape
    assert C == 3
    # B, T, N, D = trajs_g.shape
    device = video.device
    # if batch.dataset_name[0] == "pstudio":
    #     import pdb
    #     pdb.set_trace()
    # print(batch.dataset_name[0], trajs_g.shape, trajs_d.shape, vis_g.shape)
    sparse_queries = sample_sparse_queries(trajs_g, trajs_d, vis_g)
    n_sparse_queries = sparse_queries.shape[1]
    # print(batch.dataset_name[0], n_sparse_queries)
    #############################
    # n_input_queries = 256
    # NOTE add regular grid queries:
    def get_grid_queries(depth_init):
        grid_xy = get_points_on_a_grid((12, 16), video.shape[3:]).long().float()
        grid_xy = torch.cat([torch.zeros_like(grid_xy[:, :, :1]), grid_xy], dim=2).to(device)  # B, N, C
        grid_xy_d = bilinear_sampler(depth_init, rearrange(grid_xy[..., 1:3], "b n c -> b () n c"), mode="nearest")
        grid_xy_d = rearrange(grid_xy_d, "b c m n -> b (m n) c")
        grid_queries = torch.cat([grid_xy, grid_xy_d], dim=-1)
        return grid_queries

    grid_queries = get_grid_queries(depth_init)
    grid_queries_last = get_grid_queries(depth_init_last)
    
    # input_queries = torch.cat([sparse_queries, grid_queries], dim=1)
    input_queries = sparse_queries
    # with torch.amp.autocast(device_type=device.type, enabled=True):
    use_dense = args.use_dense
    use_cycle_loss = args.cycle_loss
    use_sparse = True

    if batch.dataset_name[0] == "dynamic_replica":
        use_dense = False
    
    if batch.dataset_name[0] == "pstudio":
        use_dense = False
        use_sparse = False
        use_cycle_loss = True

    sparse_predictions, dense_predictions, (sparse_train_data_dict, dense_train_data_dict), \
        sparse_predictions_inverse, dense_predictions_inverse, (sparse_train_data_dict_inverse, dense_train_data_dict_inverse) = model(
        video=video,
        videodepth=videodepth,
        sparse_queries=input_queries,
        depth_init=depth_init,
        iters=args.train_iters,
        is_train=True,
        use_dense=use_dense,
        accelerator=accelerator,
        grid_queries=grid_queries,
        grid_queries_last=grid_queries_last,
        depth_init_last=depth_init_last,
        cycle_loss=use_cycle_loss
    )

    # sparse loss
    seq_loss, seq_depth_loss, vis_loss, conf_loss, rigid_loss = calc_sparse_loss(args, sparse_train_data_dict, batch, trajs_g, trajs_d, vis_g, valids, n_sparse_queries, W, H, max_depth, use_sparse=use_sparse)
    # cycle loss
    seq_loss_cycle = torch.tensor(0.0, requires_grad=True).cuda()
    seq_depth_loss_cycle = torch.tensor(0.0, requires_grad=True).cuda()
    if use_cycle_loss:
        vis_p = sparse_predictions["vis"][..., :n_sparse_queries].clone().detach() > args.cycle_loss_vis_th
        # import pdb
        # pdb.set_trace()
        seq_loss_cycle, seq_depth_loss_cycle = calc_cycle_loss(
            args, sparse_train_data_dict, sparse_train_data_dict_inverse,
            vis_p, n_sparse_queries, W, H, max_depth)
    
    # dense loss
    dense_seq_loss = torch.tensor(0.0, requires_grad=True).cuda()
    dense_seq_depth_loss = torch.tensor(0.0, requires_grad=True).cuda()
    dense_vis_loss = torch.tensor(0.0, requires_grad=True).cuda()
    dense_conf_loss = torch.tensor(0.0, requires_grad=True).cuda()
    dense_rigid_loss = torch.tensor(0.0, requires_grad=True).cuda()
    if use_dense:
        S = args.sliding_window_len

        dense_coord_predictions = dense_train_data_dict["coords"]
        dense_coord_depth_predictions = dense_train_data_dict["coord_depths"]
        dense_vis_predictions = dense_train_data_dict["vis"]
        dense_conf_predictions = dense_train_data_dict["conf"]
        (x0, y0) = dense_train_data_dict["x0y0"]
        for idx, ind in enumerate(range(0, args.sequence_len - S // 2, S // 2)):

            if idx >= len(dense_coord_predictions):
                break

            dense_coord_prediction_ = dense_coord_predictions[idx][0].clone()  # I T, 3, H, W
            dense_coord_depth_prediction_ = dense_coord_depth_predictions[idx][0]  # I T, 3, H, W
            dense_vis_prediction_ = dense_vis_predictions[idx][0]  # I, T,  H, W
            dense_conf_prediction_ = dense_conf_predictions[idx][0]  # I, T,  H, W

            if len(dense_vis_prediction_.shape) == 3: # T H W
                dense_vis_prediction_ = dense_vis_prediction_.unsqueeze(0)
                dense_conf_prediction_ = dense_conf_prediction_.unsqueeze(0)

            pred_H, pred_W = dense_coord_prediction_.shape[-2:]

            gt_dense_traj_d = dense_trajectory_d[
                0, ind : ind + S, :, y0[0] * model_stride : (y0[0] * model_stride + pred_H), x0[0] * model_stride : (x0[0] * model_stride + pred_W)
            ].clone()  # T 1 H_crop W_crop
            gt_alpha = flow_alpha[
                0, ind : ind + S, y0[0] * model_stride : (y0[0] * model_stride + pred_H), x0[0] * model_stride : (x0[0] * model_stride + pred_W)
            ].clone()  # T 2 H W
            gt_flow = flow[
                0, ind : ind + S, :, y0[0] * model_stride : (y0[0] * model_stride + pred_H), x0[0] * model_stride : (x0[0] * model_stride + pred_W)
            ].clone()  # T 2 H_crop W_crop

            I, S = dense_coord_prediction_.shape[:2]

            dense_grid_2d = get_grid(pred_H, pred_W, normalize=False, device=dense_coord_prediction_.device)  # H W 2
            dense_grid_2d[..., 0] += x0[0] * model_stride
            dense_grid_2d[..., 1] += y0[0] * model_stride
            dense_grid_2d = rearrange(dense_grid_2d, "h w c -> () c h w")

            gt_coord_ = gt_flow + dense_grid_2d

            gt_coord_[:, 0] /= W - 1
            gt_coord_[:, 1] /= H - 1


            dense_coord_prediction_[:, :, 0] /= W - 1
            dense_coord_prediction_[:, :, 1] /= H - 1

            gt_dense_traj_d[gt_dense_traj_d < 0.01] = 0.01
            dense_coord_depth_prediction_[dense_coord_depth_prediction_ < 0.01] = 0.01
            gt_dense_traj_d /= max_depth
            dense_coord_depth_prediction_ /= max_depth

            
            dense_seq_loss_ = track_loss(
                dense_coord_prediction_, 
                gt_coord_, 
                is_dense=True, 
                has_batch_dim=False, 
                divide_n_repeat=False, 
                use_huber_loss=False,
                delta_huber_loss=6.0/W
            )
            dense_seq_loss += dense_seq_loss_
            dense_seq_depth_loss += track_loss(
                1 / dense_coord_depth_prediction_, 1 / gt_dense_traj_d, is_dense=True, has_batch_dim=False, divide_n_repeat=False,
            )
            dense_vis_loss_ = seq_bce_loss(dense_vis_prediction_, gt_alpha, divide_n_repeat=False, has_batch_dim=False)  # bce_loss(dense_vis_prediction_, gt_alpha)
            dense_vis_loss += dense_vis_loss_

            dense_conf_loss += seq_confidence_loss(
                dense_coord_prediction_,
                dense_conf_prediction_,
                gt_coord_,
                gt_alpha,
                expected_dist_thresh=12.0 / W,
                is_dense=True,
                has_batch_dim=False,
                divide_n_repeat=False
            )
            # 
            if args.lambda_rigid_loss is not None:
                dense_queries_inst_id = batch.dense_queries_inst_id.clone()
                traj_inst_ids = dense_queries_inst_id[
                    0, y0[0] * model_stride : (y0[0] * model_stride + pred_H), x0[0] * model_stride : (x0[0] * model_stride + pred_W)
                ].clone()  # H_crop W_crop
                traj_inst_ids = traj_inst_ids.reshape(1, -1)
                intrs = batch.intrs[:, ind:ind+S].clone()
                dense_coord_prediction_uv_space = dense_coord_predictions[idx][0]
                pred_uvd = torch.cat(
                    [
                        dense_coord_prediction_uv_space, 
                        gt_dense_traj_d.unsqueeze(0).repeat(dense_coord_prediction_uv_space.shape[0], 1, 1, 1, 1)
                    ], 
                axis=2)
                pred_uvd = rearrange(pred_uvd, "i t c h w -> i t (h w) c").unsqueeze(0) # B I T N C
                gt_uvd = torch.cat([gt_flow + dense_grid_2d, gt_dense_traj_d], axis=1)
                gt_uvd = rearrange(gt_uvd, "t c h w -> t (h w) c").unsqueeze(0)
                dense_rigid_loss += rigid_instane_loss(traj_inst_ids, gt_uvd, pred_uvd, intrs)

        dense_seq_loss = dense_seq_loss * args.lambda_2d / len(dense_coord_predictions)  # FIXME
        dense_seq_depth_loss = dense_seq_depth_loss * args.lambda_d / len(dense_coord_predictions)
        dense_vis_loss = dense_vis_loss * args.lambda_vis / len(dense_coord_predictions)
        dense_conf_loss = dense_conf_loss * args.lambda_conf / len(dense_coord_predictions)
        if args.lambda_rigid_loss is not None:
            dense_rigid_loss = dense_rigid_loss * args.lambda_rigid_loss / len(dense_coord_predictions)

    # print(batch.dataset_name, use_dense)
    output = {
        "track_uv": {
            "loss": seq_loss.mean(),
            "predictions": sparse_predictions["coords"][0].detach(),
        },
        "track_d": {
            "loss": seq_depth_loss.mean(),
            "predictions": sparse_predictions["coord_depths"][0].detach(),
        },
        "vis": {
            "loss": vis_loss.mean(),
            "predictions": sparse_predictions["vis"][0].detach(),
        },
        "conf": {
            "loss": conf_loss.mean(),
            "predictions": sparse_predictions["conf"][0].detach(),
        },
        "dense_track_uv": {
            "loss": dense_seq_loss.mean(),
        },
        "dense_track_d": {
            "loss": dense_seq_depth_loss.mean(),
        },
        "dense_vis": {
            "loss": dense_vis_loss.mean(),
        },
        "dense_conf": {
            "loss": dense_conf_loss.mean(),
        },
        "rigid_loss":{
            "loss": rigid_loss
        },
        "dense_rigid_loss":{
            "loss": dense_rigid_loss
        },
        "seq_loss_cycle":{
            "loss": seq_loss_cycle
        },
        "seq_depth_loss_cycle":{
            "loss": seq_depth_loss_cycle
        },
    }

    return output


def run_test_eval(accelerator, evaluator, model, dataloaders, step):
    model.eval()
    predictor = EvaluationPredictor(
        model,
        grid_size=5,
        local_grid_size=0,
        single_point=False,
        n_iters=6,
    )
    predictor = predictor.eval().cuda()

    for ds_name, dataloader in dataloaders:

        if "tapvid" in ds_name:

            metrics = evaluator.evaluate_sequence(
                predictor,
                dataloader,
                dataset_name="tapvid_davis_first",
                is_sparse=True,
                verbose=False,
                visualize_every=5,
            )

            metrics = {
                f"{ds_name}_avg_OA": metrics["avg"]["occlusion_accuracy"],
                f"{ds_name}_avg_delta": metrics["avg"]["average_pts_within_thresh"],
                f"{ds_name}_avg_Jaccard": metrics["avg"]["average_jaccard"],
            }
        elif "CVO" in ds_name:
            metrics = evaluator.evaluate_flow(model=predictor, test_dataloader=dataloader, split="clean")

            metrics = {
                f"{ds_name}_avg_epe_all": metrics["avg"]["epe_all"],
                f"{ds_name}_avg_epe_occ": metrics["avg"]["epe_occ"],
                f"{ds_name}_avg_epe_vis": metrics["avg"]["epe_vis"],
                f"{ds_name}_avg_epe_iou": metrics["avg"]["iou"],
            }
    
    return metrics


def seed_everything(seed: int):

    accelerate.utils.set_seed(seed)
    # random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_model(save_path, accelerator, model, optimizer=None, scheduler=None, total_steps=0):
    save_dict = {
        "model": accelerator.unwrap_model(model).state_dict(),
    }

    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    if total_steps is not None:
        save_dict["total_steps"] = total_steps

    logging.info(f"Saving file {save_path}")
    torch.save(save_dict, save_path)

def get_multi_dataset(args):
    kubric_dataset = KubricDataset(
        data_root=KUBRIC3D_MIX_DIR,
        
        seq_len=args.sequence_len,
        traj_per_sample=args.traj_per_sample,
        sample_vis_1st_frame=True,
        use_augs=not args.dont_use_augs,
        use_gt_depth=True,
        # read_from_s3=True,
        read_from_s3=False
    )
    if (not args.use_dr) and (not args.use_pstudio):
        return kubric_dataset

    dataset_lis = [kubric_dataset]
    if args.use_dr:
        dr_dataset = DynamicReplicaDataset(
            root="/mnt/shared-storage-user/dongjunting-group/DATA/dynamicreplica",
            crop_size=(384, 512),
            split="train",
            traj_per_sample=args.traj_per_sample,
            sample_len=args.sequence_len,
            only_first_n_samples=-1,
            rgbd_input=False,
        )
        dataset_lis.append(dr_dataset)
    
    if args.use_pstudio:
        pstudio_dataset = TapVid3DDataset(
            data_root="datasets/",
            datatype="pstudio",
            use_metric_depth=True,  # FIXME check here
            split="mini",
            read_from_s3=False,
            seq_len=args.sequence_len,
            traj_per_sample=args.traj_per_sample,
            # depth_type="zoedepth"
        )
        dataset_lis.append(pstudio_dataset)
    if isinstance(args.dataset_repeats, str):
        repeats = eval(args.dataset_repeats)
    else:
        repeats = args.dataset_repeats
    train_dataset = MixDataset(
        dataset_lis,
        repeats=repeats,
    )
    return train_dataset

def train(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_iter,
        mixed_precision="bf16" if args.bf16 else "no",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
        ],
    )
    device = accelerator.device
    
    seed = 0 + accelerator.state.process_index
    seed_everything(seed)

    # visualizer = Visualizer(
    #     save_dir=args.exp_dir,
    #     fps=7,
    # )

    model = eval(args.model)
    
    print(f"All model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.to(device)

    if False and not args.use_dr:
        train_dataset = KubricDataset(
            data_root=KUBRIC3D_MIX_DIR,
            crop_size=(384, 512),
            seq_len=args.sequence_len,
            traj_per_sample=args.traj_per_sample,
            sample_vis_1st_frame=True,
            use_augs=not args.dont_use_augs,
            use_gt_depth=True,
            # read_from_s3=True,
            read_from_s3=False
        )
    else:
        train_dataset = get_multi_dataset(args)
        # train_dataset = DynamicReplicaDataset(
        #     root="/mnt/shared-storage-user/dongjunting-group/DATA/dynamicreplica",
        #     crop_size=(384, 512),
        #     split="train",
        #     traj_per_sample=args.traj_per_sample,
        #     sample_len=args.sequence_len,
        #     only_first_n_samples=-1,
        #     rgbd_input=False,
        # )


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=True,
        collate_fn=collate_fn_train,
        drop_last=True,
    )
    
    
    # binyanrui - freeze some layer
    # results in slow training
    print("binyanrui")
    for name, p in model.named_parameters():
        if args.freeze_fnet and name.startswith("fnet"):
            p.requires_grad = False
        
        if args.freeze_corr4DCNN and name.startswith("cmdtop"):
            p.requires_grad = False

        if args.freeze_temperal_attn and name.startswith("updateformer.time_blocks"):
            p.requires_grad = False
        
        if args.freeze_dino and name.startswith("dino_net"):
            p.requires_grad = False
        # if name.startswith("upsample_transformer"):
        #     p.requires_grad = False
        # if name.startswith("inter_up"):
        #     p.requires_grad = False
        

        # if name.startswith("norm"):
        #     p.requires_grad = False
        # if name.startswith("track_feat_updater"):
        #     p.requires_grad = False
        # if name.startswith("updateformer.virual_tracks"):
        #     p.requires_grad = False
        # if name.startswith("updateformer.input_transform"):
        #     p.requires_grad = False
        # if name.startswith("updateformer.flow_head"):
        #     p.requires_grad = False
        # if name.startswith("vis_predictor"):
        #     p.requires_grad = False
        # if name.startswith("conf_predictor"):
        #     p.requires_grad = False
        # p.requires_grad = False
        # if name.startswith("updateformer.space"):
        #     p.requires_grad = True
        
        # if p.requires_grad:
        #     print(name)
    # binyanrui

    # optimizer, scheduler = fetch_optimizer(args, model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        (args.num_steps + 100) * accelerator.num_processes,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    total_steps = 0

    folder_ckpts = [
        f for f in os.listdir(args.ckpt_path) if not os.path.isdir(f) and f.endswith(".pth") and not "final" in f and not "best" in f
    ]
    if len(folder_ckpts) > 0:
        ckpt_path = sorted(folder_ckpts)[-1]
        ckpt = torch.load(os.path.join(args.ckpt_path, ckpt_path))
        logging.info(f"Loading checkpoint {ckpt_path}")
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        if "optimizer" in ckpt:
            logging.info("Load optimizer")
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            logging.info("Load scheduler")
            scheduler.load_state_dict(ckpt["scheduler"])
        if "total_steps" in ckpt:
            total_steps = ckpt["total_steps"]
            logging.info(f"Load total_steps {total_steps}")

    elif args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(".pt")
        logging.info(f"Loading checkpoint {args.restore_ckpt}")
        
        strict = False
        state_dict = torch.load(args.restore_ckpt)
        if "model" in state_dict:
            state_dict = state_dict["model"]

        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model_state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                new_state_dict[k] = v
            elif k == "updateformer.input_transform.weight":
                x = torch.zeros_like(model_state_dict[k])
                x[:, :1032] = v
                new_state_dict[k] = x
            else:
                if k in model_state_dict:
                    print(f"{k} can not load. ckpt shape {v.shape} vs model shape {model_state_dict[k].shape}")
                else:
                    print(f"unexpected key {k}")
            # new_state_dict[k] = v

        print(model.load_state_dict(new_state_dict, strict=strict))
        logging.info(f"Done loading checkpoint")

    elif args.resume_from is not None:
        ckpt = torch.load(args.resume_from)
        logging.info(f"Resume checkpoint {args.resume_from}")
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        if "optimizer" in ckpt:
            logging.info("Load optimizer")
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            logging.info("Load scheduler")
            scheduler.load_state_dict(ckpt["scheduler"])
        if "total_steps" in ckpt:
            total_steps = ckpt["total_steps"]
            logging.info(f"Load total_steps {total_steps}")

    
    accelerator.even_batches = False
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    model.train(True)

    
    
    save_video_dir = os.path.join(args.ckpt_path, "videos")
    if accelerator.is_main_process:
        os.makedirs(save_video_dir, exist_ok=True)
        logger = Logger(save_path=os.path.join(args.ckpt_path, "runs"))

    if False:
        eval_dataloaders = []

        eval_davis_dataset = TapVid2DDataset(
            data_root=TAPVID2D_DIR,
            dataset_type="davis",
            resize_to_256=True,
            queried_first=True,
            read_from_s3=False,
            num_processes=accelerator.num_processes
        )

        eval_davis_dataloader = torch.utils.data.DataLoader(
            eval_davis_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
            drop_last=True,
        )
        eval_davis_dataloader = accelerator.prepare(eval_davis_dataloader)
        eval_dataloaders.append(("tapvid_davis_first", eval_davis_dataloader))

        evaluator = Evaluator(save_video_dir)

    save_freq = args.save_freq
    # scaler = GradScaler(device="cuda", enabled=False)

    should_keep_training = True
    global_batch_num = 0
    epoch = -1

    iter_start = 0
    iter_end = 0

    best_avg_jaccard = -1


    while should_keep_training:
        epoch += 1

        for i_batch, batch in enumerate(train_loader):
            # vis_instance_mask(batch[0])
            with accelerator.accumulate(model):
                iter_start = time.time()

                batch, gotit = batch
                # print(i_batch, batch.seq_name)
                if not all(gotit):
                    print("batch is None")
                    continue
                dataclass_to_cuda_(batch)

                optimizer.zero_grad()

                assert model.training

                output = forward_batch(batch, model, args, accelerator=accelerator if args.distribute_window else None)

                loss = 0
                for k, v in output.items():
                    if "loss" in v:
                        loss += v["loss"]

                if accelerator.is_main_process:
                    for k, v in output.items():
                        if "loss" in v:
                            if "dense" in k:
                                logger.writer.add_scalar(f"dense_loss/{k}", v["loss"].item(), total_steps)
                            else:
                                logger.writer.add_scalar(f"sparse_loss/{k}", v["loss"].item(), total_steps)
                        if "metrics" in v:
                            logger.push(v["metrics"], k)
                    if len(output) > 1:
                        logger.writer.add_scalar(f"loss/total_loss", loss.item(), total_steps)
                    
                    current_lr = optimizer.param_groups[0]["lr"]
                    logger.writer.add_scalar(f"LR", current_lr, total_steps)
                    global_batch_num += 1

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                total_steps += 1

                iter_end = time.time()
                elapsed_time = iter_end - iter_start
                remaining_time = ((args.num_steps - total_steps) * elapsed_time) / 3600.0
                if accelerator.is_main_process:
                    if i_batch % 20 == 0:
                        print_mess = f"Epoch {epoch}, iter {total_steps}/{args.num_steps} | Re time: {remaining_time:.2f}h | Elap time: {elapsed_time:.2f}s | Mem: {(torch.cuda.max_memory_allocated() / (1024**2)):.2f} | Lr: {current_lr:.6f}\nLoss: {loss.item():.4f} | "
                        for k, v in output.items():
                            if isinstance(v, dict) and "loss" in v:
                                loss_value = v["loss"].item()
                                if "dense" in k:
                                    tag = f"dense_loss/{k}"
                                else:
                                    tag = f"sparse_loss/{k}"

                                print_mess += f"{tag}: {loss_value:.6f} | "

                        print(print_mess)
                    
                    # prepare output, batch for vis
                    # if output["dense_track_uv"]["loss"].item() > 1. or output["track_uv"]["loss"].item() > 1.:
                    #     import pdb
                    #     pdb.set_trace()
                    #     visualizer.visualize_byr(
                    #         video=sample.video,  # (B,T,C,H,W)
                    #         tracks=dense_traj_e_vis,  # (B,T,N,2)
                    #         visibility=None, # dense_vis_e_vis,  # (B, T, N, 1) bool
                    #         gt_tracks=gt_traj,  # (B,T,N,2)
                    #         filename=sample.seq_name[0],
                    #         writer=None,  # tensorboard Summary Writer, used for visualization during training
                    #         gt_tracks2=selected_dense_traj_e_vis,
                    #     )

                if total_steps > args.num_steps:
                    should_keep_training = False
                    break
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if (epoch + 1) % args.save_every_n_epoch == 0:
                ckpt_iter = "0" * (6 - len(str(total_steps))) + str(total_steps)
                save_path = Path(f"{args.ckpt_path}/model_{args.model_name}_{ckpt_iter}.pth")

                save_model(
                    save_path,
                    accelerator,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    total_steps=total_steps,
                )
            
            if (total_steps + 1) % 1000 == 0:
                save_path = Path(f"{args.ckpt_path}/model_{args.model_name}_latest.pth")

                save_model(
                    save_path,
                    accelerator,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    total_steps=total_steps,
                )


        if False and ((epoch + 1) % args.evaluate_every_n_epoch == 0 or (args.validate_at_start and epoch == 0)):

            model.eval()
            predictor = EvaluationPredictor(
                model,
                grid_size=5,
                local_grid_size=0,
                single_point=False,
                n_iters=6,
            )
            predictor = predictor.eval().cuda()

            for ds_name, dataloader in eval_dataloaders:


                metrics = evaluator.evaluate_sequence(
                    predictor,
                    dataloader,
                    dataset_name="tapvid_davis_first",
                    is_sparse=True,
                    verbose=False,
                    visualize_every=3 if accelerator.is_main_process else -1,
                    exp_dir=os.path.join(save_video_dir, f"iter_{total_steps}") if accelerator.is_main_process else None,
                )

                metrics = {
                    f"{ds_name}_avg_OA": metrics["avg"]["occlusion_accuracy"],
                    f"{ds_name}_avg_delta": metrics["avg"]["average_pts_within_thresh"],
                    f"{ds_name}_avg_Jaccard": metrics["avg"]["average_jaccard"],
                }

                torch.cuda.empty_cache()
                
                log_metrics = {}
                for k, v in metrics.items():
                    metric = torch.tensor(v).to(device)
                    metric = accelerator.gather_for_metrics(metric).mean()
                    if accelerator.is_main_process:
                        log_metrics[k] = metric.item()

                if accelerator.is_main_process:
                    logger.writer.add_scalars(f"Eval", log_metrics, total_steps)
                    print("log_metrics", log_metrics)

                    if best_avg_jaccard < log_metrics[f"{ds_name}_avg_Jaccard"]:
                        best_avg_jaccard = log_metrics[f"{ds_name}_avg_Jaccard"]
                        save_path = Path(f"{args.ckpt_path}/model_{args.model_name}_best.pth")
                        save_model(
                            save_path,
                            accelerator,
                            model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            total_steps=total_steps,
                        )

            model.train()
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ckpt_iter = "0" * (6 - len(str(total_steps))) + str(total_steps)
        save_path = Path(f"{args.ckpt_path}/model_{args.model_name}_final.pth")

        save_model(
            save_path,
            accelerator,
            model,
            total_steps=total_steps,
        )

    accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=os.path.join(str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "configs"),
    config_name="densetrack3dv2.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    
    logdir = os.path.join(cfg.ckpt_path, "runs")
    logdir = pathlib.Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    train(cfg)

if __name__ == '__main__':
    run()