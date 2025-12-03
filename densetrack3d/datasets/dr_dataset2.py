# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from densetrack3d.datasets.dataclass_utils import load_dataclass
from densetrack3d.datasets.utils import DeltaData
from densetrack3d.datasets.kubric_dataset import BasicDataset
from densetrack3d.datasets.utils import DeltaData, add_noise_depth, aug_depth


from pytorch3d.implicitron.dataset.types import (
    FrameAnnotation as ImplicitronFrameAnnotation,
    # load_dataclass
)
from PIL import Image
import cv2

@dataclass
class ImageAnnotation:
    # path to jpg file, relative w.r.t. dataset_root
    path: str
    # H x W
    size: Tuple[int, int]

@dataclass
class DynamicReplicaFrameAnnotation(ImplicitronFrameAnnotation):
    """A dataclass used to load annotations from json."""

    # can be used to join with `SequenceAnnotation`
    sequence_name: str
    # 0-based, continuous frame number within sequence
    frame_number: int
    # timestamp in seconds from the video start
    frame_timestamp: float

    image: ImageAnnotation
    meta: Optional[Dict[str, Any]] = None

    camera_name: Optional[str] = None
    trajectories: Optional[str] = None

    depth = None


class DynamicReplicaDataset(BasicDataset):
    def __init__(
        self,
        root,
        split="valid",
        traj_per_sample=256,
        crop_size=None,
        sample_len=-1,
        only_first_n_samples=-1,
        rgbd_input=False,
    ):
        super(DynamicReplicaDataset, self).__init__(
            data_root=root,
            crop_size=crop_size,
            seq_len=sample_len,
            traj_per_sample=traj_per_sample,
            sample_vis_1st_frame=True,
            use_augs=True,
        )

        self.root = root
        self.sample_len = sample_len
        self.split = split
        self.traj_per_sample = traj_per_sample
        self.rgbd_input = rgbd_input
        self.crop_size = crop_size
        frame_annotations_file = f"frame_annotations_{split}.jgz"
        self.sample_list = []
        with gzip.open(os.path.join(root, split, frame_annotations_file), "rt", encoding="utf8") as zipfile:
            frame_annots_list = load_dataclass(zipfile, List[DynamicReplicaFrameAnnotation])
        seq_annot = defaultdict(list)
        for frame_annot in frame_annots_list:
            if frame_annot.camera_name == "left":
                seq_annot[frame_annot.sequence_name].append(frame_annot)

        for seq_name in seq_annot.keys():
            seq_len = len(seq_annot[seq_name])

            step = self.sample_len if self.sample_len > 0 else seq_len
            counter = 0

            for ref_idx in range(0, seq_len, step):
                sample = seq_annot[seq_name][ref_idx : ref_idx + step]
                if len(sample) < 24:
                    break
                self.sample_list.append(sample)
                counter += 1
                if only_first_n_samples > 0 and counter >= only_first_n_samples:
                    break
        
        self.is_val = False
        self.add_noise_depth = True

        print("found %d unique scenes %d unique videos in %s" % (len(seq_annot.keys()), len(self.sample_list), self.data_root))

    def __len__(self):
        return len(self.sample_list)

    def crop(self, rgbs, trajs):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        H_new = H
        W_new = W

        # simple random crop
        y0 = 0 if self.crop_size[0] >= H_new else (H_new - self.crop_size[0]) // 2
        x0 = 0 if self.crop_size[1] >= W_new else (W_new - self.crop_size[1]) // 2
        rgbs = [rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for rgb in rgbs]

        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0

        return rgbs, trajs

    def _load_16big_png_depth(self, depth_png):
        with Image.open(depth_png) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )
        return depth
    
    def _get_pytorch3d_camera_cut3r(self, entry_viewpoint, image_size, scale: float):
        """
        Convert the camera parameters stored in an annotation to PyTorch3D convention.

        Returns:
            R, tvec, focal, principal_point
        """
        assert entry_viewpoint is not None
        principal_point = torch.tensor(entry_viewpoint.principal_point, dtype=torch.float)
        focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)
        half_image_size_wh_orig = (
            torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
        )

        fmt = entry_viewpoint.intrinsics_format
        if fmt.lower() == "ndc_norm_image_bounds":
            rescale = half_image_size_wh_orig
        elif fmt.lower() == "ndc_isotropic":
            rescale = half_image_size_wh_orig.min()
        else:
            raise ValueError(f"Unknown intrinsics format: {fmt}")

        principal_point_px = half_image_size_wh_orig - principal_point * rescale
        focal_length_px = focal_length * rescale

        # Prepare rotation and translation for PyTorch3D
        R = torch.tensor(entry_viewpoint.R, dtype=torch.float)
        T = torch.tensor(entry_viewpoint.T, dtype=torch.float)
        R_pytorch3d = R.clone()
        T_pytorch3d = T.clone()
        T_pytorch3d[..., :2] *= -1
        R_pytorch3d[..., :, :2] *= -1
        tvec = T_pytorch3d
        R = R_pytorch3d

        # Get camera parameters.
        # R, t, focal, pp = _get_pytorch3d_camera(
        #     viewpoint, framedata.image.size, scale=1.0
        # )
        focal = focal_length_px
        pp = principal_point_px
        R = R
        t = tvec
        intrinsics = np.eye(3)
        intrinsics[0, 0] = focal[0].item()
        intrinsics[1, 1] = focal[1].item()
        intrinsics[0, 2] = pp[0].item()
        intrinsics[1, 2] = pp[1].item()
        pose = np.eye(4)
        # Invert the camera pose.
        pose[:3, :3] = R.numpy().T
        pose[:3, 3] = -R.numpy().T @ t.numpy()
        # pose[:3,:3] = pose[:3,:3].T
        return intrinsics, pose

    def __getitem__(self, index):
        gotit = False
        while not gotit:
            sample, gotit = self.__getitem_(index)

            if gotit:
                return sample, True
            
            index = (index + 1) % self.__len__()

    def __getitem_(self, index):
        sample = self.sample_list[index]
        T = len(sample)
        rgbs, visibilities, traj_2d = [], [], []

        H, W = sample[0].image.size
        image_size = (H, W)

        depths = []
        traj_3d_world = []
        traj_2d_d = []
        c2ws = []
        intrinsics = []
        w2cs = []

        for i in range(T):
            # import pdb
            # pdb.set_trace()
            traj_path = os.path.join(self.root, self.split, sample[i].trajectories["path"])
            traj = torch.load(traj_path)

            visibilities.append(traj["verts_inds_vis"].numpy())

            rgbs.append(traj["img"].numpy())
            traj_2d.append(traj["traj_2d"].numpy()[..., :2])

            traj_2d_d.append(traj["traj_2d"].numpy()[..., 2:])
            traj_3d_world.append(traj["traj_3d_world"].numpy())

            # 2. depth
            depth_path = os.path.join(self.root, self.split, sample[i].depth.path)
            if False:
                depth_map = self._load_16big_png_depth(depth_path)

                # here it is not clear whether masking background is better.
                depth_eps = 1e-5
                depth_mask = depth_map < depth_eps
                # depth_map[depth_mask] = depth_eps
                d_invalid_mask = np.logical_not(depth_mask)
                depth_map[~d_invalid_mask] = 0
            else:
                scale_adj = sample[i].depth.scale_adjustment
                depth_map = _load_depth(depth_path, scale_adj)

                # depth_map = threshold_depth_map(
                #     depth_map, min_percentile=-1, max_percentile=98
                # )
                depths.append(depth_map)

            # 3. camera
            if False:
                intrinsic, c2w_opencv = self._get_pytorch3d_camera_cut3r(
                    sample[i].viewpoint,
                    sample[i].image.size,
                    scale = 1.0
                )
                w2c_opencv = np.linalg.inv(c2w_opencv)
                intrinsics.append(intrinsic)
                c2ws.append(c2w_opencv)
                w2cs.append(w2c_opencv)
            else:
                # 相机
                K_src = _ndc_isotropic_to_pixel_K(sample[i].viewpoint, W, H)
                w2c   = _pytorch3d_RT_to_w2c(sample[i].viewpoint)
                c2w   = np.linalg.inv(w2c).astype(np.float32)
                c2ws.append(c2w)
                w2cs.append(w2c)
                intrinsics.append(K_src)
                

        rgbs = np.stack(rgbs)
        depths = np.stack(depths)[..., None]   # T, H, W, 1
        pred_depths = np.zeros_like(depths)
        c2ws = np.stack(c2ws)
        intrinsics = np.stack(intrinsics)
        w2cs = np.stack(w2cs)

        
        sparse_traj_2d = np.stack(traj_2d) # T,N,2
        if False:
            sparse_traj_2d_int = sparse_traj_2d.astype(np.int32) # T,N,2
            u = sparse_traj_2d_int[..., 0].astype(int)   # (T, N)
            v = sparse_traj_2d_int[..., 1].astype(int)   # (T, N)
            sparse_traj_depth = depth_values = depths[np.arange(depths.shape[0])[:, None], v, u, 0] # # (T, N)
        else:
            sparse_traj_depth = np.stack(traj_2d_d)
            sparse_traj_3d_world = np.stack(traj_3d_world)
            sparse_traj_3d_world_4x4 = np.concatenate([sparse_traj_3d_world, np.ones_like(sparse_traj_3d_world[..., :1])], axis=-1) # T,N,4
            
            sparse_traj_3d_cam_4x4 = sparse_traj_3d_world_4x4 @ w2cs.transpose(0, 2, 1)
            sparse_traj_3d_cam = sparse_traj_3d_cam_4x4[..., :3]
            sparse_traj_dist = np.linalg.norm(sparse_traj_3d_cam, axis=-1)
            # print(visibilities[0][2], sparse_traj_2d[0,2])
            # print(sparse_traj_3d_cam[0, 2])
            # print(depths[0, 300, 485])
            # visibilities_0 = visibilities[0]
            # oob = np.logical_or(
            #     np.logical_or(sparse_traj_2d[0, :, 0] < 0, sparse_traj_2d[0, :, 0] > W-1),
            #     np.logical_or(sparse_traj_2d[0, :, 1] < 0, sparse_traj_2d[0, :, 1] > H-1)
            # )
            # visibilities_0 = np.logical_and(visibilities_0, np.logical_not(oob))
            # vis_2d = sparse_traj_2d[0][visibilities_0].astype(np.int32)
            # print(depths[0, vis_2d[:, 1], vis_2d[:, 0]].mean())
            # print(sparse_traj_3d_cam[0][visibilities_0][..., 2].mean())
            # import pdb
            # pdb.set_trace()
            sparse_traj_depth = sparse_traj_dist[..., None]
        sparse_visibility = np.stack(visibilities).astype(bool)

        # resize & update
        rgbs, depths, sparse_traj_2d, _, intrinsics, scale, (H, W) = resize_batch_short_edge(rgbs, depths, sparse_traj_2d, sparse_traj_3d_world, intrinsics, short_edge=384)
        
        T, N, D = sparse_traj_2d.shape

        # random inverse
        if np.random.uniform() < 0.5:
            rgbs = rgbs[::-1]
            depths = depths[::-1]
            pred_depths = pred_depths[::-1]
            sparse_traj_2d = sparse_traj_2d[::-1]
            sparse_traj_depth = sparse_traj_depth[::-1]
            sparse_visibility = sparse_visibility[::-1]
        
        if np.any(np.isinf(sparse_traj_2d)):
            print("warning: traj contains inf")
            gotit = False
            return None, gotit

        # augment
        dense_traj_grid = np.zeros((sparse_traj_2d.shape[0], H, W, 2))
        dense_visi_maps = np.zeros((sparse_traj_2d.shape[0], H, W))
        dense_traj_depth_grid = np.zeros((sparse_traj_2d.shape[0], H, W, 1))
        dense_flows = np.zeros((sparse_traj_2d.shape[0], H, W, 2))
        dense_flow_depths = np.zeros((sparse_traj_2d.shape[0], H, W, 1))
        dense_queries_inst_id = np.zeros((H, W))
        if not self.is_val:
            if self.use_augs:
                rgbs, dense_traj_grid, dense_visi_maps, sparse_traj_2d, sparse_visibility = \
                    self.add_photometric_augs(rgbs, dense_traj_grid, dense_visi_maps, sparse_traj_2d, sparse_visibility)
                rgbs, depths, pred_depths, dense_traj_grid, dense_traj_depth_grid, dense_visi_maps, dense_flows, dense_flow_depths, sparse_traj_2d, sparse_visibility, dense_queries_inst_id = \
                    self.add_spatial_augs(rgbs, depths, pred_depths, dense_traj_grid, dense_traj_depth_grid, dense_visi_maps, dense_flows, dense_flow_depths, sparse_traj_2d, sparse_visibility, dense_queries_inst_id)
            else:
                rgbs, depths, pred_depths, dense_traj_grid, dense_traj_depth_grid, dense_visi_maps, dense_flows, dense_flow_depths, sparse_traj_2d, sparse_visibility = self.crop_rgb_and_flow(rgbs, depths, pred_depths, dense_traj_grid, dense_traj_depth_grid, dense_visi_maps, dense_flows, dense_flow_depths, sparse_traj_2d, sparse_visibility)
        else:
            rgbs, depths, pred_depths, dense_traj_grid, dense_traj_depth_grid, dense_visi_maps, dense_flows, dense_flow_depths, sparse_traj_2d, sparse_visibility = self.crop_rgb_and_flow(rgbs, depths, pred_depths, dense_traj_grid, dense_traj_depth_grid, dense_visi_maps, dense_flows, dense_flow_depths, sparse_traj_2d, sparse_visibility, random=False)
        

        # revalidate the visibility
        sparse_visibility[sparse_traj_2d[:, :, 0] > image_size[1] - 1] = False
        sparse_visibility[sparse_traj_2d[:, :, 0] < 0] = False
        sparse_visibility[sparse_traj_2d[:, :, 1] > image_size[0] - 1] = False
        sparse_visibility[sparse_traj_2d[:, :, 1] < 0] = False

        # filter out points that're visible for less than 10 frames
        visible_inds_resampled = sparse_visibility.sum(0) > 10
        sparse_traj_2d = torch.from_numpy(sparse_traj_2d[:, visible_inds_resampled])
        sparse_visibility = torch.from_numpy(sparse_visibility[:, visible_inds_resampled])

        # to tensor
        rgbs = torch.from_numpy(np.ascontiguousarray(rgbs)).permute(0, 3, 1, 2).float()
        depths = torch.from_numpy(np.ascontiguousarray(depths)).permute(0, 3, 1, 2).float()
        pred_depths = torch.from_numpy(np.ascontiguousarray(pred_depths)).permute(0, 3, 1, 2).float()
        dense_flows = torch.from_numpy(np.ascontiguousarray(dense_flows)).permute(0, 3, 1, 2).float()
        dense_flow_depths = torch.from_numpy(np.ascontiguousarray(dense_flow_depths)).permute(0, 3, 1, 2).float()
        dense_visi_maps = torch.from_numpy(np.ascontiguousarray(dense_visi_maps)).float()

        dense_traj_grid = torch.from_numpy(np.ascontiguousarray(dense_traj_grid)).permute(0, 3, 1, 2).float()
        dense_traj_depth_grid = torch.from_numpy(np.ascontiguousarray(dense_traj_depth_grid)).permute(0, 3, 1, 2).float()

        sparse_traj_2d = torch.from_numpy(np.ascontiguousarray(sparse_traj_2d)).float()
        sparse_traj_depth = torch.from_numpy(np.ascontiguousarray(sparse_traj_depth)).float()
        sparse_visibility = torch.from_numpy(np.ascontiguousarray(sparse_visibility)).float()

        dense_queries_inst_id = torch.from_numpy(dense_queries_inst_id)
        sparse_queries_inst_id = None

        if not self.is_val:
            visibile_pts_first_frame_inds = (sparse_visibility[0]).nonzero(as_tuple=False)[:, 0]
            if self.sample_vis_1st_frame:
                visibile_pts_inds = visibile_pts_first_frame_inds
            else:
                visibile_pts_mid_frame_inds = (sparse_visibility[self.seq_len // 2]).nonzero(as_tuple=False)[
                    :, 0
                ]
                visibile_pts_inds = torch.cat(
                    (visibile_pts_first_frame_inds, visibile_pts_mid_frame_inds), dim=0
                )

            if len(visibile_pts_inds) < 32:
                return None, False

            if len(visibile_pts_inds) >= self.traj_per_sample:
                point_inds = torch.randperm(len(visibile_pts_inds))[: self.traj_per_sample]
            else:
                point_inds = np.random.choice(len(visibile_pts_inds), self.traj_per_sample, replace=True)
            visible_inds_sampled = visibile_pts_inds[point_inds]

            sparse_traj_2d = sparse_traj_2d[:, visible_inds_sampled].float()
            sparse_traj_depth = sparse_traj_depth[:, visible_inds_sampled].float()
            sparse_visibility = sparse_visibility[:, visible_inds_sampled]
        sparse_valids = torch.ones_like(sparse_visibility)

        # aug depth
        depth_init = depths[0].clone()
        depth_init_last = depths[-1].clone()
        if not self.is_val and self.use_augs:
            depths = aug_depth(depths,
                    grid=(8, 8),
                    scale=(0.85, 1.15),
                    shift=(-0.05, 0.05),
                    gn_kernel=(7, 7),
                    gn_sigma=(2, 2),
                    mask_depth=(depths >= 0.01)
            )

            if self.add_noise_depth:
                depths = add_noise_depth(
                    depths, 
                    gn_sigma=0.3, 
                    mask_depth=(depths >= 0.01)
                )


        dense_valid = torch.ones_like(dense_visi_maps)

        intrinsic_mat = torch.from_numpy(intrinsics).float()

        seq_name = sample[0].sequence_name
        sample = DeltaData(
            video=rgbs,
            videodepth=depths,
            videodepth_pred=pred_depths,
            depth_init=depth_init,
            trajectory=sparse_traj_2d,
            trajectory_d=sparse_traj_depth,
            visibility=sparse_visibility,
            valid=sparse_valids,
            dense_trajectory=dense_traj_grid,
            dense_trajectory_d=dense_traj_depth_grid,
            dense_valid=dense_valid,
            flow=dense_flows,
            flow_depth=dense_flow_depths,
            flow_alpha=dense_visi_maps,
            seq_name=seq_name,
            dataset_name="dynamic_replica",
            depth_min=depths.min(),
            depth_max=depths.max(),
            intrs=intrinsic_mat,
            dense_queries_inst_id=dense_queries_inst_id,
            sparse_queries_inst_id=sparse_queries_inst_id,
            depth_init_last=depth_init_last
        )

        return sample, True


def resize_batch_short_edge(rgb, depth, traj2d, traj3d, intrinsic, short_edge=384):
    """
    rgb: (T, H, W, 3)
    depth: (T, H, W, 1)
    traj2d: (T, N, 2)
    traj3d: (T, N, 3)  # will not change
    intrinsic: (3,3)
    """
    T, H, W = rgb.shape[:3]

    # compute scale
    if H < W:
        scale = short_edge / H
    else:
        scale = short_edge / W

    newH = int(round(H * scale))
    newW = int(round(W * scale))

    # resize rgb & depth for all frames
    rgb_resized = np.zeros((T, newH, newW, 3), dtype=rgb.dtype)
    depth_resized = np.zeros((T, newH, newW, 1), dtype=depth.dtype)

    for t in range(T):
        rgb_resized[t] = cv2.resize(rgb[t], (newW, newH), interpolation=cv2.INTER_LINEAR)
        d = depth[t].squeeze()
        d = cv2.resize(d, (newW, newH), interpolation=cv2.INTER_NEAREST)
        depth_resized[t] = d[..., None]

    # scale trajectory 2D
    traj2d_resized = traj2d * scale

    # 3D trajectory does not change
    traj3d_resized = traj3d.copy()

    # update intrinsics
    intrinsic_resized = intrinsic.copy()
    intrinsic_resized[0, 0] *= scale  # fx
    intrinsic_resized[1, 1] *= scale  # fy
    intrinsic_resized[0, 2] *= scale  # cx
    intrinsic_resized[1, 2] *= scale  # cy

    return (
        rgb_resized,
        depth_resized,
        traj2d_resized,
        traj3d_resized,
        intrinsic_resized,
        scale,
        (newH, newW)
    )

def threshold_depth_map(
    depth_map: np.ndarray,
    max_percentile: float = 99,
    min_percentile: float = 1,
    max_depth: float = -1,
) -> np.ndarray:
    """
    Thresholds a depth map using percentile-based limits and optional maximum depth clamping.

    Steps:
      1. If `max_depth > 0`, clamp all values above `max_depth` to zero.
      2. Compute `max_percentile` and `min_percentile` thresholds using nanpercentile.
      3. Zero out values above/below these thresholds, if thresholds are > 0.

    Args:
        depth_map (np.ndarray):
            Input depth map (H, W).
        max_percentile (float):
            Upper percentile (0-100). Values above this will be set to zero.
        min_percentile (float):
            Lower percentile (0-100). Values below this will be set to zero.
        max_depth (float):
            Absolute maximum depth. If > 0, any depth above this is set to zero.
            If <= 0, no maximum-depth clamp is applied.

    Returns:
        np.ndarray:
            Depth map (H, W) after thresholding. Some or all values may be zero.
            Returns None if depth_map is None.
    """
    if depth_map is None:
        return None

    depth_map = depth_map.astype(float, copy=True)

    # Optional clamp by max_depth
    if max_depth > 0:
        depth_map[depth_map > max_depth] = 0.0

    # Percentile-based thresholds
    depth_max_thres = (
        np.nanpercentile(depth_map, max_percentile) if max_percentile > 0 else None
    )
    depth_min_thres = (
        np.nanpercentile(depth_map, min_percentile) if min_percentile > 0 else None
    )

    # Apply the thresholds if they are > 0
    if depth_max_thres is not None and depth_max_thres > 0:
        depth_map[depth_map > depth_max_thres] = 0.0
    if depth_min_thres is not None and depth_min_thres > 0:
        depth_map[depth_map < depth_min_thres] = 0.0

    return depth_map

def _load_16bit_png_depth_as_f32(path: str) -> np.ndarray:
    """读取打包为 float16 的 16-bit PNG，返回 float32(H,W)。"""
    with Image.open(path) as depth_pil:
        arr_u16 = np.array(depth_pil, dtype=np.uint16)
        depth = np.frombuffer(arr_u16.tobytes(), dtype=np.float16).astype(np.float32)
        depth = depth.reshape(depth_pil.size[1], depth_pil.size[0])
    return depth

def _load_depth(path: str, scale_adjustment: float) -> np.ndarray:
    """支持 .png(half packed) / .exr；做尺度与清理。"""
    if path.lower().endswith(".exr"):
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if d is None:
            raise FileNotFoundError(f"Failed to read EXR depth: {path}")
        d = d[..., 0].astype(np.float32)
        d[d > 1e9] = 0.0
    elif path.lower().endswith(".png"):
        d = _load_16bit_png_depth_as_f32(path)
    else:
        raise ValueError(f"Unsupported depth file: {path}")
    d = d * float(scale_adjustment)
    d[~np.isfinite(d)] = 0.0
    d[d < 0] = 0.0
    return d.astype(np.float32)

# --------- 相机变换（按官方注释保持一致） ---------

def _ndc_isotropic_to_pixel_K(viewpoint: Dict[str, Any], w: int, h: int) -> np.ndarray:
    """从 PyTorch3D 的 ndc_isotropic 转到像素系 K。"""
    s = min(h, w)
    fx_ndc, fy_ndc = viewpoint.focal_length # viewpoint["focal_length"]
    px_ndc, py_ndc = viewpoint.principal_point# viewpoint["principal_point"]
    fx = fx_ndc * s / 2.0
    fy = fy_ndc * s / 2.0
    px = (-px_ndc * s / 2.0) + w / 2.0
    py = (-py_ndc * s / 2.0) + h / 2.0
    # 与上游对齐半像素
    px -= 0.5
    py -= 0.5
    K = np.array([[fx, 0.0, px],
                  [0.0, fy, py],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def _pytorch3d_RT_to_w2c(viewpoint: Dict[str, Any]) -> np.ndarray:
    """从 PyTorch3D 的 R/T 得到 w2c，并做轴翻转；返回 4x4。"""
    # R = np.array(viewpoint["R"], dtype=np.float32).T
    # T = np.array(viewpoint["T"], dtype=np.float32)
    R = np.array(viewpoint.R, dtype=np.float32).T
    T = np.array(viewpoint.T, dtype=np.float32)
    flip = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = flip @ R
    w2c[:3, 3]  = (flip @ T.reshape(3,)).astype(np.float32)
    return w2c
