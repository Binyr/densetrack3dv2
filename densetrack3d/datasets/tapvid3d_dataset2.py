import dataclasses

import io
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from einops import rearrange
from PIL import Image

from densetrack3d.datasets.utils import DeltaData
from densetrack3d.models.geometry_utils import least_square_align
from densetrack3d.models.model_utils import depth_to_disparity, get_grid, sample_features5d

try:
    from densetrack3d.datasets.s3_utils import create_client, get_client_stream, read_s3_json
    has_s3 = True
except:
    has_s3 = False


UINT16_MAX = 65535
TAPVID3D_ROOT = None


def get_jpeg_byte_hw(jpeg_bytes: bytes):
    with io.BytesIO(jpeg_bytes) as img_bytes:
        img = Image.open(img_bytes)
        img = img.convert("RGB")
    return np.array(img).shape[:2]


def get_new_hw_with_given_smallest_side_length(*, orig_height: int, orig_width: int, smallest_side_length: int = 256):
    orig_shape = np.array([orig_height, orig_width])
    scaling_factor = smallest_side_length / np.min(orig_shape)
    resized_shape = np.round(orig_shape * scaling_factor)
    return (int(resized_shape[0]), int(resized_shape[1])), scaling_factor


def project_points_to_video_frame(camera_pov_points3d, camera_intrinsics, height, width):
    """Project 3d points to 2d image plane."""
    u_d = camera_pov_points3d[..., 0] / (camera_pov_points3d[..., 2] + 1e-8)
    v_d = camera_pov_points3d[..., 1] / (camera_pov_points3d[..., 2] + 1e-8)

    f_u, f_v, c_u, c_v = camera_intrinsics

    u_d = u_d * f_u + c_u
    v_d = v_d * f_v + c_v

    # Mask of points that are in front of the camera and within image boundary
    masks = camera_pov_points3d[..., 2] >= 1
    masks = masks & (u_d >= 0) & (u_d < width) & (v_d >= 0) & (v_d < height)
    return np.stack([u_d, v_d], axis=-1), masks


class TapVid3DDataset(Dataset):

    def __init__(
        self,
        data_root,
        datatype="pstudio",
        crop_size=256,
        debug=False,
        use_metric_depth=True,
        split="minival",
        read_from_s3=False,
        depth_type="unidepth",
        seq_len=24,
        traj_per_sample=768,
    ):
        self.seq_len = seq_len
        self.traj_per_sample = traj_per_sample

        if split == "all":
            datatype = datatype + "_all"

        self.read_from_s3 = read_from_s3
        self.datatype = datatype
        self.data_root = os.path.join(data_root, datatype)

        if self.read_from_s3:
            assert has_s3
            self.client = create_client()

            tapvid3d_metadata_path = os.path.join(data_root, "meta_data.json")
            # self.data_root = os.path.join(data_root, datatype)

            meta_data = read_s3_json(self.client, tapvid3d_metadata_path)
            self.video_names = meta_data[datatype]
        else:
            self.video_names = sorted([f.split(".")[0] for f in os.listdir(self.data_root) if f.endswith(".npz")])

        self.debug = debug
        self.crop_size = crop_size
        self.use_metric_depth = use_metric_depth
        self.depth_type = depth_type

        print(f"Found {len(self.video_names)} samples for TapVid3D {datatype}")

    def __len__(self):
        if self.debug:
            return 10
        return len(self.video_names)

    def __getitem__(self, index):
        got_it = False
        while not got_it:
            data, got_it = self.__getitem_(index)
            index = (index + 1) % len(self)
        return data

    def __getitem_(self, index):
        video_name = self.video_names[index]

        gt_path = os.path.join(self.data_root, f"{video_name}.npz")

        # with open(gt_path, 'rb') as in_f:
        #     in_npz = np.load(in_f, allow_pickle=True)

        if self.read_from_s3:
            in_npz = np.load(get_client_stream(self.client, gt_path), allow_pickle=True)
        else:
            in_npz = np.load(gt_path, allow_pickle=True)

        # print("debug", in_npz.files)

        images_jpeg_bytes = in_npz["images_jpeg_bytes"]
        video = []
        for frame_bytes in images_jpeg_bytes:
            arr = np.frombuffer(frame_bytes, np.uint8)
            image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            video.append(image_rgb)
        video = np.stack(video, axis=0)

        if self.depth_type == "unidepth":
            metric_videodepth = in_npz["depth_preds"]  # NOTE UniDepth
        elif self.depth_type == "zoedepth":
            metric_videodepth = in_npz['depth_preds_zoe'] # NOTE ZoeDepth
        elif self.depth_type == "mvmonst3r":
            metric_videodepth = in_npz['depth_preds_mvmonst3r_16']
        else:
            raise ValueError(f"Unknown depth type: {self.depth_type}")

        if self.use_metric_depth:
            videodepth = metric_videodepth
        else:
            videodisp = in_npz["depth_preds_depthcrafter"]
            videodisp = videodisp.astype(np.float32) / UINT16_MAX
            videodepth = least_square_align(metric_videodepth, videodisp, return_align_scalar=False)

        queries_xyt = in_npz["queries_xyt"]
        tracks_xyz = in_npz["tracks_XYZ"]
        
        visibles = in_npz["visibility"]
        intrinsics_params = in_npz["fx_fy_cx_cy"]

        tracks_uv, _ = project_points_to_video_frame(tracks_xyz, intrinsics_params, video.shape[1], video.shape[2])

        # - binyanrui: the original query is noisy, use some point in gt_traj instead
        q_t = queries_xyt[:, 2].astype(np.int32)
        ttt = np.arange(queries_xyt.shape[0])
        queries_xyt = np.concatenate([tracks_uv[q_t, ttt], q_t[:, None]], axis=1)
        
        scaling_factor = 1.0
        intrinsics_params_resized = intrinsics_params * scaling_factor
        intrinsic_mat = np.array(
            [
                [intrinsics_params_resized[0], 0, intrinsics_params_resized[2]],
                [0, intrinsics_params_resized[1], intrinsics_params_resized[3]],
                [0, 0, 1],
            ]
        )

        
        # 1. random reverse
        if np.random.uniform() < 0.5:
            video = np.ascontiguousarray(video[::-1])
            videodepth = np.ascontiguousarray(videodepth[::-1])
            T = video.shape[0]
            queries_xyt[:, 2] = T - queries_xyt[:, 2] # reverse query frame
            tracks_xyz = np.ascontiguousarray(tracks_xyz[::-1])
            visibles = np.ascontiguousarray(visibles[::-1])
            tracks_uv = np.ascontiguousarray(tracks_uv[::-1])
        
        # 2. resize
        video, videodepth, tracks_uv, tracks_xyz, intrinsic_mat, \
            scale, (newH, newW)  = resize_batch_short_edge(video, videodepth, tracks_uv, tracks_xyz, intrinsic_mat, short_edge=384)
        
        # 3. sample frame
        T = video.shape[0]
        start_t = np.random.randint(0, T - self.seq_len)
        end_t = start_t + self.seq_len
        video = video[start_t:end_t]
        videodepth = videodepth[start_t:end_t]
        # queries_xyt = queries_xyt
        tracks_xyz = tracks_xyz[start_t:end_t]
        visibles = visibles[start_t:end_t]
        tracks_uv = tracks_uv[start_t:end_t] # 24, 110, 2

        # 4. to tensor
        intrinsic_mat = torch.from_numpy(intrinsic_mat).float()
        intrinsic_mat = intrinsic_mat[None].repeat(video.shape[0], 1, 1)

        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        videodepth = torch.from_numpy(videodepth).float().permute(0, 3, 1, 2)
        segs = torch.ones_like(videodepth)

        trajectory_3d = torch.from_numpy(tracks_xyz).float()  # T N D
        trajectory_2d = torch.from_numpy(tracks_uv).float()  # T N 2
        visibility = torch.from_numpy(visibles)
        query_points = torch.from_numpy(queries_xyt).float()

        # 5. sample predict depth
        T, N = tracks_uv.shape[:2]
        tracks_t = np.arange(T)[:, None, None].repeat(N, 1)
        track_tuv = np.concatenate([tracks_t, tracks_uv], axis=2)
        track_tuv = torch.from_numpy(track_tuv)[None, ...].float() # 1 T N 3
        sample_coords = track_tuv
        track_tuv_depth = sample_features5d(videodepth[None], sample_coords, mode="nearest")
        track_tuv_depth = track_tuv_depth.squeeze(0)
        track_tuvd = torch.cat(
            [track_tuv.squeeze(0), track_tuv_depth], dim=-1
        )  # NOTE by default, query is N 3: xyt but we use N 3: txy
        track_uvd = track_tuvd[:, :, 1:]
        
        # 6. 
        # revalidate the visibility
        image_size = (newH, newW)
        visibility[trajectory_2d[:, :, 0] > image_size[1] - 1] = False
        visibility[trajectory_2d[:, :, 0] < 0] = False
        visibility[trajectory_2d[:, :, 1] > image_size[0] - 1] = False
        visibility[trajectory_2d[:, :, 1] < 0] = False

        # filter out points that're visible for less than 10 frames
        visible_inds_resampled = visibility.sum(0) > 10
        if visible_inds_resampled.sum() < 32:
            return None, False
        # import pdb
        # pdb.set_trace()
        visibile_pts_inds = visible_inds_resampled.nonzero(as_tuple=False)[:, 0]
        if len(visibile_pts_inds) >= self.traj_per_sample:
            point_inds = torch.randperm(len(visibile_pts_inds))[: self.traj_per_sample]
        else:
            point_inds = np.random.choice(len(visibile_pts_inds), self.traj_per_sample, replace=True)

        visible_inds_sampled = visibile_pts_inds[point_inds]

        trajectory_2d = trajectory_2d[:, visible_inds_sampled]
        visibility = visibility[:, visible_inds_sampled]
        trajectory_3d = trajectory_3d[:, visible_inds_sampled]
        track_tuv_depth = track_tuv_depth[:, visible_inds_sampled]

        trajectory_d = track_tuv_depth

        # trajectory_3d =
        # ...

        if False:
            sample_coords = torch.cat([query_points[:, 2:3], query_points[:, :2]], dim=-1)[None, None, ...]  # 1 1 N 3

            rgb_h, rgb_w = video.shape[2], video.shape[3]
            depth_h, depth_w = videodepth.shape[2], videodepth.shape[3]
            if rgb_h != depth_h or rgb_w != depth_w:
                sample_coords[..., 1] = sample_coords[..., 1] * depth_w / rgb_w
                sample_coords[..., 2] = sample_coords[..., 2] * depth_h / rgb_h

            query_points_depth = sample_features5d(videodepth[None], sample_coords, mode="nearest")
            query_points_depth = query_points_depth.squeeze(0, 1)
            query_points_3d = torch.cat(
                [query_points[:, 2:3], query_points[:, :2], query_points_depth], dim=-1
            )  # NOTE by default, query is N 3: xyt but we use N 3: txy

        
        
        data = DeltaData(
            video=video,
            videodepth=videodepth,
            segmentation=segs,
            trajectory=trajectory_2d,
            trajectory3d=trajectory_3d,
            visibility=visibility,
            seq_name=f"{video_name}_{start_t}_{end_t}",
            query_points=None,
            intrs=intrinsic_mat,
            dataset_name="pstudio",
            trajectory_d=trajectory_d,
            depth_init=videodepth[0].clone(),
            depth_init_last=videodepth[-1].clone()
        )
        return data, True


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

    newH = 384
    newW = 512

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