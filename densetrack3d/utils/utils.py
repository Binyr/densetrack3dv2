import torch
import numpy as np

def depth_to_distance_numpy(depths, intrinsics):
    """
    depths:     (T, 1, H, W)  depth is z in camera coords
    intrinsics: (T, 3, 3)

    return:
        distance: (T, 1, H, W)
    """
    T, _, H, W = depths.shape

    # pixel grid (H, W)
    ys, xs = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij"
    )

    # reshape to broadcast: (1,1,H,W)
    xs = xs[None, None, ...]
    ys = ys[None, None, ...]

    # intrinsics: reshape for broadcast (T,1,1,1)
    fx = intrinsics[:, 0, 0].reshape(T, 1, 1, 1)
    fy = intrinsics[:, 1, 1].reshape(T, 1, 1, 1)
    cx = intrinsics[:, 0, 2].reshape(T, 1, 1, 1)
    cy = intrinsics[:, 1, 2].reshape(T, 1, 1, 1)

    # depth (T,1,H,W)
    z = depths

    # backproject to camera coordinates
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z

    # Euclidean distance to camera center
    distance = np.sqrt(x * x + y * y + z * z + 1e-8)

    return distance

def depth_to_distance_th(depths, intrinsics):
    """
    depths:     (T, 1, H, W)
    intrinsics: (T, 3, 3)
    
    return:
        distance_to_camera: (T, 1, H, W)
    """
    T, _, H, W = depths.shape

    # pixel grid
    ys, xs = torch.meshgrid(
        torch.arange(H, device=depths.device),
        torch.arange(W, device=depths.device),
        indexing="ij"
    )  # ys: (H,W), xs: (H,W)

    xs = xs.float()[None, None]  # (1,1,H,W)
    ys = ys.float()[None, None]  # (1,1,H,W)

    # reshape intrinsics
    fx = intrinsics[:, 0, 0].view(T, 1, 1, 1)  # (T,1,1,1)
    fy = intrinsics[:, 1, 1].view(T, 1, 1, 1)
    cx = intrinsics[:, 0, 2].view(T, 1, 1, 1)
    cy = intrinsics[:, 1, 2].view(T, 1, 1, 1)

    # broadcast depths: (T,1,H,W)
    z = depths

    # backproject (camera coords)
    x = (xs - cx) / fx * z        # (T,1,H,W)
    y = (ys - cy) / fy * z        # (T,1,H,W)

    # Euclidean distance
    dist = torch.sqrt(x * x + y * y + z * z + 1e-8)

    return dist

def distance_to_depth(distance, intrinsics):
    """
    distance:   (T, 1, H, W)   distance-to-camera
    intrinsics: (T, 3, 3)

    return:
        depth: (T, 1, H, W)    Z in camera coordinates
    """
    T, _, H, W = distance.shape

    # pixel grid
    ys, xs = torch.meshgrid(
        torch.arange(H, device=distance.device),
        torch.arange(W, device=distance.device),
        indexing="ij"
    )
    xs = xs.float()[None, None]  # (1,1,H,W)
    ys = ys.float()[None, None]

    # reshape intrinsics for broadcast
    fx = intrinsics[:, 0, 0].view(T, 1, 1, 1)
    fy = intrinsics[:, 1, 1].view(T, 1, 1, 1)
    cx = intrinsics[:, 0, 2].view(T, 1, 1, 1)
    cy = intrinsics[:, 1, 2].view(T, 1, 1, 1)

    dx = (xs - cx) / fx
    dy = (ys - cy) / fy

    # denominator sqrt(dx^2 + dy^2 + 1)
    denom = torch.sqrt(dx * dx + dy * dy + 1.0)

    # final depth (T,1,H,W)
    depth = distance / (denom + 1e-8)

    return depth