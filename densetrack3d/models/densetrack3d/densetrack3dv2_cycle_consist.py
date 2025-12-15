
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float
from torch import Tensor
from typing import Optional
import random

import os
from densetrack3d.models.densetrack3d.blocks import BasicEncoder, Mlp
from densetrack3d.models.densetrack3d.corr4d_blocks import Corr4DMLP, Corr4DCNN, Corr4DCNN2, get_cmdtop_params_from_diameter
# from densetrack3d.models.densetrack3d.update_transformer import EfficientUpdateFormer
from densetrack3d.models.densetrack3d.update_transformer_random_group import EfficientUpdateFormer
from densetrack3d.models.densetrack3d.upsample_transformer import UpsampleTransformerAlibi
from densetrack3d.models.densetrack3d.interpolator import LearnableInterpolator
from densetrack3d.models.embeddings import get_1d_sincos_pos_embed_from_grid, get_2d_embedding, get_2d_sincos_pos_embed
from densetrack3d.models.model_utils import (
    bilinear_sampler,
    get_grid,
    sample_features4d,
    sample_features5d,
    smart_cat,
)
from densetrack3d.utils.dinov3_encoder import DINOv3_Encoder
from densetrack3d.utils.chrono_track_model import LocoTrack
from densetrack3d.utils.dino_encoder import DINO_Encoder


VideoType = Float[Tensor, "b t c h w"]

torch.manual_seed(0)

vis_count = 0

def custom_interp(chosen_tensor, unchosen_indices, sample_factor):
    weight1 = (unchosen_indices % sample_factor).float() / sample_factor
    weight2 = 1.0 - weight1

    ind1 = unchosen_indices // sample_factor
    ind2 = ind1 + 1

    interp_tensor = chosen_tensor[..., ind1] * weight1[None, None] + chosen_tensor[..., ind2] * weight2[None, None]
    
    return interp_tensor

# Corr4DCNN = torch.compile(Corr4DCNN)

class DenseTrack3DV2(nn.Module):
    def __init__(
        self,
        window_len=8,
        stride=4,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        only_learnup=False,
        upsample_factor=4,
        flash=False,
        coarse_to_fine_dense=True,
        dH=16, dW=24,
        freeze_fnet=False,
        freeze_corr4DCNN=False,
        num_traj_groups=None,
        use_dino="",
        dino_size="(32, 32)",
        freeze_dino=False,
        use_deconv1=False,
        use_deconv2=False,
        use_anyup=False,
        use_merge_conv=False,
        merge_dino_method="cat",
        merge_dino_corr_method="mask", # mask, extra
        radius_supp="3",
        stride_supp="1",
        radius_corr="3",
        stride_corr="1",
        cycle_loss=False,
        point_semantic="",
    ):
        super().__init__()
        self.window_len = window_len
        self.stride = stride
        self.hidden_dim = 256
        self.latent_dim = 128
        self.upsample_factor = upsample_factor
        self.add_space_attn = add_space_attn
        self.coarse_to_fine_dense = coarse_to_fine_dense
        self.coarse_to_fine_dense_temporal = False

        self.dH_train = dH
        self.dW_train = dW
        self.freeze_fnet = freeze_fnet
        self.freeze_corr4DCNN = freeze_corr4DCNN
        
        self.fnet = BasicEncoder(
            input_dim=3, 
            output_dim=self.latent_dim,
            stride=self.stride,
        )
        # cycle loss
        self.cycle_loss = cycle_loss

        ###############
        # define dino #
        ###############
        self.use_dino = use_dino
        
        dino_size = eval(dino_size)
        self.dino_size = dino_size
        self.use_merge_conv = use_merge_conv
        if self.use_merge_conv:
            merge_dino_method = "merge_conv"
        self.merge_dino_method = merge_dino_method
        self.merge_dino_corr_method = merge_dino_corr_method

        if self.use_dino in ["dinov2_vitl14", "dinov3_vitl16", "dinov2_vitb14", "dinov3_vitb16"]:
            self.dino_net = DINO_Encoder(
                model_name=use_dino, patch_size=dino_size, 
                use_deconv1=use_deconv1, use_deconv2=use_deconv2,
                use_anyup=use_anyup,
            )
            
        elif self.use_dino == "chrono":
            model_kwargs = {"dino_size": "base", "dino_reg": False, "adapter_intermed_channels": 128}
            self.dino_net = LocoTrack(**model_kwargs)
            state_dict = torch.load("/mnt/shared-storage-user/idc2-shared/binyanrui/pretrained_models/chrono_tracking/chrono_base.ckpt", map_location=torch.device('cpu'))
            state_dict = state_dict["state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k in ["model.occ_linear.weight", "model.occ_linear.bias"]:
                    continue
                if k.startswith("model."):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
            missing_keys, unexpected_keys = self.dino_net.load_state_dict(state_dict, strict=False)
            print(f"unexpected_keys: {unexpected_keys}")
            print(f"missing_keys: {missing_keys}")
            # self.dino_net = self.dino_net.to(self.fnet.device)
        
        if not use_dino:
            dino_c_dim = 0
            # self.input_dim = 1032  
        elif use_dino in ["dinov2_vitb14", "dinov3_vitb16", "dinov2_vitl14", "dinov3_vitl16"]:
            dino_c_dim = self.dino_net.C
            # self.input_dim = 1032 + self.dino_net.C
            # self.latent_dim += self.dino_net.C
        elif use_dino in ["chrono"]:
            dino_c_dim = 768
            # self.input_dim = 1032 + 768
            # self.latent_dim += 768
        else:
            raise

        if use_dino in ["dinov2_vitb14", "dinov3_vitb16", "dinov2_vitl14", "dinov3_vitl16"]:
            def make_layer(in_c, out_c):
                merge_conv = torch.nn.Linear(in_c, out_c, bias=False)
                merge_conv.weight.requires_grad = False
                merge_conv.weight.zero_()
                merge_conv.weight[:, :out_c] = torch.eye(out_c)
                merge_conv.weight.requires_grad = True
                return merge_conv
            if self.merge_dino_method == "merge_conv":
                in_c = self.latent_dim + dino_c_dim
                self.merge_conv0 = make_layer(in_c, self.latent_dim)
                self.merge_conv1 = make_layer(in_c, self.latent_dim)
                self.merge_conv2 = make_layer(64 + dino_c_dim, 64)
                self.input_dim = 1032
            if self.merge_dino_method == "merge_conv_low_res":
                in_c = self.latent_dim + dino_c_dim
                # self.merge_conv = make_layer(in_c, 64)
                self.merge_conv = torch.nn.Linear(dino_c_dim, self.latent_dim, bias=False)
                self.input_dim = 1032

            elif self.merge_dino_method in ["corr", "track_feat", "corr_track_feat"]:
                if self.merge_dino_method in ["track_feat", "corr_track_feat"]:
                    in_c = self.latent_dim + dino_c_dim
                    self.track_feat_linear = make_layer(in_c, self.latent_dim)
                
                self.input_dim = 1032
                if self.merge_dino_corr_method == "extra":
                    # cmdtop_params = {
                    #     "in_channel": 64,
                    #     "out_channels": (64, 128, 128),
                    #     "kernel_shapes": (3, 3, 2),
                    #     "strides": (2, 2, 2),
                    # }
                    # self.cmdtop_dino = Corr4DCNN(**cmdtop_params)
                    self.input_dim = 1032 + 256
                
            else:
                self.input_dim = 1032 + dino_c_dim
                self.latent_dim += dino_c_dim
        else:
            self.input_dim = 1032
        
        self.point_semantic = point_semantic
        if self.point_semantic:
            self.input_dim += 128
            if self.point_semantic == "dino":
                in_c = self.dino_net.C
                self.point_semantic_net = torch.nn.Linear(in_c, 128, bias=False)


        ###############
        # define dino #
        ###############

        self.updateformer = EfficientUpdateFormer(
            num_blocks=6, #
            input_dim=self.input_dim,
            hidden_size=384,
            output_dim=self.latent_dim + 3,
            mlp_ratio=4.0,
            add_space_attn=add_space_attn,
            num_virtual_tracks=num_virtual_tracks,
            flash=flash,
            use_local_attn=False,
            num_traj_groups=num_traj_groups,
        )

        self.num_virtual_tracks = num_virtual_tracks
        self.model_resolution = model_resolution

        time_grid = torch.linspace(0, window_len - 1, window_len).reshape(1, window_len, 1)

        self.register_buffer("time_emb", get_1d_sincos_pos_embed_from_grid(self.input_dim, time_grid[0]))
        
        self.register_buffer(
            "pos_emb",
            get_2d_sincos_pos_embed(
                embed_dim=self.input_dim,
                grid_size=(
                    model_resolution[0] // stride,
                    model_resolution[1] // stride,
                ),
            ),
            persistent=False,
        )

        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.track_feat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.vis_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
        )

        self.conf_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
        )

        self.upsample_kernel_size = 5

        self.upsample_transformer_prev = UpsampleTransformerAlibi(
            kernel_size=self.upsample_kernel_size, # kernel_size=3, # 
            stride=self.stride,
            latent_dim=self.latent_dim,
            num_attn_blocks=2,
            upsample_factor=self.upsample_factor,
        )

        self.upsample_transformer = UpsampleTransformerAlibi(
            kernel_size=self.upsample_kernel_size, # kernel_size=3, # 
            stride=self.stride,
            latent_dim=self.latent_dim,
            num_attn_blocks=2,
            upsample_factor=self.upsample_factor,
        )

        if self.coarse_to_fine_dense:

            self.inter_up = LearnableInterpolator(
                kernel_size=2,
                latent_dim=self.latent_dim,
                # hiddent_dim=self.latent_dim
            )


        self.initialize_up_weight()

        def init_radius_or_stride(x):
            x = eval(x)
            if isinstance(x, int):
                x = [x] * 3
            assert isinstance(x, list)
            return x
        self.radius_supps = init_radius_or_stride(radius_supp)
        self.stride_supps = init_radius_or_stride(stride_supp)
        self.radius_corrs = init_radius_or_stride(radius_corr)
        self.stride_corrs = init_radius_or_stride(stride_corr)
        def get_delta(radius, stride):
            num_point = int(2 * radius / stride + 1)
            dx = torch.linspace(-radius, radius, num_point)
            dy = torch.linspace(-radius, radius, num_point)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)
            delta = delta.view(num_point, num_point, 2)
            return delta
        delta_corrs = []
        for r, s in zip(self.radius_corrs, self.stride_corrs):
            delta_corr = get_delta(r, s)
            delta_corrs.append(delta_corr)
        self.delta_corrs = delta_corrs
        # self.register_buffer("delta_corr", delta_corr)
        delta_supps = []
        for r, s in zip(self.radius_supps, self.stride_supps):
            delta_supp = get_delta(r, s)
            delta_supps.append(delta_supp)
        self.delta_supps = delta_supps
        # delta_supp = get_delta(self.radius_supp, self.stride_supp)
        # self.register_buffer("delta_supp", delta_supp)
        print(delta_corrs[0])
        print(delta_supps[0])
        cmdtop_params = {
            "in_channel": 64,
            "out_channels": (64, 128, 128),
            "kernel_shapes": (3, 3, 2),
            "strides": (2, 2, 2),
        }
        l = []
        for i in range(len(delta_corrs)):
            diameter1 = int(2 * self.radius_corrs[i] / self.stride_corrs[i] + 1)
            diameter2 = int(2 * self.radius_supps[i] / self.stride_supps[i] + 1)
            if diameter1 == 7 and diameter2 == 7:
                linear_in_c = diameter1 ** 2
                l.append(Corr4DCNN(linear_in_c=linear_in_c, **cmdtop_params))
            else:
                linear_in_c1 = diameter1 ** 2
                linear_in_c2 = diameter2 ** 2
                cmdtop_params1 = get_cmdtop_params_from_diameter(diameter1)
                cmdtop_params2 = get_cmdtop_params_from_diameter(diameter2, postfix="2")
                l.append(Corr4DCNN2(linear_in_c1=linear_in_c2, linear_in_c2=linear_in_c1, **cmdtop_params1, **cmdtop_params2))
        self.cmdtop = nn.ModuleList(l)

        self.only_learnup = only_learnup

        self.fixed_modules = []

    def train(self, mode=True):
        super().train(mode)
        if self.only_learnup:
            for mod in self.fixed_modules:
                mod = getattr(self, mod)
                mod.eval()

    def get_latent_sim(self):
        latent = rearrange(self.updateformer.virual_tracks, "1 n 1 c -> n c")  # N C

        latent_norm = F.normalize(latent, p=2, dim=-1)  # [1xKnovelxC]
        latent_sim = einsum(latent_norm, latent_norm, "n c, m c -> n m")  # [KnovelxKnovel]

        return latent_sim

    def initialize_up_weight(self):
        def _basic_init(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                # torch.nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def upsample_with_mask(
        self, inp: Float[Tensor, "b c h_down w_down"], mask: Float[Tensor, "b 1 k h_up w_up"], current_upsample_factor: Optional[int] = None, current_upsample_kernel_size: Optional[int] = None,
    ) -> Float[Tensor, "b c h_up w_up"]:
        """Upsample flow field [H/P, W/P, 2] -> [H, W, 2] using convex combination"""
        H, W = inp.shape[-2:]

        upsample_factor = self.upsample_factor if current_upsample_factor is None else current_upsample_factor
        upsample_kernel_size = self.upsample_kernel_size if current_upsample_kernel_size is None else current_upsample_kernel_size

        up_inp = F.unfold(
            inp, [upsample_kernel_size, upsample_kernel_size], padding=(upsample_kernel_size - 1) // 2
        )
        up_inp = rearrange(up_inp, "b c (h w) -> b c h w", h=H, w=W)
        up_inp = F.interpolate(up_inp, scale_factor=upsample_factor, mode="nearest")
        up_inp = rearrange(
            up_inp, "b (c i j) h w -> b c (i j) h w", i=upsample_kernel_size, j=upsample_kernel_size
        )

        up_inp = torch.sum(mask * up_inp, dim=2)
        return up_inp

    def upsample_with_mask2(
        self, inp: Float[Tensor, "b c h_down w_down"], mask: Float[Tensor, "b 1 k h_up w_up"], current_upsample_factor: Optional[int] = None, current_upsample_kernel_size: Optional[int] = None,
    ) -> Float[Tensor, "b c h_up w_up"]:
        """Upsample flow field [H/P, W/P, 2] -> [H, W, 2] using convex combination"""
        H, W = inp.shape[-2:]

        upsample_factor = self.upsample_factor if current_upsample_factor is None else current_upsample_factor
        upsample_kernel_size = self.upsample_kernel_size if current_upsample_kernel_size is None else current_upsample_kernel_size

        inp = F.pad(inp, (0, 1, 0, 1), mode="replicate")
        up_inp = F.unfold(
            inp, [upsample_kernel_size, upsample_kernel_size], padding=0
        )
        up_inp = rearrange(up_inp, "b c (h w) -> b c h w", h=H, w=W)
        up_inp = F.interpolate(up_inp, scale_factor=upsample_factor, mode="nearest")
        up_inp = rearrange(
            up_inp, "b (c i j) h w -> b c (i j) h w", i=upsample_kernel_size, j=upsample_kernel_size
        )

        up_inp = torch.sum(mask * up_inp, dim=2)
        return up_inp
    
    def get_single_corr_depth(
        self,
        depths: Float[Tensor, "b t 1 h w"],
        coords: Float[Tensor, "b t n c"],
        coord_depths: Float[Tensor, "b t n 1"],
        mode: str = "bilinear",
    ) -> Float[Tensor, "*"]:

        B, S, N = coords.shape[:3]
        local_queried_frames = torch.arange(0, S, device=coords.device).long()
        local_queried_frames = repeat(local_queried_frames, "s -> b s n 1", b=B, n=N)
        depth_queried_coords = torch.cat(
            [local_queried_frames, coords],
            dim=-1,
        )  # B S N 3

        sample_depth = sample_features5d(depths, depth_queried_coords, mode=mode)  # B, S, N, 1
        dcorrs = sample_depth / coord_depths

        return dcorrs
    
    def split_fmaps_dinos(self, supp_x):
        # fmaps, dinos = x[..., :-self.dino_net.C], x[..., -self.dino_net.C:]
        fmaps_supp, dinos_supp = supp_x[..., :-self.dino_net.C], supp_x[..., -self.dino_net.C:]
        return fmaps_supp, dinos_supp
    
    def get_4Dcorr_features_dinos(
            self,
            dino_maps,
            coords,
            supp_track_dinos,
    ):
        B, S, N = coords.shape[:3]
        diameter = int(2 * self.radius_corrs[0] / self.stride_corrs[0] + 1)
        
        lvl, supp_track_feat_ = 1, supp_track_dinos

        centroid_lvl = coords.reshape(B * S, N, 1, 1, 2) / 2 ** (lvl - 1)
        coords_lvl = centroid_lvl + self.delta_corrs[0][None, None].to(centroid_lvl.device)  # (B S) N (2r+1) (2r+1) 2

        _, _, C_, H_, W_ = dino_maps.shape
        
        sample_tgt_feat = bilinear_sampler(
            dino_maps.reshape(B * S, -1, H_, W_),
            coords_lvl.reshape(B * S, N, diameter * diameter, 2),
            padding_mode="border",
        )

        sample_tgt_feat = sample_tgt_feat.view(B, S, -1, N, diameter, diameter)

        patches_input = einsum(sample_tgt_feat, supp_track_feat_, "b s c n h w, b n i j c -> b s n h w i j")
        patches_input = patches_input / torch.sqrt(torch.tensor(C_).float())
        patches_input = rearrange(patches_input, "b s n h w i j -> (b s n) h w i j")

        return patches_input
    
    def get_4dcorr_features_dino_as_mask(
        self,
        fmaps_pyramid: tuple[VideoType, ...],
        coords: Float[Tensor, "b t n c"],
        supp_track_feat: tuple[Float[Tensor, "*"], ...],
        dino_maps: Tensor,
        supp_track_dinos: Tensor
    ) -> Float[Tensor, "*"]:

        B, S, N = coords.shape[:3]

        corrs_pyr = []
        # dino 
        dino_corr = self.get_4Dcorr_features_dinos(dino_maps, coords, supp_track_dinos)
        if self.merge_dino_corr_method == "mask":
            # mask
            dino_corr_mask = torch.nn.functional.sigmoid(dino_corr) * 2 # 1 at the begining
        elif self.merge_dino_corr_method == "extra":
            dino_corr_emd = self.cmdtop_dino(dino_corr)
            dino_corr_emd = rearrange(dino_corr_emd, "(b s n) c -> b s n c", b=B, s=S)
            corrs_pyr.append(dino_corr_emd)
        for lvl, supp_track_feat_ in enumerate(supp_track_feat):
            diameter =  int(2 * self.radius_corrs[lvl] / self.stride_corrs[lvl] + 1)
            centroid_lvl = coords.reshape(B * S, N, 1, 1, 2) / 2 ** (lvl - 1)
            coords_lvl = centroid_lvl + self.delta_corrs[lvl][None, None].to(centroid_lvl.device)  # (B S) N (2r+1) (2r+1) 2

            _, _, C_, H_, W_ = fmaps_pyramid[lvl].shape
            
            sample_tgt_feat = bilinear_sampler(
                fmaps_pyramid[lvl].reshape(B * S, -1, H_, W_),
                coords_lvl.reshape(B * S, N, diameter * diameter, 2),
                padding_mode="border",
            )

            sample_tgt_feat = sample_tgt_feat.view(B, S, -1, N, diameter, diameter)

            patches_input = einsum(sample_tgt_feat, supp_track_feat_, "b s c n h w, b n i j c -> b s n h w i j")
            patches_input = patches_input / torch.sqrt(torch.tensor(C_).float())
            patches_input = rearrange(patches_input, "b s n h w i j -> (b s n) h w i j")

            if self.merge_dino_corr_method == "mask":
                # this means area with out semantic consist will be ignored
                patches_input = patches_input * dino_corr_mask

            patches_emb = self.cmdtop[lvl](patches_input)

            patches = rearrange(patches_emb, "(b s n) c -> b s n c", b=B, s=S)

            corrs_pyr.append(patches)
        fcorrs = torch.cat(corrs_pyr, dim=-1)  # B S N C

        return fcorrs

    def get_4dcorr_features(
        self,
        fmaps_pyramid: tuple[VideoType, ...],
        coords: Float[Tensor, "b t n c"],
        supp_track_feat: tuple[Float[Tensor, "*"], ...],
    ) -> Float[Tensor, "*"]:

        B, S, N = coords.shape[:3]
        
        corrs_pyr = []
        for lvl, supp_track_feat_ in enumerate(supp_track_feat):
            diameter = int(2 * self.radius_corrs[lvl] / self.stride_corrs[lvl] + 1)

            if lvl < 3:
                centroid_lvl = coords.reshape(B * S, N, 1, 1, 2) / 2 ** (lvl - 1)
            else:
                _, _, _, dinoH, dinoW = fmaps_pyramid[lvl].shape
                fH, fW = [x // self.stride for x in self.model_resolution]
                assert fH / dinoH == fW / dinoW
                centroid_lvl = coords.reshape(B * S, N, 1, 1, 2) / fH * dinoH

            coords_lvl = centroid_lvl + self.delta_corrs[lvl][None, None].to(centroid_lvl.device)  # (B S) N (2r+1) (2r+1) 2

            _, _, C_, H_, W_ = fmaps_pyramid[lvl].shape
            
            sample_tgt_feat = bilinear_sampler(
                fmaps_pyramid[lvl].reshape(B * S, -1, H_, W_),
                coords_lvl.reshape(B * S, N, diameter * diameter, 2),
                padding_mode="border",
            )
            
            sample_tgt_feat = sample_tgt_feat.view(B, S, -1, N, diameter, diameter)
            
            try:
                patches_input = einsum(sample_tgt_feat, supp_track_feat_, "b s c n h w, b n i j c -> b s n h w i j")
                patches_input = patches_input / torch.sqrt(torch.tensor(C_).float())
                patches_input = rearrange(patches_input, "b s n h w i j -> (b s n) h w i j")
            except:
                print(sample_tgt_feat.shape, supp_track_feat_.shape)
                import pdb
                pdb.set_trace()
                raise
            
            patches_emb = self.cmdtop[lvl](patches_input)

            patches = rearrange(patches_emb, "(b s n) c -> b s n c", b=B, s=S)

            corrs_pyr.append(patches)
        fcorrs = torch.cat(corrs_pyr, dim=-1)  # B S N C

        return fcorrs

    def get_track_feat(
        self,
        fmaps: VideoType,
        fmaps_pyramid: tuple[VideoType, ...],
        queried_frames: Float[Tensor, "b n"],
        queried_coords: Float[Tensor, "b n c"],
        num_levels: int = 3,
        radius: int = 3,
        dino_fmaps=None,
    ) -> tuple[Float[Tensor, "*"], tuple[Float[Tensor, "*"], ...]]:

        B = fmaps.shape[0]
        N = queried_coords.shape[1]

        sample_frames = queried_frames[:, None, :, None]
        sample_coords = torch.cat([sample_frames, queried_coords[:, None]], dim=-1)
        sample_track_feats = sample_features5d(fmaps, sample_coords)

        # dino
        # here use zero conv to merge with original feats
        if self.merge_dino_method in ["track_feat", "corr_track_feat"]:
            sample_track_dinos = sample_features5d(dino_fmaps, sample_coords)
            sample_track_feats = torch.cat([sample_track_feats, sample_track_dinos], dim=-1)
            sample_track_feats = self.track_feat_linear(sample_track_feats)
            
        
        supp_track_feats_pyramid = []
        for lvl in range(num_levels):
            centroid_lvl = queried_coords.reshape(B * N, 1, 1, 2) / 2 ** (lvl - 1)
            coords_lvl = centroid_lvl + self.delta_supps[lvl][None].to(centroid_lvl.device)
            diameter = int(2 * self.radius_supps[lvl] / self.stride_supps[lvl] + 1)
            coords_lvl = coords_lvl.reshape(B, 1, N * diameter * diameter, 2)

            sample_frames = queried_frames[:, None, :, None, None, None].expand(
                B, 1, N, diameter, diameter, 1
            )
            sample_frames = sample_frames.reshape(B, 1, N * diameter * diameter, 1)
            sample_coords = torch.cat(
                [
                    sample_frames,
                    coords_lvl,
                ],
                dim=-1,
            )  # B 1 N 3

            supp_track_feats = sample_features5d(
                fmaps_pyramid[lvl],
                sample_coords,
            )
            supp_track_feats = supp_track_feats.view(B, N, diameter, diameter, -1)

            supp_track_feats_pyramid.append(supp_track_feats)
        
        # dino
        # supp_track_dinos will be used later. As a attn or an extra corr
        if self.merge_dino_method in ["corr", "corr_track_feat"]: # main feature
            _, _, _, dinoH, dinoW = dino_fmaps.shape
            fH, fW = [x // self.stride for x in self.model_resolution]
            assert fH / dinoH == fW / dinoW

            diameter = int(2 * self.radius_supps[num_levels] / self.stride_supps[num_levels] + 1)

            centroid_lvl = queried_coords.reshape(B * N, 1, 1, 2) / fH * dinoH
            coords_lvl = centroid_lvl + self.delta_supps[num_levels][None].to(centroid_lvl.device)
            coords_lvl = coords_lvl.reshape(B, 1, N * diameter * diameter, 2)

            sample_frames = queried_frames[:, None, :, None, None, None].expand(
                B, 1, N, diameter, diameter, 1
            )
            sample_frames = sample_frames.reshape(B, 1, N * diameter * diameter, 1)
            sample_coords = torch.cat(
                [
                    sample_frames,
                    coords_lvl,
                ],
                dim=-1,
            )  # B 1 N 3

            supp_track_feats_dino = sample_features5d(
                dino_fmaps,
                sample_coords,
            )
            supp_track_feats_dino = supp_track_feats_dino.view(B, N, diameter, diameter, -1)
            supp_track_feats_pyramid.append(supp_track_feats_dino)

        return sample_track_feats, supp_track_feats_pyramid

    def get_dense_track_feat(
        self,
        fmaps: VideoType,
        fmaps_pyramid: tuple[VideoType, ...],
        dense_coords: Float[Tensor, "b n c"],
        num_levels: int = 3,
        radius: int = 3,
        dino_fmaps=None,
    ) -> tuple[Float[Tensor, "*"], tuple[Float[Tensor, "*"], ...]]:

        B, N = dense_coords.shape[:2]

        sample_track_feats = sample_features4d(fmaps[:, 0], dense_coords)
        # dino
        # here use zero conv to merge with original feats
        if self.merge_dino_method in ["track_feat", "corr_track_feat"]:
            sample_track_dinos = sample_features4d(dino_fmaps[:, 0], dense_coords)
            sample_track_feats = torch.cat([sample_track_feats, sample_track_dinos], dim=-1)
            sample_track_feats = self.track_feat_linear(sample_track_feats)

        supp_track_feats_pyramid = []
        for lvl in range(num_levels):
            centroid_lvl = rearrange(dense_coords, "b n c -> (b n) () () c") / 2 ** (lvl - 1)

            coords_lvl = centroid_lvl + self.delta_supps[lvl][None].to(centroid_lvl.device)
            coords_lvl = rearrange(coords_lvl, "(b n) r1 r2 c -> b (n r1 r2) c", b=B, n=N)

            supp_track_feats = sample_features4d(fmaps_pyramid[lvl][:, 0], coords_lvl)
            diameter = int(2 * self.radius_supps[lvl] / self.stride_supps[lvl] + 1)
            supp_track_feats = rearrange(
                supp_track_feats, "b (n r1 r2) c -> b n r1 r2 c", n=N, r1=diameter, r2=diameter
            )
            supp_track_feats_pyramid.append(supp_track_feats)
        
        # dino
        # supp_track_dinos will be used later. As a attn or an extra corr
        if self.merge_dino_method in ["corr", "corr_track_feat"]: # main feature
            _, _, _, dinoH, dinoW = dino_fmaps.shape
            fH, fW = [x // self.stride for x in self.model_resolution]
            assert fH / dinoH == fW / dinoW

            diameter = int(2 * self.radius_supps[num_levels] / self.stride_supps[num_levels] + 1)

            centroid_lvl = rearrange(dense_coords, "b n c -> (b n) () () c") / fH * dinoH
            coords_lvl = centroid_lvl + self.delta_supps[num_levels][None].to(centroid_lvl.device)
            coords_lvl = rearrange(coords_lvl, "(b n) r1 r2 c -> b (n r1 r2) c", b=B, n=N)

            supp_track_feats_dino = sample_features4d(dino_fmaps[:, 0], coords_lvl)

            supp_track_feats_dino = rearrange(
                supp_track_feats_dino, "b (n r1 r2) c -> b n r1 r2 c", n=N, r1=diameter, r2=diameter
            )
            supp_track_feats_pyramid.append(supp_track_feats_dino)

        return sample_track_feats, supp_track_feats_pyramid

    def update_step(
        self,
        fmaps_pyramid: tuple[VideoType, ...],
        depthmaps: Float[Tensor, "b t 1 h w"],
        coords: Float[Tensor, "b t n c"],
        coord_depths: Float[Tensor, "b t n 1"],
        track_mask_vis: Float[Tensor, "b t n c"],
        # vis: Float[Tensor, "b t n"],
        # conf: Float[Tensor, "b t n"],
        track_feat: Float[Tensor, "b t n c"],
        supp_track_feat: tuple[Float[Tensor, "b t n c"], ...],
        # track_mask: Float[Tensor, "b t n"],
        attention_mask: Bool[Tensor, "b t n"],
        # iters: int = 4,
        use_efficient_global_attn: bool = False,
        use_local_attn: bool = False,
        time_emb = None
    ):
        B, S, N = coords.shape[:3]
        # NOTE Prepare input to transformer
        if self.freeze_corr4DCNN:
            with torch.no_grad():
                fcorrs = self.get_4dcorr_features(fmaps_pyramid, coords, supp_track_feat)
        else:
            if False and self.merge_dino_method in ["corr", "corr_track_feat"]:
                dino_maps = fmaps_pyramid[-1]
                fmaps_pyramid_ = fmaps_pyramid[:-1]

                fmaps_supp, dinos_supp = self.split_fmaps_dinos(supp_track_feat[1])
                new_supp_track_feat = [x for x in supp_track_feat]
                new_supp_track_feat[1] = fmaps_supp
                fcorrs = self.get_4dcorr_features_dino_as_mask(
                    fmaps_pyramid_, coords, tuple(new_supp_track_feat),
                    dino_maps, dinos_supp
                )
            else:
                fcorrs = self.get_4dcorr_features(fmaps_pyramid, coords, supp_track_feat)

        with torch.no_grad():
            dcorrs = self.get_single_corr_depth(depthmaps, coords, coord_depths)

        track_feat_cur = None
        if self.point_semantic:
            # use queried feature instead
            _h = fmaps_pyramid[-1].shape[3]
            if _h == 192:
                sample_coords = coords * 2.
            elif _h == 96:
                sample_coords = coords
            elif _h == 48:
                sample_coords = coords / 2.
            else:
                raise
            # sample_tgt_feat = bilinear_sampler(
            #     fmaps_pyramid[lvl].reshape(B * S, -1, H_, W_),
            #     coords_lvl.reshape(B * S, N, diameter * diameter, 2),
            #     padding_mode="border",
            # )
            b, s = fmaps_pyramid[-1].shape[:2]
            feats_reshape = rearrange(fmaps_pyramid[-1], "b s c h w -> (b s) c h w")
            sample_coords = rearrange(sample_coords, "b s n c -> (b s) n 1 c")
            track_feat_cur = bilinear_sampler(feats_reshape, sample_coords)
            track_feat_cur = rearrange(track_feat_cur, "(b s) c n 1 -> b s n c", b=b, s=s)
            if self.point_semantic == "dino":
                track_feat_cur = self.point_semantic_net(track_feat_cur)
        

        # Get the 2D flow embeddings
        flows_2d = coords - coords[:, 0:1]
        flows_2d_emb = get_2d_embedding(flows_2d, 64, cat_coords=True)  # N S E
        flows_3d = coord_depths / coord_depths[:, 0:1]

        
        # for x in ["flows_2d_emb", "flows_2d", "flows_3d", "fcorrs", "dcorrs", "track_feat", "track_mask_vis"]:
        #     print(f"{x}", eval(x).shape)
        
        if fcorrs.shape[-1] > 768:
            transformer_input = torch.cat(
                [
                    flows_2d_emb,
                    flows_2d,
                    flows_3d,
                    fcorrs[..., :768],
                    dcorrs,
                    track_feat,
                    track_mask_vis,
                    fcorrs[..., 768:],
                ],
                dim=-1,
            )
        else:
            transformer_input = torch.cat(
                [
                    flows_2d_emb,
                    flows_2d,
                    flows_3d,
                    fcorrs,
                    dcorrs,
                    track_feat,
                    track_mask_vis,
                ],
                dim=-1,
            )
        if self.point_semantic:   
            transformer_input = torch.cat(
                [
                    transformer_input,
                    track_feat_cur
                ],
                dim=-1
            )


        pos_emb = sample_features4d(self.pos_emb.repeat(B, 1, 1, 1), coords[:, 0])
        
        # x = transformer_input + pos_emb[:, None] + self.time_emb[:, :, None]
        # x = transformer_input + pos_emb[:, None] + self.time_emb[:, :S, None]
        
        x = transformer_input + pos_emb[:, None] + time_emb[:, :, None]

        # NOTE Transformer part
        delta = self.updateformer(
            input_tensor=x,
            attn_mask=attention_mask,
            n_sparse=self.N_sparse,
            dH=self.dH,
            dW=self.dW,
            use_efficient_global_attn=use_efficient_global_attn,
            use_local_attn=False
        )

        # print("debug", delta.shape, x.shape)

        return delta

    def temporal_compress(
        self,
        fmaps_pyramid: tuple[VideoType, ...],
        depthmaps: Float[Tensor, "b t 1 h w"],
        coords: Float[Tensor, "b t n c"],
        coord_depths: Float[Tensor, "b t n 1"],
        track_mask_vis: Float[Tensor, "b t n c"],
        track_feat: Float[Tensor, "b t n c"],
        supp_track_feat: tuple[Float[Tensor, "b t n c"], ...],
        attention_mask: Bool[Tensor, "b t n"],
        use_efficient_global_attn: bool = False,
        iteration: int = 0,
        iters: int = 4,
    ):
        B, S, N = coords.shape[:3]

        if (iteration < iters - 1) and self.coarse_to_fine_dense_temporal:

            downsample_time_factor = 2


            timesteps = torch.arange(S, device=coords.device).long()
            chosen_timestep_mask = torch.zeros(S, device=coords.device).bool()
            chosen_timesteps = torch.cat([timesteps[::downsample_time_factor], timesteps[-1:]], dim=0) # NOTE always include last frame
            chosen_timestep_mask[chosen_timesteps] = True
            unchosen_timesteps = torch.arange(S, device=coords.device).long()[~chosen_timestep_mask]

            # vis_new = vis[:, chosen_timesteps]
            # conf_new = conf[:, chosen_timesteps]
            coords_new = coords[:, chosen_timesteps]
            coord_depths_new = coord_depths[:, chosen_timesteps]
            track_mask_vis_new = track_mask_vis[:, chosen_timesteps]
            attention_mask_new = attention_mask[:, chosen_timesteps]
            track_feat_new = track_feat[:, chosen_timesteps]

            fmaps_pyramid_new = [
                fmaps_pyramid[lvl][:, chosen_timesteps] for lvl in range(len(fmaps_pyramid))
            ]
            depthmaps_new = depthmaps[:, chosen_timesteps]


            delta_new = self.update_step(
                fmaps_pyramid_new,
                depthmaps_new,
                coords_new,
                coord_depths_new,
                track_mask_vis_new,
                track_feat_new,
                supp_track_feat,
                attention_mask_new,
                use_efficient_global_attn,
                time_emb=self.time_emb[:, chosen_timesteps] # NOTE use time_emb for the chosen timesteps
            )
            delta_new = rearrange(delta_new, "b t n c -> 1 (b n c) t")

            delta = torch.zeros((delta_new.shape[0], delta_new.shape[1], S), device=delta_new.device, requires_grad=delta_new.requires_grad)
            delta[:, :, chosen_timesteps.long()] = delta_new

            chosen_timesteps1 = chosen_timesteps[:-1] # remove last chunk
            delta_new1 = delta_new[:, :, :-1]
            unchosen_timesteps1 = unchosen_timesteps[unchosen_timesteps<chosen_timesteps1[-1]] # remove timestep > chosen_timesteps1[-1]
            
            delta_interp1 = custom_interp(delta_new1, unchosen_timesteps1, downsample_time_factor)
            delta[:, :, unchosen_timesteps1.long()] = delta_interp1


            chosen_timesteps2 = chosen_timesteps[-2:] # last chunk
            delta_new2 = delta_new[:, :, -2:]
            unchosen_timesteps2 = unchosen_timesteps[unchosen_timesteps>chosen_timesteps1[-1]] # remove timestep > chosen_timesteps1[-1]

            if len(unchosen_timesteps2) > 0:

                delta_interp2 = custom_interp(
                    delta_new2, 
                    unchosen_timesteps2 - chosen_timesteps2[0], 
                    chosen_timesteps2[1] - chosen_timesteps2[0]
                )
                delta[:, :, unchosen_timesteps2.long()] = delta_interp2

            delta = rearrange(delta, '1 (b n c) t -> b t n c', b=B, n=N)


        else:
            delta = self.update_step(
                fmaps_pyramid,
                depthmaps,
                coords,
                coord_depths,
                track_mask_vis,
                track_feat,
                supp_track_feat,
                attention_mask,
                use_efficient_global_attn,
                time_emb=self.time_emb,
            )

        return delta
    
    def forward_window(
        self,
        fmaps_pyramid: tuple[VideoType, ...],
        depthmaps: Float[Tensor, "b t 1 h w"],
        coords: Float[Tensor, "b t n c"],
        coord_depths: Float[Tensor, "b t n 1"],
        vis: Float[Tensor, "b t n"],
        conf: Float[Tensor, "b t n"],
        track_feat: Float[Tensor, "b t n c"],
        supp_track_feat: tuple[Float[Tensor, "b t n c"], ...],
        track_mask: Float[Tensor, "b t n"],
        attention_mask: Bool[Tensor, "b t n"],
        iters: int = 4,
        use_efficient_global_attn: bool = False,
        inter_up_mask_dict: Optional[dict] = None,
    ) -> tuple[
        Float[Tensor, "b t n c"],
        Float[Tensor, "b t n 1"],
        Float[Tensor, "b t n"],
        Float[Tensor, "b t n"],
        Float[Tensor, "b t n c"],
    ]:

        B, S, N = coords.shape[:3]

        track_mask_vis = torch.cat([track_mask, vis], dim=-1)  # b s n c

        coord_preds = []
        coord_depth_preds = []

        # FIXME fix depth
        depthmaps[depthmaps < 1e-2] = 1e-2

        for iteration in range(iters):

            coords = coords.detach()  # B S N 3
            coord_depths = coord_depths.detach()
            coord_depths[coord_depths < 1e-2] = 1e-2


            compress_spatial = False
            if self.training:
                if (iteration < iters - 1 and iteration < 2) and self.use_dense and self.coarse_to_fine_dense:
                    compress_spatial = True

                    if iteration == 0:
                        downsample_factor = 8 if random.random() < 0.5 else 4
                    elif iteration == 1:
                        downsample_factor = 4 if random.random() < 0.5 else 2
            else:
                if (iteration < iters - 1 and iteration < 3) and self.use_dense and self.coarse_to_fine_dense:
                    compress_spatial = True
                    downsample_factor = 2**(3 - iteration)


            if compress_spatial:
                concat_feat = torch.cat([coords[:, :, self.N_sparse:, :], coord_depths[:, :, self.N_sparse:, :], track_feat[:, :, self.N_sparse:, :]], dim=-1)
                concat_feat = rearrange(concat_feat, "b t (h w) c -> b t h w c", h=self.dH, w=self.dW)
                concat_feat = concat_feat[:, :, ::downsample_factor, ::downsample_factor]
                concat_feat = rearrange(concat_feat, "b t h w c -> b t (h w) c")

                coords_new = torch.cat([coords[:, :, : self.N_sparse, :], concat_feat[...,:2]], dim=2)
                coord_depths_new = torch.cat([coord_depths[:, :, : self.N_sparse, :], concat_feat[...,2:3]], dim=2)
                track_feat_new = torch.cat([track_feat[:, :, : self.N_sparse, :], concat_feat[...,3:]], dim=2)

                concat_feat = torch.cat([attention_mask[:, :, self.N_sparse:][..., None], track_mask_vis[:, :, self.N_sparse:, :]], dim=-1)
                concat_feat = rearrange(concat_feat, "b t (h w) c -> b t h w c", h=self.dH, w=self.dW)
                concat_feat = concat_feat[:, :, ::downsample_factor, ::downsample_factor]
                concat_feat = rearrange(concat_feat, "b t h w c -> b t (h w) c")
                
                attention_mask_new = torch.cat([attention_mask[:, :, : self.N_sparse], concat_feat[...,0]], dim=2).bool()
                track_mask_vis_new = torch.cat([track_mask_vis[:, :, : self.N_sparse, :], concat_feat[...,1:]], dim=2)

                # NOTE Prepare input to transformer
                if False: # in this way, each element of the support track feat is expected to have the same shape
                    C1, C2, C3 = supp_track_feat[0][:, self.N_sparse:].shape[-1], supp_track_feat[1][:, self.N_sparse:].shape[-1], supp_track_feat[2][:, self.N_sparse:].shape[-1], 
                    supp_track_feat_concat = torch.cat([supp_track_feat[0][:, self.N_sparse:], supp_track_feat[1][:, self.N_sparse:], supp_track_feat[2][:, self.N_sparse:]], dim=-1)
                    supp_track_feat_concat = rearrange(supp_track_feat_concat, "b (h w) r1 r2 c -> b h w r1 r2 c", h=self.dH, w=self.dW)
                    supp_track_feat_concat = supp_track_feat_concat[:, ::downsample_factor, ::downsample_factor]

                    supp_track_feat_concat = rearrange(supp_track_feat_concat, "b h w r1 r2 c -> b (h w) r1 r2 c")

                    supp_track_feat_new = [
                        torch.cat([supp_track_feat[0][:, :self.N_sparse], supp_track_feat_concat[..., :C1]], dim=1),
                        torch.cat([supp_track_feat[1][:, :self.N_sparse], supp_track_feat_concat[..., C1:C1+C2]], dim=1),
                        torch.cat([supp_track_feat[2][:, :self.N_sparse], supp_track_feat_concat[..., C1+C2:]], dim=1)
                    ]
                else:
                    supp_track_feat_new = []
                    for i, tmp_feat in enumerate(supp_track_feat):
                        tmp_feat = tmp_feat[:, self.N_sparse:]
                        tmp_feat = rearrange(tmp_feat, "b (h w) r1 r2 c -> b h w r1 r2 c", h=self.dH, w=self.dW)
                        tmp_feat = tmp_feat[:, ::downsample_factor, ::downsample_factor]

                        tmp_feat = rearrange(tmp_feat, "b h w r1 r2 c -> b (h w) r1 r2 c")
                        supp_track_feat_new.append(
                            torch.cat([supp_track_feat[i][:, :self.N_sparse], tmp_feat], dim=1)
                        )


                delta = self.temporal_compress(
                    fmaps_pyramid,
                    depthmaps,
                    coords_new,
                    coord_depths_new,
                    track_mask_vis_new,
                    track_feat_new,
                    supp_track_feat_new,
                    attention_mask_new,
                    use_efficient_global_attn,
                    iteration=iteration,
                    iters=iters
                )
                

                delta_coords = delta[..., :2]
                ratio_coords_depths = torch.exp(0.1 * torch.clamp(delta[..., 2:3], -50, 50))
                delta_feat = self.track_feat_updater(self.norm(rearrange(delta[..., 3:], "b t n c -> (b t n) c")))
                delta_feat = rearrange(delta_feat, "(b t n) c -> b t n c", b=B, t=S)  ###########################


                if inter_up_mask_dict.get(f"up_mask_{downsample_factor}", None) is None:
                    
                    feat_dense_up = rearrange(track_feat[:, 0, self.N_sparse:, :], "b (h w) c -> b c h w", h=self.dH, w=self.dW)
                    feat_dense_down = feat_dense_up[:, :, ::downsample_factor, ::downsample_factor]

                    inter_up_mask = self.inter_up( 
                        feat_dense_down,
                        feat_dense_up,
                        upsample_factor=downsample_factor,
                    )
                    inter_up_mask_dict[f"up_mask_{downsample_factor}"] = inter_up_mask
                else:
                    inter_up_mask = inter_up_mask_dict[f"up_mask_{downsample_factor}"]


                inter_up_mask = repeat(inter_up_mask, "b k h w -> b t k h w", t=S)

                concat_feat = torch.cat([delta_coords[:, :, self.N_sparse:, :], ratio_coords_depths[:, :, self.N_sparse:, :], delta_feat[:, :, self.N_sparse:, :]], dim=-1)

                concat_feat_up = self.upsample_with_mask2(
                    rearrange(concat_feat, 'b t (h w) c -> (b t) c h w', h=self.dH//downsample_factor, w=self.dW//downsample_factor),  # dense_coords_depths_down * (self.d_far-self.d_near) / self.Dz + self.d_near,
                    rearrange(inter_up_mask, 'b t k h w -> (b t) 1 k h w', h=self.dH, w=self.dW),
                    current_upsample_factor=downsample_factor, current_upsample_kernel_size=2,
                )   

                concat_feat_up = rearrange(concat_feat_up, '(b t) c h w -> b t (h w) c', b=B, t=S)

                delta_coords = torch.cat([delta_coords[:, :, : self.N_sparse, :], concat_feat_up[..., :2]], dim=2)
                ratio_coords_depths = torch.cat([ratio_coords_depths[:, :, : self.N_sparse, :], concat_feat_up[..., 2:3]], dim=2)
                delta_feat = torch.cat([delta_feat[:, :, : self.N_sparse, :], concat_feat_up[..., 3:]], dim=2)

            else:
                delta = self.temporal_compress(
                    fmaps_pyramid,
                    depthmaps,
                    coords,
                    coord_depths,
                    track_mask_vis,
                    track_feat,
                    supp_track_feat,
                    attention_mask,
                    use_efficient_global_attn,
                    iteration=iteration,
                    iters=iters
                )


                delta_coords = delta[..., :2]
                ratio_coords_depths = torch.exp(0.1 * torch.clamp(delta[..., 2:3], -50, 50))
                delta_feat = self.track_feat_updater(self.norm(rearrange(delta[..., 3:], "b t n c -> (b t n) c")))
                delta_feat = rearrange(delta_feat, "(b t n) c -> b t n c", b=B, t=S)  ###########################

            # NOTE Update
            track_feat = track_feat + delta_feat
            coords = coords + delta_coords
            coord_depths = coord_depths * ratio_coords_depths
            ###########################

            coord_preds.append(coords.clone())
            coord_depth_preds.append(coord_depths.clone())

        vis_pred = self.vis_predictor(track_feat).squeeze(-1)  # b s n
        conf_pred = self.conf_predictor(track_feat).squeeze(-1)  # b s n

        return coord_preds, coord_depth_preds, vis_pred, conf_pred, track_feat
    
    def extract_features(self, video: VideoType) -> tuple[Float[Tensor, "*"], ...]:

        B, T = video.shape[:2]
        fmaps, higher_fmaps, lower_fmaps = self.fnet(
            rearrange(video, "b t c h w -> (b t) c h w"), return_intermediate=True
        )

        fmaps = rearrange(fmaps, "(b t) c h w -> b t c h w", b=B, t=T)
        higher_fmaps = rearrange(higher_fmaps, "(b t) c h w -> b t c h w", b=B, t=T)
        lower_fmaps = rearrange(lower_fmaps, "(b t) c h w -> b t c h w", b=B, t=T)
        
        if self.use_dino:
            # get dino feats
            if self.use_dino in ["dinov2_vitl14", "dinov3_vitl16", "dinov2_vitb14", "dinov3_vitb16"]:
                dino_maps = self.dino_net(rearrange(video, "b t c h w -> (b t) c h w"))
            elif self.use_dino == "chrono": # chrono track
                img_mult = self.dino_size
                dino_maps = self.dino_net.forward_dino(video, img_mult=img_mult)
                dino_maps = rearrange(dino_maps, "b t c h w -> (b t) c h w")
            else:
                raise
            if self.merge_dino_method in ["merge_conv", "cat"]:
                ret = [fmaps, higher_fmaps, lower_fmaps]
                for i, m in enumerate(ret):
                    mH, mW = m.shape[3:]
                    dino_maps_ = torch.nn.functional.interpolate(
                        dino_maps,
                        size=(mH, mW),
                        mode='bicubic',
                        align_corners=True,
                    )
                    dino_maps_ = rearrange(dino_maps_, "(b t) c h w -> b t c h w", b=B, t=T)
                    ret_i = torch.cat([ret[i], dino_maps_], dim=2)
                    if self.merge_dino_method == "merge_conv":
                        ret_i = rearrange(ret_i, "b t c h w -> b (t h w) c")
                        
                        if i == 2: # lower_res_fmaps
                            ret_i = self.merge_conv0(ret_i)
                        elif i == 0: 
                            ret_i = self.merge_conv1(ret_i)
                        else: # higher_fmaps
                            ret_i = self.merge_conv2(ret_i)
                        
                        ret_i = rearrange(ret_i, "b (t h w) c-> b t c h w", b=B, t=T, h=mH, w=mW)
                        if False:
                            ret_i = ret_i - ret[i]
                            import matplotlib.pyplot as plt
                            ret_i = ret_i.detach().cpu().numpy().reshape(-1)
                            plt.hist(ret_i, bins=20)
                            plt.savefig(f"{i}_dino_mergeConv_hist.png")

                    ret[i] = ret_i
                fmaps, higher_fmaps, lower_fmaps = ret
            elif self.merge_dino_method == "merge_conv_low_res":
                mH, mW = dino_maps.shape[2:]
                dino_maps_ = rearrange(dino_maps, "(b t) c h w -> b (t h w) c", b=B, t=T)
                dino_maps_ = self.merge_conv(dino_maps_)
                # ret_i = torch.cat([lower_fmaps, dino_maps_], dim=2)
                dino_maps_ = rearrange(dino_maps_, "b (t h w) c -> b t c h w", b=B, t=T, h=mH, w=mW)
                lower_fmaps = lower_fmaps + dino_maps_
                
            elif self.merge_dino_method in ["corr", "track_feat", "corr_track_feat"]:
                # mH, mW = fmaps.shape[3:]
                # dino_maps = torch.nn.functional.interpolate(
                #         dino_maps,
                #         size=(mH, mW),
                #         mode='bicubic',
                #         align_corners=True,
                #     )
                dino_maps = rearrange(dino_maps, "(b t) c h w -> b t c h w", b=B, t=T)
                return fmaps, higher_fmaps, lower_fmaps, dino_maps

        return fmaps, higher_fmaps, lower_fmaps, None

    def prepare_sparse_queries(
        self,
        sparse_queries: Float[Tensor, "b n c"],
        fmaps: VideoType,
        fmaps_pyramid: tuple[VideoType, ...],
        dino_fmaps: None,
    ) -> tuple:

        S = self.window_len
        B, _, _, fH, fW = fmaps.shape
        device = fmaps.device

        self.N_sparse = N_sparse = sparse_queries.shape[1]
        sparse_queried_frames = sparse_queries[:, :, 0].long()

        # NOTE normalize queries
        sparse_queried_coords = sparse_queries[..., 1:4].clone()
        sparse_queried_coords[..., :2] = sparse_queried_coords[..., :2] / self.stride
        # We compute track features # FIXME only get 2d coord
        track_feat, supp_track_feats_pyramid = self.get_track_feat(
            fmaps,
            fmaps_pyramid,
            sparse_queried_frames,
            sparse_queried_coords[..., :2],
            dino_fmaps=dino_fmaps,
            # radius=self.radius_corr
        )
        track_feat = repeat(track_feat, "b 1 n c -> b s n c", s=S)

        coords_init = sparse_queried_coords[..., :2].reshape(B, 1, N_sparse, 2).expand(B, S, N_sparse, 2).float()
        depths_init = sparse_queried_coords[..., 2:3].reshape(B, 1, N_sparse, 1).expand(B, S, N_sparse, 1).float()
        vis_init = torch.ones((B, S, N_sparse, 1), device=device).float() * 10
        conf_init = torch.ones((B, S, N_sparse, 1), device=device).float() * 10

        return (
            coords_init,
            depths_init,
            vis_init,
            conf_init,
            track_feat,
            supp_track_feats_pyramid,
            sparse_queried_frames,
        )

    def prepare_dense_queries(
        self,
        coords_init: Float[Tensor, "b t n c"],
        coord_depths_init: Float[Tensor, "b t n 1"],
        vis_init: Float[Tensor, "b t n"],
        conf_init: Float[Tensor, "b t n"],
        track_feat: Float[Tensor, "b t n c"],
        supp_track_feats_pyramid: tuple[Float[Tensor, "*"], ...],
        fmaps: VideoType,
        fmaps_pyramid: tuple[VideoType, ...],
        depth_init_downsample: Float[Tensor, "b 1 h w"],
        is_train: bool = False,
        x0_y0_dW_dH = None,
        dino_fmaps = None,
    ) -> tuple:
    
        S = self.window_len
        B, _, _, fH, fW = fmaps.shape
        device = fmaps.device

        if is_train:
            if self.upsample_factor == 4:
                # dH, dW = 15, 20
                dH, dW = self.dH_train, self.dW_train
                # print(dH, dW)
                # dH, dW = 24, 32
            else:
                dH, dW = 9, 12
            y0 = np.random.randint(0, fH * self.stride / self.upsample_factor - dH, size=B)
            x0 = np.random.randint(0, fW * self.stride / self.upsample_factor - dW, size=B)
        else:
            if x0_y0_dW_dH is None:
                # dH, dW = self.model_resolution[0] // (self.stride), self.model_resolution[1] // (self.stride)
                dH, dW = self.input_reso[0] // (self.upsample_factor), self.input_reso[1] // (self.upsample_factor)
                x0, y0 = [0] * B, [0] * B
            else:
                x0, y0, dW, dH = x0_y0_dW_dH
                dH, dW = dH // (self.upsample_factor), dW // (self.upsample_factor)
                x0, y0 = [x0 // self.upsample_factor] * B, [y0 // self.upsample_factor] * B

        self.dH, self.dW = dH, dW

        cropped_depths = torch.stack(
            [depth_init_downsample[b, :, y0_ : y0_ + dH, x0_ : x0_ + dW] for (b, y0_, x0_) in zip(range(B), y0, x0)],
            dim=0,
        )
        cropped_depths = rearrange(cropped_depths, "b c h w -> b (h w) c")

        dense_grid_2d = (
            get_grid(dH, dW, normalize=False, device=fmaps.device).reshape(-1, 2).unsqueeze(0).repeat(B, 1, 1)
        )  # B, (H, W) 2
        dense_grid_2d = dense_grid_2d * self.upsample_factor / self.stride 

        for b in range(B):
            dense_grid_2d[b, :, 0] += x0[b]
            dense_grid_2d[b, :, 1] += y0[b]

        dense_coords_init = repeat(dense_grid_2d, "b n c -> b s n c", s=S)
        dense_coord_depths_init = repeat(cropped_depths, "b n c -> b s n c", s=S)
        dense_vis_init = torch.ones((B, S, dH * dW, 1), device=device).float() * 10
        dense_conf_init = torch.ones((B, S, dH * dW, 1), device=device).float() * 10

        dense_track_feat, dense_supp_track_feats_pyramid = self.get_dense_track_feat(
            fmaps,
            fmaps_pyramid,
            dense_grid_2d,
            dino_fmaps=dino_fmaps,
            # radius=self.radius_corr
        )
        dense_track_feat = repeat(dense_track_feat, "b n c -> b s n c", s=S)

        dense_grid_2d_up = (
            get_grid(dH * self.upsample_factor, dW * self.upsample_factor, normalize=False, device=fmaps.device)
            .reshape(-1, 2)
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )
        for b in range(B):
            dense_grid_2d_up[b, :, 0] += x0[b] * self.upsample_factor
            dense_grid_2d_up[b, :, 1] += y0[b] * self.upsample_factor

        self.original_grid_low_reso = rearrange(dense_grid_2d.clone(), "b (h w) c -> b c h w", h=dH, w=dW)
        self.original_grid_high_reso = rearrange(
            dense_grid_2d_up.clone(), "b (h w) c -> b c h w", h=dH * self.upsample_factor, w=dW * self.upsample_factor
        )

        coords_init = smart_cat(
            coords_init, dense_coords_init, dim=2
        )  # torch.cat([coords_init, dense_coords_init], dim=2)
        coord_depths_init = smart_cat(
            coord_depths_init, dense_coord_depths_init, dim=2
        )  # torch.cat([coord_depths_init, dense_coord_depths_init], dim=2)
        vis_init = smart_cat(vis_init, dense_vis_init, dim=2)  # torch.cat([vis_init, dense_vis_init], dim=2)
        conf_init = smart_cat(conf_init, dense_conf_init, dim=2)  # torch.cat([vis_init, dense_vis_init], dim=2)
        track_feat = smart_cat(track_feat, dense_track_feat, dim=2)  # torch.cat([track_feat, dense_track_feat], dim=2)
        
        # fix bug when use sparse=False
        if self.merge_dino_method == "corr" and len(supp_track_feats_pyramid) == 3:
            supp_track_feats_pyramid.append(supp_track_feats_pyramid[-1])

        supp_track_feats_pyramid = [
            smart_cat(sf, dense_sf, dim=1)
            for sf, dense_sf in zip(supp_track_feats_pyramid, dense_supp_track_feats_pyramid)
        ]

        return (coords_init, coord_depths_init, vis_init, conf_init, track_feat, supp_track_feats_pyramid, (x0, y0))

    def forward(
        self,
        video: VideoType,
        videodepth: Float[Tensor, "b t 1 h w"],
        sparse_queries: Float[Tensor, "b n c"] = None,
        depth_init: Float[Tensor, "b 1 h w"] = None,
        iters: int = 4,
        is_train: bool = False,
        use_dense: bool = True,
        use_efficient_global_attn: bool = False,
        accelerator = None,
        x0_y0_dW_dH = None,
        grid_queries = None,
        grid_queries_last = None,
        depth_init_last = None,
        cycle_loss  = False,
        ret_feats = False,
    ) -> tuple[dict | None, dict | None, tuple[dict | None, dict | None] | None]:
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3, H, W]): input videos.
            videodepth (FloatTensor[B, T, 1, H, W]): input videodepths.
            queries (FloatTensor[B, N, 3]): point queries.
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
            is_online (bool, optional): enables online mode. Defaults to False. Before enabling, call model.init_video_online_processing().
        """

        B, T, C, H, W = video.shape
        S = self.window_len
        device = video.device

        self.input_reso = (H, W)

        use_sparse = True if sparse_queries is not None else False
        
        assert use_sparse or use_dense, "At least one of use_sparse and use_dense must be True"
        self.use_dense = use_dense
        self.use_sparse = use_sparse

        assert S >= 2  # A tracker needs at least two frames to track something

        self.ori_T = ori_T = T
        self.Dz = Dz = W // self.stride

        step = S // 2  # How much the sliding window moves at every step
        video = 2 * (video / 255.0) - 1.0

        # Pad the video so that an integer number of sliding windows fit into it
        # TODO: we may drop this requirement because the transformer should not care
        # TODO: pad the features instead of the video
        if is_train:
            pad = 0
        else:
            pad = (S - T % S) % S  # We don't want to pad if T % S == 0

        if pad > 0:
            video = F.pad(video.reshape(B, 1, T, C * H * W), (0, 0, 0, pad), "replicate").reshape(B, -1, C, H, W)
            videodepth = F.pad(videodepth.reshape(B, 1, T, H * W), (0, 0, 0, pad), "replicate").reshape(B, -1, 1, H, W)

        if depth_init is None:
            depth_init = videodepth[:, 0].clone()

        if self.freeze_fnet:
            with torch.no_grad():
                fmaps, higher_fmaps, lower_fmaps, dino_fmaps = self.extract_features(video)
        else:
            fmaps, higher_fmaps, lower_fmaps, dino_fmaps = self.extract_features(video)
        
        n_sparse = sparse_queries.shape[1]
        if grid_queries is not None:
            sparse_queries = torch.cat([sparse_queries, grid_queries], dim=1)
        sparse_predictions, dense_predictions, train_data = self.forward_tracking_head(
            video,
            videodepth,
            sparse_queries,
            depth_init,
            iters,
            is_train,
            use_dense,
            use_efficient_global_attn,
            accelerator,
            x0_y0_dW_dH,
            higher_fmaps,
            fmaps,
            lower_fmaps,
            dino_fmaps,
        )
        
        if self.cycle_loss or cycle_loss:
            # prepare cycle
            # sparse_queries: 1, 320, 4 (t u v d)
            
            sparse_predictions_uv = sparse_predictions["coords"] # 1, 24, 320, 2
            sparse_predictions_d = sparse_predictions["coord_depths"] # 1, 24, 320, 1
            
            sparse_queries_inverse = torch.cat([torch.zeros_like(sparse_predictions_d[:, -1, :n_sparse]), sparse_predictions_uv[:, -1, :n_sparse], sparse_predictions_d[:, -1, :n_sparse]], dim=2)
            
            sparse_queries_inverse = torch.cat([sparse_queries_inverse, grid_queries_last], dim=1)
            # import pdb
            # pdb.set_trace()
            sparse_predictions_inverse, dense_predictions_inverse, train_data_inverse = self.forward_tracking_head(
                video.flip(1),
                videodepth.clip(1),
                sparse_queries_inverse,
                depth_init_last,
                iters,
                is_train,
                False, # use_dense,
                use_efficient_global_attn,
                accelerator,
                x0_y0_dW_dH,
                higher_fmaps.flip(1),
                fmaps.flip(1),
                lower_fmaps.flip(1),
                dino_fmaps.flip(1) if dino_fmaps is not None else None,
            )
            # import pdb
            # pdb.set_trace()
            if True:
                # flip back the prediction
                for k, v in sparse_predictions_inverse.items():
                    sparse_predictions_inverse[k] = v.flip(1)
                
                
                for element in train_data_inverse:
                    if element is None:
                        continue
                    for k, v in element.items():
                        if k == "mask":
                            element[k] = v.flip(1)
                            continue

                        # element[k] is list of window
                        new_v = []
                        for v_window in v:
                            if k in ["coords", "coord_depths"]:
                                new_v.insert(0, v_window.flip(2)) # insert to inverse window seq
                            elif k in ["vis", "conf"]:
                                new_v.insert(0, v_window.flip(1)) # insert to inverse window seq
                            else:
                                raise

                        element[k] = new_v
                    
            return sparse_predictions, dense_predictions, train_data, sparse_predictions_inverse, dense_predictions_inverse, train_data_inverse
        else:
            if ret_feats:
                return sparse_predictions, dense_predictions, train_data, None, None, (None, None), (fmaps, higher_fmaps, lower_fmaps, dino_fmaps)
            else:
                return sparse_predictions, dense_predictions, train_data, None, None, (None, None)
    
    def forward_tracking_head(
        self,
        video: VideoType,
        videodepth: Float[Tensor, "b t 1 h w"],
        sparse_queries: Float[Tensor, "b n c"] = None,
        depth_init: Float[Tensor, "b 1 h w"] = None,
        iters: int = 4,
        is_train: bool = False,
        use_dense: bool = True,
        use_efficient_global_attn: bool = False,
        accelerator = None,
        x0_y0_dW_dH = None,
        higher_fmaps = None,
        fmaps = None,
        lower_fmaps = None,
        dino_fmaps = None,
    ) -> tuple[dict | None, dict | None, tuple[dict | None, dict | None] | None]:
        B, _, C, H, W = video.shape
        S = self.window_len
        device = video.device
        step = S // 2
        use_sparse = self.use_sparse
        T = ori_T = self.ori_T
        Dz = self.Dz

        # print(dino_fmaps.shape)
        fmaps_pyramid = [higher_fmaps, fmaps, lower_fmaps]
        fH, fW = fmaps.shape[-2:]

        videodepth_downsample = F.interpolate(videodepth.reshape(-1, 1, H, W), size=(fH, fW), mode="nearest").reshape(
            B, -1, 1, fH, fW
        )

        depth_init_downsample = F.interpolate(
            depth_init,
            size=(int(fH * self.stride/self.upsample_factor), int(fW * self.stride/self.upsample_factor)),
            mode='nearest'
        )

        self.dH, self.dW = 0, 0
        self.N_sparse = 0

        coords_init, depths_init, vis_init, conf_init, track_feat, supp_track_feats_pyramid = (
            None,
            None,
            None,
            None,
            None,
            [None] * 3,
        )
        if use_sparse:
            (
                coords_init,
                depths_init,
                vis_init,
                conf_init,
                track_feat,
                supp_track_feats_pyramid,
                sparse_queried_frames,
            ) = self.prepare_sparse_queries(sparse_queries, fmaps, fmaps_pyramid, dino_fmaps=dino_fmaps)
            
            # We store our predictions here
            coords_predicted = torch.zeros((B, ori_T, self.N_sparse, 2), device=device)
            if not is_train:
                coords_predicted_lis = [coords_predicted.clone() for _ in range(iters-1)]
            coords_depths_predicted = torch.zeros((B, ori_T, self.N_sparse, 1), device=device)
            vis_predicted = torch.zeros((B, ori_T, self.N_sparse), device=device)
            conf_predicted = torch.zeros((B, ori_T, self.N_sparse), device=device)
            all_coords_predictions, all_coord_depths_predictions, all_vis_predictions, all_conf_predictions = (
                [],
                [],
                [],
                [],
            )

        if use_dense:
            coords_init, depths_init, vis_init, conf_init, track_feat, supp_track_feats_pyramid, (x0, y0) = (
                self.prepare_dense_queries(
                    coords_init,
                    depths_init,
                    vis_init,
                    conf_init,
                    track_feat,
                    supp_track_feats_pyramid,
                    fmaps,
                    fmaps_pyramid,
                    depth_init_downsample,
                    is_train,
                    x0_y0_dW_dH,
                    dino_fmaps=dino_fmaps,
                )
            )

            dense_coords_up_predicted = torch.zeros(
                (B, ori_T, 2, self.dH * self.upsample_factor, self.dW * self.upsample_factor), device=device
            )
            dense_coord_depths_up_predicted = torch.zeros(
                (B, ori_T, 1, self.dH * self.upsample_factor, self.dW * self.upsample_factor), device=device
            )
            dense_vis_up_predicted = torch.zeros(
                (B, ori_T, self.dH * self.upsample_factor, self.dW * self.upsample_factor), device=device
            )
            dense_conf_up_predicted = torch.zeros(
                (B, ori_T, self.dH * self.upsample_factor, self.dW * self.upsample_factor), device=device
            )

            dense_coords_down_predicted = torch.zeros(
                (B, ori_T, 2, self.dH, self.dW), device=device
            )
            dense_coord_depths_down_predicted = torch.zeros(
                (B, ori_T, 1, self.dH, self.dW), device=device
            )
            dense_vis_down_predicted = torch.zeros(
                (B, ori_T, self.dH, self.dW), device=device
            )
            dense_conf_down_predicted = torch.zeros(
                (B, ori_T, self.dH, self.dW), device=device
            )

            (
                all_dense_coords_predictions,
                all_dense_coord_depths_predictions,
                all_dense_vis_predictions,
                all_dense_conf_predictions,
            ) = ([], [], [], [])
            up_mask = None

        if self.merge_dino_method in ["corr", "corr_track_feat"]:
            fmaps_pyramid.append(dino_fmaps)
        #     supp_track_feats_lvl1 = supp_track_feats_pyramid[1]
        #     supp_track_feats_lvl1, supp_track_dinos = supp_track_feats_lvl1[..., :-self.dino_net.C], supp_track_feats_lvl1[..., -self.dino_net.C:]
        #     supp_track_feats_pyramid[1] = supp_track_feats_lvl1

        # above just init feature of len window

        # We process ((num_windows - 1) * step + S) frames in total, so there are
        # (ceil((T - S) / step) + 1) windows
        num_windows = (T - S + step - 1) // step + 1
        # We process only the current video chunk in the online mode
        indices = range(0, step * num_windows, step)
        if len(indices) == 0:
            indices = [0]

        if self.use_dense and self.coarse_to_fine_dense:
            inter_up_mask_dict = dict()
            # inter_up_feat_dense = self.inter_up_proj(track_feat[:, 0, self.N_sparse:]) # B (H W) C
        else:
            inter_up_mask_dict = None

        for ind in indices:
            # We copy over coords and vis for tracks that are queried
            # by the end of the previous window, which is ind + overlap
            if ind > 0:
                overlap = S - step

                copy_over = None

                if use_sparse:
                    copy_over = (sparse_queried_frames < ind + overlap)[:, None, :, None]  # B 1 N 1
                if use_dense:
                    copy_over = smart_cat(
                        copy_over, torch.ones((B, 1, self.dH * self.dW, 1), device=device), dim=2
                    ).bool()

                last_coords = coords[-1][:, -overlap:].clone()
                last_depths = coords_depths[-1][:, -overlap:].clone()
                last_vis = vis[:, -overlap:].clone()[..., None]
                last_conf = conf[:, -overlap:].clone()[..., None]

                coords_prev = torch.nn.functional.pad(
                    last_coords,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 2
                depths_prev = torch.nn.functional.pad(
                    last_depths,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 2
                vis_prev = torch.nn.functional.pad(
                    last_vis,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 1
                conf_prev = torch.nn.functional.pad(
                    last_conf,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 1

                coords_init = torch.where(copy_over.expand_as(coords_init), coords_prev, coords_init)
                depths_init = torch.where(copy_over.expand_as(depths_init), depths_prev, depths_init)
                vis_init = torch.where(copy_over.expand_as(vis_init), vis_prev, vis_init)
                conf_init = torch.where(copy_over.expand_as(conf_init), conf_prev, conf_init)

            # The attention mask is 1 for the spatio-temporal points within a track which is updated in the current window
            attention_mask, track_mask = None, None
            if use_sparse:
                attention_mask = (
                    (sparse_queried_frames < ind + S).reshape(B, 1, self.N_sparse).repeat(1, S, 1)
                )  # B S N

                # The track mask is 1 for the spatio-temporal points that actually need updating: only after begin queried, and not if contained in a previous window
                track_mask = (
                    sparse_queried_frames[:, None, :, None]
                    <= torch.arange(ind, ind + S, device=device)[None, :, None, None]
                ).contiguous()  # B S N 1
                if ind > 0:
                    track_mask[:, :overlap, :, :] = False

            if use_dense:
                track_mask = smart_cat(track_mask, torch.ones((B, S, self.dH * self.dW, 1), device=device), dim=2)
                attention_mask = smart_cat(
                    attention_mask, torch.ones((B, S, self.dH * self.dW), device=device), dim=2
                ).bool()
            
            activate_fst_window = accelerator.process_index % 2 == 0 if accelerator is not None else None
            class blank_contexts:
                def __enter__(self):
                    pass
                def __exit__(self, *args):
                    pass
            if accelerator is None or (activate_fst_window and ind == 0) or ((not activate_fst_window) and ind > 0):
                context_manager = blank_contexts()
            else:
                context_manager = torch.no_grad()
            with context_manager:
            # if True:
                coords, coords_depths, vis, conf, track_feat_updated = self.forward_window(
                    fmaps_pyramid=[f[:, ind : ind + S] for f in fmaps_pyramid],
                    depthmaps=videodepth_downsample[:, ind : ind + S],
                    coords=coords_init,
                    coord_depths=depths_init,
                    vis=vis_init,
                    conf=conf_init,
                    track_feat=attention_mask.unsqueeze(-1).float() * track_feat,
                    supp_track_feat=supp_track_feats_pyramid,
                    track_mask=track_mask,
                    attention_mask=attention_mask,
                    iters=iters,
                    use_efficient_global_attn=use_efficient_global_attn,
                    inter_up_mask_dict=inter_up_mask_dict,
                )
            S_trimmed = min(T - ind, S)  # accounts for last window duration

            if use_sparse:
                coords_predicted[:, ind : ind + S] = coords[-1][:, :S_trimmed, : self.N_sparse] * self.stride
                coords_depths_predicted[:, ind : ind + S] = coords_depths[-1][:, :S_trimmed, : self.N_sparse]
                vis_predicted[:, ind : ind + S] = vis[:, :S_trimmed, : self.N_sparse]
                conf_predicted[:, ind : ind + S] = conf[:, :S_trimmed, : self.N_sparse]

                if not is_train:
                    for iti in range(iters - 1):
                        coords_predicted_lis[iti][:, ind : ind + S] = coords[iti][:, :S_trimmed, : self.N_sparse] * self.stride

                # if is_train:
                #     all_coords_predictions.append(
                #         [(coord[:, :S_trimmed, : self.N_sparse] * self.stride) for coord in coords]
                #     )
                #     all_coord_depths_predictions.append(
                #         [coords_d[:, :S_trimmed, : self.N_sparse] for coords_d in coords_depths]
                #     )
                #     all_vis_predictions.append(torch.sigmoid(vis[:, :S_trimmed, : self.N_sparse]))
                #     all_conf_predictions.append(torch.sigmoid(conf[:, :S_trimmed, : self.N_sparse]))
                if is_train:
                    all_coords_predictions.append(torch.stack([(coord[:, :S_trimmed, :self.N_sparse] * self.stride) for coord in coords], dim=1))
                    all_coord_depths_predictions.append(torch.stack([coords_d[:, :S_trimmed, :self.N_sparse] for coords_d in coords_depths], dim=1))
                    all_vis_predictions.append(torch.sigmoid(vis[:, :S_trimmed, :self.N_sparse]))
                    all_conf_predictions.append(torch.sigmoid(conf[:, :S_trimmed, :self.N_sparse]))

            if use_dense:

                dense_coords_down_predicted[:, ind : ind + S] = rearrange(coords[-1][:, :S_trimmed, self.N_sparse :], "b s (h w) c -> b s c h w", h=self.dH, w=self.dW)
                dense_coord_depths_down_predicted[:, ind : ind + S] = rearrange(coords_depths[-1][:, :S_trimmed, self.N_sparse :], "b s (h w) c -> b s c h w", h=self.dH, w=self.dW)
                dense_vis_down_predicted[:, ind : ind + S] = rearrange(vis[:, :S_trimmed, self.N_sparse :], "b s (h w) -> b s h w", h=self.dH, w=self.dW)
                dense_conf_down_predicted[:, ind : ind + S] = rearrange(conf[:, :S_trimmed, self.N_sparse :], "b s (h w) -> b s h w", h=self.dH, w=self.dW)


                dense_coords_down_list = [
                    rearrange(coord_[:, :, self.N_sparse :], "b s (h w) c -> (b s) c h w", h=self.dH, w=self.dW)
                    for coord_ in coords
                ]
                dense_coords_depths_down_list = [
                    rearrange(coords_depth_[:, :, self.N_sparse :], "b s (h w) c -> (b s) c h w", h=self.dH, w=self.dW)
                    for coords_depth_ in coords_depths
                ]
                dense_vis_down = rearrange(
                    vis[:, :, self.N_sparse :], "b s (h w) -> (b s) 1 h w", h=self.dH, w=self.dW
                )
                dense_conf_down = rearrange(
                    conf[:, :, self.N_sparse :], "b s (h w) -> (b s) 1 h w", h=self.dH, w=self.dW
                )

                if up_mask is None:
                    flow_guidance = (dense_coords_down_list[-1] - self.original_grid_low_reso) / self.Dz
                    flow_guidance = rearrange(flow_guidance, "(b s) c h w -> b s c h w", b=B, s=S)
                    upsample_featmap = rearrange(
                        track_feat_updated[:, 0, self.N_sparse :], "b (h w) c -> b c h w", h=self.dH, w=self.dW
                    )

                    # NOTE upsample for prev iterations
                    if self.training:
                        up_mask_prev = self.upsample_transformer_prev(
                            feat_map=upsample_featmap,
                            flow_map=flow_guidance,
                        )
                        up_mask_prev = repeat(up_mask_prev, "b k h w -> b s k h w", s=S)
                        up_mask_prev = rearrange(up_mask_prev, "b s c h w -> (b s) 1 c h w")

                    up_mask = self.upsample_transformer(
                        feat_map=upsample_featmap,
                        flow_map=flow_guidance,
                    )
                    up_mask = repeat(up_mask, "b k h w -> b s k h w", s=S)
                    up_mask = rearrange(up_mask, "b s c h w -> (b s) 1 c h w")

                dense_coords_up_list, dense_coords_depths_up_list = [], []
                for pred_id_ in range(len(dense_coords_down_list)):
                    
                    if not self.training:
                        up_mask_current = up_mask
                    else:
                        if pred_id_ == len(dense_coords_down_list) - 1:
                            up_mask_current = up_mask
                        else:
                            up_mask_current = up_mask_prev

                    dense_coords_down = dense_coords_down_list[pred_id_]
                    dense_coords_depths_down = dense_coords_depths_down_list[pred_id_]

                    dense_coords_up = self.original_grid_high_reso + self.upsample_with_mask(
                        (dense_coords_down - self.original_grid_low_reso) * self.stride,
                        up_mask_current,
                    )
                    dense_coords_up = rearrange(dense_coords_up, "(b s) c h w -> b s c h w", b=B, s=S)
                    dense_coords_up_list.append(dense_coords_up)

                    dense_coords_depths_up = self.upsample_with_mask(
                        dense_coords_depths_down,  # dense_coords_depths_down * (self.d_far-self.d_near) / self.Dz + self.d_near,
                        up_mask_current,
                    )

                    dense_coords_depths_up = rearrange(dense_coords_depths_up, "(b s) c h w -> b s c h w", b=B, s=S)

                    dense_coords_depths_up_list.append(dense_coords_depths_up)

                dense_vis_up = self.upsample_with_mask(dense_vis_down, up_mask)
                dense_vis_up = rearrange(dense_vis_up, "(b s) 1 h w -> b s h w", b=B, s=S)

                dense_conf_up = self.upsample_with_mask(dense_conf_down, up_mask)
                dense_conf_up = rearrange(dense_conf_up, "(b s) 1 h w -> b s h w", b=B, s=S)

                dense_coords_up_predicted[:, ind : ind + S] = dense_coords_up_list[-1][
                    :, :S_trimmed
                ]  # dense_coords_out[:, -1]
                dense_coord_depths_up_predicted[:, ind : ind + S] = dense_coords_depths_up_list[-1][
                    :, :S_trimmed
                ]  #  dense_coord_depths_out[:, -1]
                dense_vis_up_predicted[:, ind : ind + S] = dense_vis_up[:, :S_trimmed]
                dense_conf_up_predicted[:, ind : ind + S] = dense_conf_up[:, :S_trimmed]


                if is_train:
                    all_dense_coords_predictions.append(torch.stack([dense_coord[:, :S_trimmed] for dense_coord in dense_coords_up_list], dim=1)) # B I T C H W
                    all_dense_coord_depths_predictions.append(torch.stack([dense_coord_depths[:, :S_trimmed] for dense_coord_depths in dense_coords_depths_up_list], dim=1)) # B I T C H W
                    all_dense_vis_predictions.append(torch.sigmoid(dense_vis_up[:, :S_trimmed]))
                    all_dense_conf_predictions.append(torch.sigmoid(dense_conf_up[:, :S_trimmed]))

        sparse_predictions, dense_predictions = None, None

        if use_sparse:
            vis_predicted = torch.sigmoid(vis_predicted)
            conf_predicted = torch.sigmoid(conf_predicted)
            sparse_predictions = dict(
                coords=coords_predicted,
                coord_depths=coords_depths_predicted,
                vis=vis_predicted,
                conf=conf_predicted,
            )

        if use_dense:
            dense_vis_up_predicted = torch.sigmoid(dense_vis_up_predicted)
            dense_conf_up_predicted = torch.sigmoid(dense_conf_up_predicted)


            dense_vis_down_predicted = torch.sigmoid(dense_vis_down_predicted)
            dense_conf_down_predicted = torch.sigmoid(dense_conf_down_predicted)

            dense_predictions = dict(
                coords=dense_coords_up_predicted,
                coord_depths=dense_coord_depths_up_predicted,
                vis=dense_vis_up_predicted,
                conf=dense_conf_up_predicted,

                coords_down=dense_coords_down_predicted,
                coord_depths_down=dense_coord_depths_down_predicted,
                vis_down=dense_vis_down_predicted,
                conf_down=dense_conf_down_predicted,
            )

        if not is_train:
            return sparse_predictions, dense_predictions, coords_predicted_lis

        sparse_train_data_dict, dense_train_data_dict = None, None
        if use_sparse:
            mask = sparse_queried_frames[:, None] <= torch.arange(0, T, device=device)[None, :, None]
            sparse_train_data_dict = dict(
                coords=all_coords_predictions,
                coord_depths=all_coord_depths_predictions,
                vis=all_vis_predictions,
                conf=all_conf_predictions,
                mask=mask,
            )

        if use_dense:
            dense_train_data_dict = dict(
                coords=all_dense_coords_predictions,
                coord_depths=all_dense_coord_depths_predictions,
                vis=all_dense_vis_predictions,
                conf=all_dense_conf_predictions,
                x0y0=(x0, y0),
            )

        train_data = (sparse_train_data_dict, dense_train_data_dict)

        return sparse_predictions, dense_predictions, train_data
