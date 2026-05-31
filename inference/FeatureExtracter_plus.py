from __future__ import annotations

import random
import GeodisTK
from scipy import ndimage
import os.path as osp
import os
import torch
from torch import nn
import numpy as np
from numpy import inf
from time import time
from tqdm import tqdm
from monai.transforms import *
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
from utils.ops import *
from utils import view_ops
from utils import view_transforms
from utils.losses import *
from time import time
from utils.util import update_log, get_time, mkdir
import torch.nn.functional as F
from typing import Tuple
import pdb

from monai.losses import DiceLoss
from monai.metrics import compute_dice
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast
from monai.networks.utils import one_hot
from loss_functions.deep_supervision import MultipleOutputLoss
from utils.util import update_log, get_lr, get_time, poly_lr, create_sitkImage

from monai.utils import (
    LazyAttr,
    Method,
    PytorchPadMode,
    TraceKeys,
    TransformBackends,
    convert_data_type,
    convert_to_tensor,
    deprecated_arg_default,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    pytorch_after,
)
from collections.abc import Callable, Sequence


def divisible_padding(tensor, k):
    shape = tensor.shape
    dims_to_pad = shape[-3:]

    pad_D = (k - dims_to_pad[0] % k) % k
    pad_H = (k - dims_to_pad[1] % k) % k
    pad_W = (k - dims_to_pad[2] % k) % k

    padding = (0, pad_W, 0, pad_H, 0, pad_D)

    padded_tensor = torch.nn.functional.pad(tensor, padding)

    return padded_tensor


class MyPad(Pad):
    backend = SpatialPad.backend

    def __init__(
            self,
            k: Sequence[int] | int,
            mode: str = PytorchPadMode.CONSTANT,
            method: str = Method.SYMMETRIC,
            lazy: bool = False,
            **kwargs,
    ) -> None:

        self.k = k
        self.method: Method = Method(method)
        super().__init__(mode=mode, lazy=lazy, **kwargs)

    def compute_pad_width(self, spatial_shape: Sequence[int]) -> tuple[tuple[int, int]]:
        new_size = compute_divisible_spatial_size(spatial_shape=spatial_shape, k=self.k)
        spatial_pad = SpatialPad(spatial_size=new_size, method=self.method)
        return spatial_pad.compute_pad_width(spatial_shape)


class AddChannel(Transform):
    """
    Adds a 1-length channel dimension to the input image.

    Most of the image transformations in ``monai.transforms``
    assumes the input image is in the channel-first format, which has the shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]).

    This transform could be used, for example, to convert a (spatial_dim_1[, spatial_dim_2, ...])
    spatial image into the channel-first format so that the
    multidimensional image array can be correctly interpreted by the other
    transforms.
    """

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        return img[None]


class AddChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    backend = AddChannel.backend

    def __init__(self, keys, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.adder = AddChannel()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adder(d[key])
        return d


class FeatureExtracter(object):
    def __init__(self, data, model, opt):
        self.opt = opt
        self.infer_log = osp.join(self.opt.expr_dir, 'recon.log')
        self.test_ds = data.get_data()
        self.model = model.cuda() if torch.cuda.is_available() else model

        if self.opt.multi_gpu and torch.cuda.device_count() > 1:
            print(f"multiple GPUs are used: {torch.cuda.device_count()}")
            self.model = torch.nn.DataParallel(self.model).cuda().module
            
        self.result_dir = osp.join(self.opt.expr_dir, f'results_epoch_{self.opt.epoch}')
        self.activation = {}

        if self.opt.save_output:
            mkdir(self.result_dir)

        if self.opt.load_ckpt:
            ckpt_path = self.opt.load_ckpt
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt['state_dict'])
            update_log(f"model and optimizer are initialized from {ckpt_path}", self.infer_log)

        # data_path = osp.join(self.opt.dataroot, f'{self.opt.organ[0]}','feats', f'{opt.mode}_data.tsv')
        data_path = osp.join(self.opt.dataroot, f'{self.opt.organ[0]}','feats', 'Ours_features.tsv')
        metadata_path = osp.join(self.opt.dataroot, f'{self.opt.organ[0]}','feats', f'{opt.mode}_metadata.tsv')
        # score_path = osp.join(self.opt.dataroot, f'{self.opt.organ[0]}','feats', f'{opt.mode}_score.tsv')
        score_path = osp.join(self.opt.dataroot, f'{self.opt.organ[0]}','feats', 'Ours_scores.tsv')
        self.data_npz_path = osp.join(self.opt.dataroot, f'{self.opt.organ[0]}','feats', f'Ours.npz')

        self.f_data = open(data_path, 'w')
        self.f_metadata = open(metadata_path, 'w')
        self.f_score = open(score_path, 'w')


    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook


    def extract(self, mode='global'):
        feats, name_list = [], []
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(self.test_ds)) as pbar:
                for i, test_data in enumerate(self.test_ds):
                    img, target, sub = test_data["data_image"].cuda(), test_data["data_mask"].cuda(), test_data["subject"]

                    if mode == 'global':
                        img = SqueezeDim(dim=0)(img)
                        img = divisible_padding(img, 16)
                        # img = SpatialPad(self.opt.crop_size)
                        img = AddChannel()(img)
                    
                    elif mode == 'local':
                        img = Compose([SqueezeDim(dim=0), BorderPad(spatial_border=16), AddChannel()])(img)
                        target = Compose([SqueezeDim(dim=0), BorderPad(spatial_border=16), AddChannel()])(target)
                        cropper = CropForeground(margin=5, k_divisible=16, return_coords=True)
                        _, coords_1, coords_2 = cropper(target[0, ...])
                        img = img[:, :, coords_1[0]:coords_2[0], coords_1[1]:coords_2[1], coords_1[2]:coords_2[2]]

                    pid = osp.splitext(osp.basename(sub[0]))[0]
                    start = time()      
                    
                    x0, rot0 = view_ops.rot_rand_0(img)
                    x1, rot1 = view_ops.rot_rand_1(img)
                    x2, rot2 = view_ops.rot_rand_2(img)
                    
                    window_sizes = tuple(self.opt.window_size for _ in range(3))
                    input_sizes = self.opt.crop_size
                    
                    print(x0.shape)
                    print(input_sizes)
                    
                    x0_masked_view0, mask0 = mask_rand_patch(window_sizes, input_sizes, self.opt.mask_ratio, x0)
                    x1_masked_view0, mask1 = mask_rand_patch(window_sizes, input_sizes, self.opt.mask_ratio, x1)
                    x2_masked_view0, mask2 = mask_rand_patch(window_sizes, input_sizes, self.opt.mask_ratio, x2)
                    
                    permutations_candidates = set(
                            view_transforms.permutation_transforms.keys()) - {0}
                    permutations = [
                            random.choice(list(permutations_candidates)) for _ in range(6)
                    ]
                    
                    x0_masked_view1, x0_masked_view2, x1_masked_view1, x1_masked_view2, x2_masked_view1, x2_masked_view2 = [
                        view_transforms.permutation_inverse_transforms[vn](val)
                        for vn, val in zip(permutations, [x0_masked_view0, x0_masked_view0, x1_masked_view0, x1_masked_view0, x2_masked_view0, x2_masked_view0])
                    ]
                    
                    with autocast(enabled=False):
                        rot0_p, contrastive0_p, rec_x0_view0 = self.model(x0_masked_view0)
                        rot0_p, contrastive0_p, rec_x0_view1 = self.model(x0_masked_view1)
                        rot0_p, contrastive0_p, rec_x0_view2 = self.model(x0_masked_view2)
                        
                        rot0_p, contrastive0_p, rec_x1_view0 = self.model(x1_masked_view0)
                        rot0_p, contrastive0_p, rec_x1_view1 = self.model(x1_masked_view1)
                        rot0_p, contrastive0_p, rec_x1_view2 = self.model(x1_masked_view2)
                        
                        rot0_p, contrastive0_p, rec_x2_view0 = self.model(x2_masked_view0)
                        rot0_p, contrastive0_p, rec_x2_view1 = self.model(x2_masked_view1)
                        rot0_p, contrastive0_p, rec_x2_view2 = self.model(x2_masked_view2)
                        
                                     
                        rec_x0_view1, rec_x0_view2, rec_x1_view1, rec_x1_view2, rec_x2_view1, rec_x2_view2 = [
                            view_transforms.permutation_inverse_transforms[vn](val)
                            for vn, val in zip(permutations, [rec_x0_view1, rec_x0_view2, rec_x1_view1, rec_x1_view2, rec_x2_view1, rec_x2_view2])
                        ]
                        
                        def _align_rot(x, src_rot, dst_rot):
                                x = ensure_five_dims(x)
                                return view_transforms.rotation_transforms[dst_rot](
                                    view_transforms.rotation_inverse_transforms[src_rot]
                                    (x)).contiguous()
                                
                        def ensure_five_dims(tensor):
                                """
                                Ensure the input MetaTensor has five dimensions. If it has four dimensions,
                                add an additional dimension at the beginning.

                                Args:
                                    tensor (MetaTensor): The input MetaTensor to be checked.

                                Returns:
                                    MetaTensor: The MetaTensor with five dimensions.
                                """
                                if tensor.dim() == 4:
                                    tensor = tensor.unsqueeze(0)  # Add batch dimension
                                return tensor
                        

                        rec_x1_view0 = torch.stack([
                            _align_rot(val, src_rot.item(), dst_rot.item())
                            for val, src_rot, dst_rot in zip(rec_x1_view0, rot1, rot0)
                        ]).squeeze(dim=0)
                        
                        rec_x1_view1 = torch.stack([
                            _align_rot(val, src_rot.item(), dst_rot.item())
                            for val, src_rot, dst_rot in zip(rec_x1_view1, rot1, rot0)
                        ]).squeeze(dim=0)
                        
                        rec_x1_view2 = torch.stack([
                            _align_rot(val, src_rot.item(), dst_rot.item())
                            for val, src_rot, dst_rot in zip(rec_x1_view2, rot1, rot0)
                        ]).squeeze(dim=0)
                        
                        rec_x2_view0 = torch.stack([
                            _align_rot(val, src_rot.item(), dst_rot.item())
                            for val, src_rot, dst_rot in zip(rec_x2_view0, rot1, rot0)
                        ]).squeeze(dim=0)
                        
                        rec_x2_view1 = torch.stack([
                            _align_rot(val, src_rot.item(), dst_rot.item())
                            for val, src_rot, dst_rot in zip(rec_x2_view1, rot1, rot0)
                        ]).squeeze(dim=0)
                        
                        rec_x2_view2 = torch.stack([
                            _align_rot(val, src_rot.item(), dst_rot.item())
                            for val, src_rot, dst_rot in zip(rec_x2_view2, rot1, rot0)
                        ]).squeeze(dim=0)
                        
                        tensors = torch.stack([rec_x0_view0, rec_x0_view1, rec_x0_view2, rec_x1_view0, rec_x1_view1, rec_x1_view2,rec_x2_view0, rec_x2_view1, rec_x2_view2], dim=0)
                        # mean_tensor = torch.mean(tensors, dim=0)
                        variance_tensor = torch.var(tensors, dim=0)
                        mean_variance = torch.mean(variance_tensor).cpu().item()                        
  

                    uncertainty_score_list = []
                    uncertainty_score = mean_variance
                    print(uncertainty_score)
                    
                    uncertainty_score_list.append(uncertainty_score)
                    
                    # extract the bottleneck feature
                    # self.model.model[1].submodule[1].submodule[1].submodule[1].submodule.register_forward_hook(self.get_activation('residual'))
                    # pred = self.model(img)
                    # feat = self.activation['residual']  # shape: [1, 128, 8, 8, 8]
                    feat = self.model.swinViT(img.contiguous())[4]

                    feat = F.adaptive_avg_pool3d(feat, 1).squeeze(4).squeeze(3).squeeze(2)

                    pool = nn.AdaptiveAvgPool1d(256)
                    feat = pool(feat).squeeze(0)        
                    print(feat.shape)
                    feat = feat.detach().cpu().numpy() # reformat for plotting
                    
                    self.f_data.write(f'{feat.tolist()}\n'.replace('[', '').replace(']', '').replace(', ', '    '))
                    self.f_score.write(f'{uncertainty_score}\n')
                    self.f_metadata.write(f'{pid}\n')
                    feats.append(feat)
                    name_list.append(pid)
                    pbar.update(1)

        np.savez(
            self.data_npz_path, 
            feats=feats, 
            name_list=name_list)


class FeatureExtracter_slicing_window(object):
    def __init__(self, data, model, opt):
        self.opt = opt
        self.infer_log = osp.join(self.opt.expr_dir, 'recon.log')
        self.test_ds = data.get_data()
        self.model = model.cuda() if torch.cuda.is_available() else model

        if self.opt.multi_gpu and torch.cuda.device_count() > 1:
            print(f"multiple GPUs are used: {torch.cuda.device_count()}")
            self.model = torch.nn.DataParallel(self.model).cuda().module
            
        self.result_dir = osp.join(self.opt.expr_dir, f'results_epoch_{self.opt.epoch}')
        self.activation = {}

        if self.opt.save_output:
            os.makedirs(self.result_dir, exist_ok=True)

        if self.opt.load_ckpt:
            ckpt_path = self.opt.load_ckpt
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt['state_dict'])
            update_log(f"model and optimizer are initialized from {ckpt_path}", self.infer_log)

        data_path = osp.join(self.opt.dataroot, f'{self.opt.organ[0]}','feats', f'{opt.mode}_data.tsv')
        metadata_path = osp.join(self.opt.dataroot, f'{self.opt.organ[0]}','feats', f'{opt.mode}_metadata.tsv')
        score_path = osp.join(self.opt.dataroot, f'{self.opt.organ[0]}','feats', f'{opt.mode}_score.tsv')

        self.f_data = open(data_path, 'w')
        self.f_metadata = open(metadata_path, 'w')
        self.f_score = open(score_path, 'w')


    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def extract(self, mode='global'):
        feats, name_list = [], []
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(self.test_ds)) as pbar:
                for i, test_data in enumerate(self.test_ds):
                    img, target, sub = test_data["data_image"].cuda(), test_data["data_mask"].cuda(), test_data["subject"]

                    if mode == 'global':
                        img = SqueezeDim(dim=0)(img)
                        img = divisible_padding(img, 16)
                        img = AddChannel()(img)
                    
                    elif mode == 'local':
                        img = Compose([SqueezeDim(dim=0), BorderPad(spatial_border=16), AddChannel()])(img)
                        target = Compose([SqueezeDim(dim=0), BorderPad(spatial_border=16), AddChannel()])(target)
                        cropper = CropForeground(margin=5, k_divisible=16, return_coords=True)
                        _, coords_1, coords_2 = cropper(target[0, ...])
                        img = img[:, :, coords_1[0]:coords_2[0], coords_1[1]:coords_2[1], coords_1[2]:coords_2[2]]

                    pid = osp.splitext(osp.basename(sub[0]))[0]
                    start = time()      
                    
                    # Sliding-window processing
                    uncertainty_score = self.sliding_window_processing(img, window_size=(64, 64, 64), stride=(64, 64, 64))
                    
                    self.f_score.write(f'{uncertainty_score}\n')
                    pbar.update(1)
    
    def sliding_window_processing(self, img, window_size, stride):
        # Compute the padding size
        padding = [(w - s % w) % w for s, w in zip(img.shape[2:], window_size)]
        pad = [(p // 2, p - p // 2) for p in padding]
        pad = [item for sublist in pad for item in sublist]  # Flatten list
        img = torch.nn.functional.pad(img, pad, mode='constant', value=0)
        
        # Buffers accumulating the variance and the visit count per voxel
        variance_sum = torch.zeros_like(img)
        count = torch.zeros_like(img)
        
        # Iterate over the sliding-window start/end positions
        for i in range(0, img.shape[2] - window_size[0] + 1, stride[0]):
            for j in range(0, img.shape[3] - window_size[1] + 1, stride[1]):
                for k in range(0, img.shape[4] - window_size[2] + 1, stride[2]):
                    window = img[:, :, i:i+window_size[0], j:j+window_size[1], k:k+window_size[2]]
                    window_variance = self.calculate_variance(window)
                    
                    variance_sum[:, :, i:i+window_size[0], j:j+window_size[1], k:k+window_size[2]] += window_variance
                    count[:, :, i:i+window_size[0], j:j+window_size[1], k:k+window_size[2]] += 1
        
        # Average variance per voxel
        mean_variance = variance_sum / count
        mean_variance = torch.mean(mean_variance).cpu().item()
        return mean_variance

    def calculate_variance(self, img):
        x0, rot0 = view_ops.rot_rand_0(img)
        x1, rot1 = view_ops.rot_rand_1(img)
        x2, rot2 = view_ops.rot_rand_2(img)
        
        window_sizes = tuple(self.opt.window_size for _ in range(3))
        input_sizes = self.opt.crop_size
        
        x0_masked_view0, mask0 = mask_rand_patch(window_sizes, input_sizes, self.opt.mask_ratio, x0)
        x1_masked_view0, mask1 = mask_rand_patch(window_sizes, input_sizes, self.opt.mask_ratio, x1)
        x2_masked_view0, mask2 = mask_rand_patch(window_sizes, input_sizes, self.opt.mask_ratio, x2)
        
        permutations_candidates = set(
                view_transforms.permutation_transforms.keys()) - {0}
        permutations = [
                random.choice(list(permutations_candidates)) for _ in range(6)
        ]
        
        x0_masked_view1, x0_masked_view2, x1_masked_view1, x1_masked_view2, x2_masked_view1, x2_masked_view2 = [
            view_transforms.permutation_inverse_transforms[vn](val)
            for vn, val in zip(permutations, [x0_masked_view0, x0_masked_view0, x1_masked_view0, x1_masked_view0, x2_masked_view0, x2_masked_view0])
        ]

        with autocast(enabled=False):
            rec_x0_view0 = self.model(x0_masked_view0)[2]
            rec_x0_view1 = self.model(x0_masked_view1)[2]
            rec_x0_view2 = self.model(x0_masked_view2)[2]
            
            rec_x1_view0 = self.model(x1_masked_view0)[2]
            rec_x1_view1 = self.model(x1_masked_view1)[2]
            rec_x1_view2 = self.model(x1_masked_view2)[2]
            
            rec_x2_view0 = self.model(x2_masked_view0)[2]
            rec_x2_view1 = self.model(x2_masked_view1)[2]
            rec_x2_view2 = self.model(x2_masked_view2)[2]
            
            tensors = torch.stack([rec_x0_view0, rec_x0_view1, rec_x0_view2, rec_x1_view0, rec_x1_view1, rec_x1_view2,rec_x2_view0, rec_x2_view1, rec_x2_view2], dim=0)
            variance_tensor = torch.var(tensors, dim=0)
            return variance_tensor