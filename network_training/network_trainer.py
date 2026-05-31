#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 07/20/2022


import random
import GeodisTK
import os.path as osp
import torch
import numpy as np
from scipy import ndimage
from utils.ops import *
from utils import view_ops
from utils import view_transforms
from utils.losses import *
from time import time
from monai.losses import DiceLoss
from monai.metrics import compute_dice
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast
from monai.networks.utils import one_hot
from loss_functions.deep_supervision import MultipleOutputLoss
from utils.util import update_log, get_lr, get_time, poly_lr, create_sitkImage


#------------------------------
#         basic trainer
#------------------------------
# Adam optimizer
# No learning rate scheduler
# Loss: Dice loss
# w/ or w/o gradient clipping


class NetworkTrainer(object):

    def __init__(self, data, model, opt):
        self.opt = opt
        self.train_ds, self.valid_ds = data.get_data()
        self.model = model.cuda() 

        if self.opt.multi_gpu and torch.cuda.device_count() > 1:
            print(f"multiple GPUs are used: {torch.cuda.device_count()}")
            self.model = torch.nn.DataParallel(self.model).cuda().module

        self.optim = torch.optim.Adam(
            params=self.model.parameters(), 
            lr=self.opt.init_lr, 
            weight_decay=self.opt.weight_decay)

        self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True).cuda()
        self.include_background = False

        self.run_log = osp.join(self.opt.expr_dir, 'run.log')
        self.val_log = osp.join(self.opt.expr_dir, 'valid_dsc.log')
        # self.max_epoch = self.opt.max_iterations // len(self.train_ds)
        self.max_epoch = self.opt.max_epoch
        self.val_interval = self.opt.val_iterations // len(self.train_ds)
        self.skip_val_epoch = self.opt.skip_val_iterations // len(self.train_ds)
        self.iter_num, self.epoch, self.best_metric, self.best_metric_epoch, self.epochs_no_improve = 0, 1, -1, -1, 0

        if self.opt.load_ckpt:
            self.load_ckpt()
            
    def train(self):
        # if self.epoch % 100 == 0:
        update_log(f"\n{get_time():%Y-%m-%d %H:%M:%S}", self.run_log) 
        update_log(f'epoch:  {self.epoch}', self.run_log)

        self.train_loss = 0
        self.set_training_mode()
        
        for data in self.train_ds:
            self.iter_num += 1

            if self.opt.debug:
                if self.iter_num > 1:
                    break

            self.fit(data)
            self.lr_scheduler()

            if self.opt.display_per_iter:
                update_log((f'[Train]: epoch={self.epoch}, '
                    f'iteration={self.iter_num}/{self.opt.max_iterations}, '
                    f'loss={self.loss.item():.4f}'), self.run_log)

        # if self.iter_num % 100 == 0:
        update_log((f'{get_time():%Y-%m-%d %H:%M:%S}: '
            f'train loss: {(self.train_loss/len(self.train_ds)):.4f}'), self.run_log)

    def valid(self):
        torch.cuda.empty_cache()
        self.set_evaluation_mode()
        self.val_HD95 = []
        self.val_ASD = []
        self.val_dice = []

        with torch.no_grad():
            for i, data in enumerate(self.valid_ds):
                start = time()
                image, target = data["data_image"], data["data_mask"]
                image, target = image.cuda(), target.cuda()
                
                print(image.shape)
                print(target.shape)
                
                target = self.post_label(target)
                print(target.shape)
                
                sub = data["subject"]
                sub = osp.splitext(osp.basename(sub[0]))[0]
                
                pred = self.predict(image)
                pred = self.post_pred(pred)
                print(pred.shape)


                dice = self.get_dice(pred, target, self.include_background)
                
                # print(pred.shape)
                # print(target.shape)
                # ASD, HD95 = self.compute_metrics_for_batch(pred, target, HD95=False)
                
                self.val_dice.append(dice)
                # self.val_ASD.append(ASD)
                # self.val_HD95.append(HD95)

                if self.opt.display_per_iter:

                    update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                        f"epoch={self.epoch}, id={i+1}/{len(self.valid_ds.dataset)}, "
                        f"subject={sub}, time={time()-start:.4f}, "
                        f"dsc={list(map('{:.4f}'.format, dice))}"), self.run_log)
                    
                    # update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                    #  f"epoch={self.epoch}, id={i+1}/{len(self.valid_ds.dataset)}, "
                    #  f"subject={sub}, time={time()-start:.4f}, "
                    #  f"HD95={HD95.tolist()}"), self.run_log)
                    
                    # update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                    #  f"epoch={self.epoch}, id={i+1}/{len(self.valid_ds.dataset)}, "
                    #  f"subject={sub}, time={time()-start:.4f}, "
                    #  f"ASD={ASD.tolist()}"), self.run_log)

                    
            self.val_dice = np.nanmean(self.val_dice, axis=0)  # report dice without nans
            # self.val_ASD = np.nanmean(self.val_ASD, axis=0)
            # self.val_HD95 = np.nanmean(self.val_HD95, axis=0)
            
            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                "validation foreground dice: "
                f"{list(map('{:.4f}'.format, self.val_dice))}"), self.run_log)
            
            # update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
            #     "validation foreground HD95: "
            #     f"{list(map('{:.4f}'.format, self.val_HD95))}"), self.run_log)
            
            # update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
            #    "validation foreground ASD: "
            #    f"{list(map('{:.4f}'.format, self.val_ASD))}"), self.run_log)

            self.val_dice = np.nanmean(self.val_dice)
            # self.val_ASD = np.nanmean(self.val_ASD)
            # self.val_HD95 = np.nanmean(self.val_HD95)
            
            # mean_dice = np.nanmean(self.val_dice)
            # mean_asd = np.nanmean([a.mean() for a in self.val_ASD])
            # mean_hd95 = np.nanmean([h.mean() for h in self.val_HD95])
            update_log(f"{self.val_dice}", self.val_log, verbose=False)
            
            self.save_ckpt()
            
            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: current mean dice:"
                f" {self.val_dice:.4f}, best mean dice: {self.best_metric:.4f}"
                f" at epoch {self.best_metric_epoch}"), self.run_log)

    def set_training_mode(self):
        self.model.train()

    def set_evaluation_mode(self):
        self.model.eval()

    def fit(self, data):
        # Swin-UNETR
        if self.opt.nid == 13:
            loss_function = Loss(self.opt.batch_size, self.opt)

            image, target = data['data_image'], data['data_mask']
            image, target = image.cuda(), target.cuda()
            target = self.prep_label(target)

            x1, rot1 = rot_rand(self.opt, image)
            x2, rot2 = rot_rand(self.opt, image)
            x1_augment = aug_rand(self.opt, x1)
            x2_augment = aug_rand(self.opt, x2)
            x1_augment = x1_augment
            x2_augment = x2_augment
            with autocast(enabled=False):
                rot1_p, contrastive1_p, rec_x1 = self.model(x1_augment)
                rot2_p, contrastive2_p, rec_x2 = self.model(x2_augment)
                rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                rots = torch.cat([rot1, rot2], dim=0)
                imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                imgs = torch.cat([x1, x2], dim=0)
                self.loss, _ = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)
        
        # SwinMM
        elif self.opt.nid == 14:
            mutual_loss_function = MutualLoss(self.opt)
            loss_function = Loss_V2(self.opt.batch_size, self.opt)

            image, target = data['data_image'], data['data_mask']
            image, target = image.cuda(), target.cuda()

            x1, rot1 = view_ops.rot_rand(image)
            x2, rot2 = view_ops.rot_rand(image)

            window_sizes = tuple(self.opt.window_size for _ in range(3))
            input_sizes = self.opt.crop_size

            x1_masked, mask1 = mask_rand_patch(window_sizes, input_sizes, self.opt.mask_ratio, x1)
            x2_masked, mask2 = mask_rand_patch(window_sizes, input_sizes, self.opt.mask_ratio, x2)

            permutations_candidates = set(
                view_transforms.permutation_transforms.keys()) - {0}
            permutations = [
                random.choice(list(permutations_candidates)) for _ in range(2)
            ]
             
            x1_masked_permuted, x2_masked_permuted = [
                view_transforms.permutation_transforms[vn](val)
                for vn, val in zip(permutations, [x1_masked, x2_masked])
            ]

            with autocast(enabled=False):
                rot1_p, contrastive1_p, rec_x1 = self.model(x1_masked)
                rot2_p, contrastive2_p, rec_x2 = self.model(x2_masked)
                _, contrastive3_p, rec_x3 = self.model(x1_masked_permuted)
                _, contrastive4_p, rec_x4 = self.model(x2_masked_permuted)

                # masked voxels: [2, H, W, D]
                mask = torch.stack([mask1, mask2], dim=0)
                rec_x3, rec_x4 = [
                    view_transforms.permutation_inverse_transforms[vn](val)
                    for vn, val in zip(permutations, [rec_x3, rec_x4])
                ]

                rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                rots = torch.cat([rot1, rot2], dim=0)
                # [B, 2, H, W, D]
                imgs_recon = torch.cat([rec_x1, rec_x2], dim=1)
                imgs = torch.cat([x1, x2], dim=1)

                loss1, losses_tasks1 = loss_function(rot_p, rots,
                                                     contrastive1_p,
                                                     contrastive2_p, imgs_recon,
                                                     imgs, mask)

                mutual_loss1 = mutual_loss_function(rec_x3, rec_x1, mask1)

                imgs_recon = torch.cat([rec_x3, rec_x4], dim=1)
                loss2 = loss_function(rot_p,
                                      rots,
                                      contrastive3_p,
                                      contrastive4_p,
                                      imgs_recon,
                                      imgs,
                                      mask,
                                      only_mae=True)

                loss = loss1 + loss2 + mutual_loss1
                # loss = loss1 + loss2

                mutual_loss2 = None
                if self.opt.mutual_learning_on_more_view:
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

                    def _align_rot(x, src_rot, dst_rot):
                        x = ensure_five_dims(x)
                        return view_transforms.rotation_transforms[dst_rot](
                            view_transforms.rotation_inverse_transforms[src_rot]
                            (x)).contiguous()

                    # [B, C, H, W, D]
                    rec_x4_aligned = torch.stack([
                        _align_rot(val, src_rot.item(), dst_rot.item())
                        for val, src_rot, dst_rot in zip(rec_x4, rot2, rot1)
                    ])
                    # [B, 1, H, W, D]
                    mask2_aligned = torch.concat([
                        _align_rot(mask2[None, None], src_rot.item(),
                                   dst_rot.item())
                        for src_rot, dst_rot in zip(rot2, rot1)
                    ])
                    mask_intersection = torch.logical_and(mask2_aligned, mask1)
                    # Rescale to the same scale of mutual_loss1
                    rescaler = (mask1.sum() * mask2_aligned.size(0) /
                                (mask2_aligned.sum() + 1e-6))
                    mutual_loss2 = mutual_loss_function(
                        rec_x4_aligned, rec_x1, mask_intersection) * rescaler

                    # self.loss = loss + mutual_loss2
                    self.loss = loss

        else:
            image, target = data["data_image"], data["data_mask"]
            image, target = image.cuda(), target.cuda()
            
            # print(image.shape)
            # print(target.shape)
            
            target = self.prep_label(target)
            # print(target.shape)
            pred = self.model(image)
            self.loss = self.loss_fn(pred, target)
        
        self.optim.zero_grad()
        self.loss.backward()
        self.gradient_clipping()
        self.optim.step()
        self.train_loss += self.loss.item()

    def warp_ds(self):
        ds_weights = np.array([1 / (2 ** i) for i in range(self.opt.num_pool)])
        self.loss_fn = MultipleOutputLoss(self.loss_fn, self.opt.num_classes, ds_weights)

    def prep_label(self, target):
        return target

    def gradient_clipping(self):
        pass

    def lr_scheduler(self):
        pass

    def predict(self, data, **kwargs):
        return sliding_window_inference(
            inputs=data, 
            roi_size=self.opt.crop_size, 
            sw_batch_size=self.opt.sw_batch_size, 
            predictor=self.model,
            overlap=self.opt.overlap,
            mode=self.opt.blend_mode,
            sigma_scale=self.opt.blend_sigma,
            padding_mode=self.opt.padding_mode,
            cval=self.opt.padding_val)

    def post_pred(self, pred):
        pred = decollate_batch(pred)[0]
        pred = AsDiscrete(argmax=True, to_onehot=self.opt.num_classes)(pred)
        pred = pred.unsqueeze(0)
        return pred

    def post_label(self, target):
        return one_hot(target, self.opt.num_classes, dim=1)

    def get_dice(self, pred, target, include_background):
        dice = np.array(compute_dice(pred, target)[0].cpu())
        if not include_background:
            dice = dice[1:]
        return dice
    
    def get_edge_points(self, img):
        """
        get edge points of a binary segmentation result
        """
        img = img.cpu()
        dim = len(img.shape)
        if (dim == 2):
            strt = ndimage.generate_binary_structure(2, 1)
        else:
            strt = ndimage.generate_binary_structure(3, 1)  # 3D structuring element: 6-connectivity neighborhood
        ero = ndimage.morphology.binary_erosion(img, strt)
        edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
        return edge

    def get_binary_hausdorff95(self, s, g, spacing=None):
        """
        get the hausdorff distance between a binary segmentation and the ground truth
        inputs:
            s: a 3D or 2D binary image for segmentation
            g: a 2D or 2D binary image for ground truth
            spacing: a list for image spacing, length should be 3 or 2
        """
        s_edge = self.get_edge_points(s)
        g_edge = self.get_edge_points(g)
        image_dim = len(s.shape)
        assert (image_dim == len(g.shape))
        if (spacing == None):
            spacing = [1.0] * image_dim
        else:
            assert (image_dim == len(spacing))
        img = np.zeros_like(s)
        if (image_dim == 2):
            s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
            g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
        elif (image_dim == 3):
            s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
            g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)
        
        dist_list1 = s_dis[g_edge > 0]
        dist_list1 = sorted(dist_list1)
        if len(dist_list1) == 0:
            dist1 = float('inf')
        else:
            dist1 = dist_list1[int(len(dist_list1) * 0.95)]

        dist_list2 = g_dis[s_edge > 0]
        dist_list2 = sorted(dist_list2)
        if len(dist_list2) == 0:
            dist2 = float('inf')
        else:
            dist2 = dist_list2[int(len(dist_list2) * 0.95)]
        return max(dist1, dist2)

    def get_binary_assd(self, s, g, spacing=None):
        """
        get the average symetric surface distance between a binary segmentation and the ground truth
        inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
        """
        s_edge = self.get_edge_points(s)
        g_edge = self.get_edge_points(g)
        image_dim = len(s.shape)
        assert (image_dim == len(g.shape))
        if (spacing == None):
            spacing = [1.0] * image_dim
        else:
            assert (image_dim == len(spacing))
        img = np.zeros_like(s)
        if (image_dim == 2):
            s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
            g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
        elif (image_dim == 3):
            s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
            g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

        ns = s_edge.sum()
        ng = g_edge.sum()
        s_dis_g_edge = s_dis * g_edge
        g_dis_s_edge = g_dis * s_edge
        assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
        return assd

    def compute_metrics_for_batch(self, pred, target, HD95=True):
        """
        Compute ASD and HD95 for a batch of predictions and targets
        """
        batch_size = pred.shape[1]
        asd_list = []
        hd95_list = []

        for b in range(1, batch_size):
            single_pred = pred[:, b, ...].squeeze(0)
            single_target = target[:, b, ...].squeeze(0)
            asd = self.get_binary_assd(s=single_pred, g=single_target)
            if HD95:
                hd95 = self.get_binary_hausdorff95(s=single_pred, g=single_target)
            else:
                hd95 = None
            asd_list.append(asd)
            hd95_list.append(hd95)

        # avg_asd = np.mean(asd_list)
        # avg_hd95 = np.mean(hd95_list)
        return np.array(asd_list), np.array(hd95_list)
    
    def load_ckpt(self):
        ckpt_path = osp.join(self.opt.expr_dir, f'{self.opt.epoch}')
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['state_dict'])    
          
        optimizer_state_dict = ckpt['optimizer_state_dict']
        for state in optimizer_state_dict['state'].values():
            if 'step' not in state:
                state['step'] = torch.tensor(20000)
                
        self.optim.load_state_dict(ckpt['optimizer_state_dict'])
        self.epoch = ckpt['epoch']
        self.best_metric = ckpt['best_metric']
        self.best_metric_epoch = ckpt['best_metric_epoch']
        self.optim.param_groups[0]['lr'] = poly_lr(self.epoch + 1, self.opt.max_epoch, self.opt.init_lr, 0.9)
        
        update_log(f"model and optimizer are initialized from {ckpt_path}", self.run_log)
        update_log((f"Epoch={self.epoch}, LR={np.round(get_lr(self.optim), decimals=6)}, "
            f"best_metric={self.best_metric}, best_epoch={self.best_metric_epoch} now"), self.run_log)
        
        self.epoch += 1

    def save_ckpt(self):
        if self.val_dice > self.best_metric:
            self.epochs_no_improve = 0
            self.best_metric = self.val_dice
            self.best_metric_epoch = self.epoch
            torch.save({'epoch': self.epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        'best_metric': self.best_metric,
                        'best_metric_epoch': self.best_metric_epoch,
                        'step': self.optim.step()},
                        osp.join(self.opt.expr_dir, f'best_model.pth'))
        else:
            self.epochs_no_improve += 1

    def run(self):
        # if self.epoch % 100 == 0:
        update_log(('\nexperiment timestamp: '
            f'{get_time():%Y-%m-%d %H:%M:%S}'), self.run_log)

        if self.opt.do_ds:
            self.warp_ds()

        while self.epoch <= self.max_epoch:
            start = time()
            self.train()

            if self.epoch > self.skip_val_epoch:
                if self.epoch % self.val_interval == 0:
                    
                    self.valid()
                    if self.epochs_no_improve == self.opt.early_stop:
                        update_log('Early Stopping', self.run_log)
                        break

                    torch.save(
                        {'epoch': self.epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        'best_metric': self.best_metric,
                        'best_metric_epoch': self.best_metric_epoch,
                        'step': self.optim.step()},
                        osp.join(self.opt.expr_dir, f'current_model.pth'))
            # if self.epoch % 100 == 0:
            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                f"lr: {np.round(get_lr(self.optim), decimals=6)}"), self.run_log)        
            
            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                f"This epoch took {time()-start:.2f} s"), self.run_log)

            self.epoch += 1

        print(f"training completed, best_metric={self.best_metric:.4f} at epoch={self.best_metric_epoch}")
