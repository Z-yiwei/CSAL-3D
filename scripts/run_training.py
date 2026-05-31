#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 06/02/2022

# Program description
# General training pipeline for medical image segmentation
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from monai.utils import set_determinism
from options.options import Options
from utils.util import parse_options
from data_loading.BasicLoader import BasicLoader
from transform.BasicTransform import BasicTransform, BasicTransform_for_BraTs
from models.NetworkLoader import NetworkLoader
from network_training.TrainerV2 import TrainerV2
import torch

def main() -> None:
    torch.backends.cudnn.benchmark = False
    opt = parse_options(Options())

    # reproducibility
    set_determinism(seed=opt.seed)  

    transform = BasicTransform_for_BraTs(
        crop_size=opt.crop_size, 
        num_samples=opt.num_samples,
        modality=opt.modality)

    data = BasicLoader(
        tr=transform, 
        opt=opt, 
        phase='train')

    model = NetworkLoader(opt).load()

    TrainerV2(data=data, model=model, opt=opt).run()


if __name__ == "__main__":
    print(torch.__version__)
    main()
