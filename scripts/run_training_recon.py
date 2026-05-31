#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/10/2023

# Program description
# Reconstruction task: I -> I' with Auto-Encoder
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch

from monai.utils import set_determinism
from options.options import Options
from utils.util import parse_options
from data_loading.BasicLoader import BasicLoader
from transform.ReconTransform import ReconTransform
from models.NetworkLoader import NetworkLoader
from network_training.TrainerV2_Recon import TrainerV2_Recon
from network_training.TrainerV3 import TrainerV3


def main() -> None:
    opt = parse_options(Options())

    # reproducibility
    set_determinism(seed=opt.seed)  

    transform = ReconTransform(
        crop_size=opt.crop_size, 
        num_samples=opt.num_samples,
        modality=opt.modality)

    data = BasicLoader(
        tr=transform, 
        opt=opt, 
        phase='train')

    model = NetworkLoader(opt).load()

    TrainerV2_Recon(data=data, model=model, opt=opt).run()
    # TrainerV3(data=data, model=model, opt=opt).run()
    


if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    main()
