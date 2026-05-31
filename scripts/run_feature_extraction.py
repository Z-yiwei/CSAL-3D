#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/11/2023

# Program description
# Run the pipeline to measure the representativeness via image reconstruction
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from monai.transforms import *
from monai.utils import set_determinism
from options.options import Options
from utils.util import parse_options
from data_loading.AL_Loader import AL_Loader
from transform.BasicTransform import BasicTransform
from models.NetworkLoader import NetworkLoader
from inference.FeatureExtracter_plus import FeatureExtracter


def main() -> None:
    opt = parse_options(Options(), save_config=False)

    # reproducibility
    set_determinism(seed=opt.seed)  

    transform = BasicTransform(
        crop_size=opt.crop_size, 
        num_samples=opt.num_samples,
        modality=opt.modality)

    data = AL_Loader(tr=transform, opt=opt)

    model = NetworkLoader(opt).load()

    FeatureExtracter(data=data, model=model, opt=opt).extract(mode=opt.mode)


if __name__ == "__main__":
    main()
