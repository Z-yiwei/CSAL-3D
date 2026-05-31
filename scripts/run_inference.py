#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai.utils import set_determinism
from options.options import Options
from utils.util import parse_options
from data_loading.BasicLoader import BasicLoader
# from data_loading.ListLoader import ListLoader
from transform.BasicTransform import BasicTransform, BasicTransform_for_BraTs
from models.NetworkLoader import NetworkLoader
from inference.BasicPredictor import BasicPredictor


def main() -> None:
    opt = parse_options(Options(), save_config=False)

    # reproducibility
    set_determinism(seed=opt.seed)  

    transform = BasicTransform(
        crop_size=opt.crop_size, 
        num_samples=opt.num_samples,
        modality=opt.modality)

    data = BasicLoader(
        tr=transform, 
        opt=opt, 
        phase='test')

    model = NetworkLoader(opt).load()

    BasicPredictor(data=data, model=model, opt=opt).make_inference()


if __name__ == "__main__":
    main()
