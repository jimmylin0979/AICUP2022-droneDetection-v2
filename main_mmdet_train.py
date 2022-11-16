# mmdetection 
import mmdet
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed

import mmcv
from mmcv import Config

# 
import os.path as osp

def main_train(cfg):
    """
    """

    #
    datasets = [build_dataset(cfg.data.train)]

    #
    model = build_detector(cfg.model)
    model.CLASSES = datasets[0].CLASSES

    # 
    train_detector(model, datasets, cfg, distributed=False, validate=True)


if __name__ == "__main__":
    # main_train(cfg)
    pass