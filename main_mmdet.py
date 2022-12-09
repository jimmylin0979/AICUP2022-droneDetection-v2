#
import mmcv
from mmcv import Config
import mmdet

#
import argparse
import os
import os.path as osp
import shutil

#
from main_mmdet_train import main_train
from main_mmdet_eval import main_eval

###################################################################################

if __name__ == '__main__':
    
    """
    Train example:
    > dir="results/faster_rcnn"
    > python3 main.py --mode=train --config_file=configs/faster_rcnn.yaml
    """

    """
    Evaluate example:
    > dir="results/faster_rcnn"
    > python3 main.py --mode=eval --config_file=$dir/config.py --with_confidence=True
    """

    # 1.
    parser = argparse.ArgumentParser(description='AICUP2022 - Agriculture33')
    parser.add_argument('--mode', '-m', default='train', type=str, required=True,
                                help='Selecting whether to train or to evaluate')
    parser.add_argument('--config_file', type=str, required=True,
                                help='The folder to store the training stats of current model')
    parser.add_argument('--with_confidence', default=True, type=bool, required=False,
                                help='Whether to include condifence into prediction.csv')
    args = parser.parse_args()

    # 2. 
    cfg = None
    cfg = Config.fromfile(args.config_file)
    print(cfg)
    
    # 
    mmdet.datasets.coco.CocoDataset.CLASSES = cfg.CLASSES

    # 3.
    assert args.mode == "train" or args.mode == "eval", "Please select either train or eval mode"
    # mode train
    if args.mode == 'train':
        #
        isHasCheckpoint = False
        if os.path.isdir(cfg.work_dir):
            for filename in os.listdir(cfg.work_dir):
                if filename.endswith('.pth'):
                    isHasCheckpoint = True
                    break
        assert isHasCheckpoint == False, "cfg.work_dir is already in used, please rename the session"

        # First copy the config into the training folder
        print(cfg.work_dir)
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        src = f"{args.config_file}"
        dst = f"{cfg.work_dir}/config.py"
        shutil.copy(src, dst)
        # 
        main_train(cfg)
    # mode eval
    elif args.mode == 'eval':
        main_eval(cfg, args.with_confidence)

###################################################################################