# 
from mmcv import Config
#
import argparse

# 
if __name__ == "__main__":
    
    #
    
    parser = argparse.ArgumentParser(description='AICUP2022 - Agriculture33')
    parser.add_argument('--file', type=str, required=True,
                                help='The folder to store the training stats of current model')
    args = parser.parse_args()

    #
    filename = args.file
    cfg = Config.fromfile(filename)
    print(cfg.pretty_text)