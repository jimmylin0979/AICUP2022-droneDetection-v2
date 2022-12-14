#
import argparse

# 
import logging
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)

#
from src import TiledSet

if __name__ == "__main__":
    
    """
    """
    
    # Argument Parser
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mode", type=str, default="tile", required=True)
    parser.add_argument("--root-src", type=str, required=True)
    parser.add_argument("--root-dst", type=str)
    # # There maybe be more than one variable to describe the tile size 
    # parser.add_argument("--tileSize", nargs="+", type=int, required=True,
    #                         help="")
    # parser.add_argument("--stride", nargs="+", type=int, required=True,
    #                         help="")
    opts = parser.parse_args()

    # # 
    # if len(args.tileSize) == 1:
    #     args.tileSize = [args.tileSize, args.tileSize]
    # args.tileSize = tuple(args.tileSize)
    # if len(args.stride) == 1:
    #     args.stride = [args.stride, args.stride]
    # args.stride = tuple(args.stride)

    # # Call Main Function 
    # tileSet = TileSet(args.tileSize, args.stride)
    # tileSet.tile(args.root_src, args.root_dst)

    ###################################################################################################
    # Uncomment this to generate tiled train/valid dataset (only support fiftyone.types.YOLOv4Dataset)
    # Testing .tile() (PASS)
    # tileSet = TiledSet((540, 960), (540 // 2, 960 // 2))
    # root_src = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/train_yolo_train"
    # root_dst = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/train_yolo_train_tiled"
    # tileSet.tile(root_src=root_src, root_dst=root_dst)

    # tileSet = TiledSet((540, 960), (540 // 2, 960 // 2))
    # root_src = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/train_yolo_valid"
    # root_dst = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/train_yolo_valid_tiled"
    # tileSet.tile(root_src=root_src, root_dst=root_dst)
    
    if opts.mode == "tile":

        tileSet = TiledSet((540, 960), (540 // 2, 960 // 2))
        root_src = opts.root_src
        root_dst = opts.root_dst
        tileSet.tile(root_src=root_src, root_dst=root_dst)
    
    ###################################################################################################
    # # # Uncomment this to generate tiled test dataset (only support fiftyone.types.YOLOv4Dataset)
    # tileSet = TiledSet((540, 960), (540 // 2, 960 // 2))
    
    # # root_src = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/test/public"
    # # root_dst = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/test/public_tiled"
    # root_src = "/home/user/??????/FredH/AICUP/AICUP2022-droneDetection-v2/data/Private Testing Dataset_v2/private"
    # root_dst = "/home/user/??????/FredH/AICUP/AICUP2022-droneDetection-v2/data/Private Testing Dataset_v2/private_tiled"
    
    if opts.mode == "tile_test":

        tileSet = TiledSet((540, 960), (540 // 2, 960 // 2))
        root_src = opts.root_src
        root_dst = opts.root_dst
        tileSet.tile(root_src=root_src, root_dst=root_dst, with_annotations=False)
    
    ###################################################################################################
    # Uncomment this to generate merged predictions of tiled test dataset (csv)

    # # # # Testing .merge (PASS)
    # tileSet = TiledSet((540, 960), (540 // 2, 960 // 2))
    # root_anns = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/test/public_tiled"
    # # root_originalImage = "/home/user/??????/FredH/AICUP/AICUP2022-droneDetection-v2/data/Private Testing Dataset_v2/private"
    # root_originalImage = "/home/user/??????/FredH/AICUP/AICUP2022-droneDetection-v2/data/Public Testing Dataset_v2/public"
    # tileSet.merge(prediction_path="../results/yolov7/detect/yolov7-e6e-fusion-v6/predictions_tiled.csv", root_originalImage=root_originalImage, sep=",")

    if opts.mode == "merge":

        tileSet = TiledSet((540, 960), (540 // 2, 960 // 2))
        # root_originalImage = "/home/user/??????/FredH/AICUP/AICUP2022-droneDetection-v2/data/Public Testing Dataset_v2/public"
        root_originalImage = opts.root_src
        prediction_path = opts.root_dst
        tileSet.merge(prediction_path=prediction_path, root_originalImage=root_originalImage, sep=",")

    ###################################################################################################
    # 
    # tileSet.ensemble(prediction_path="../results/yolov7/detect/ensemble/ensemble_sort.csv", root_originalImage=None, sep=",")
    if opts.mode == "ensemble":
        tileSet = TiledSet((540, 960), (540 // 2, 960 // 2))
        prediction_path = opts.root_src
        tileSet.ensemble(prediction_path=prediction_path, root_originalImage=None, sep=",")