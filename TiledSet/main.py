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
    
    # # Argument Parser
    # parser = argparse.ArgumentParser(description="")
    # parser.add_argument("--root_src", type=str, required=True,
    #                         help="")
    # parser.add_argument("--root_dst", type=str, required=True,
    #                         help="")
    # # There maybe be more than one variable to describe the tile size 
    # parser.add_argument("--tileSize", nargs="+", type=int, required=True,
    #                         help="")
    # parser.add_argument("--stride", nargs="+", type=int, required=True,
    #                         help="")
    # args = parser.parse_args()

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
    # # Uncomment this to generate tiled train/valid dataset (only support fiftyone.types.YOLOv4Dataset)
    # # Testing .tile() (PASS)
    # tileSet = TiledSet((540, 960), (540 // 2, 960 // 2))
    # root_src = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/train_yolo_train"
    # root_dst = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/train_yolo_train_tiled"
    # tileSet.tile(root_src=root_src, root_dst=root_dst)

    # tileSet = TiledSet((540, 960), (540 // 2, 960 // 2))
    # root_src = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/train_yolo_valid"
    # root_dst = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/train_yolo_valid_tiled"
    # tileSet.tile(root_src=root_src, root_dst=root_dst)

    
    ###################################################################################################
    # # Uncomment this to generate tiled test dataset (only support fiftyone.types.YOLOv4Dataset)
    # tileSet = TiledSet((540, 960), (540 // 2, 960 // 2))
    # root_src = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/test/public"
    # root_dst = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/test/public_tiled"
    # tileSet.tile(root_src=root_src, root_dst=root_dst, with_annotations=False)
    
    ###################################################################################################
    # Uncomment this to generate merged predictions of tiled test dataset (csv)

    # # # Testing .merge (PASS)
    tileSet = TiledSet((540, 960), (540 // 2, 960 // 2))
    root_anns = "/home/jimmylin0979/Desktop/datas/AICUP2022-droneDetection/test/public_tiled"
    root_originalImage = "/home/user/桌面/FredH/AICUP/AICUP2022-droneDetection/data/Public Testing Dataset_v2/public"
    tileSet.merge(prediction_path="../results/yolov7/detect/yolov7-e6e-tiled/predictions_tiled.csv", root_originalImage=root_originalImage, sep=",")

    # tileSet.ensemble(prediction_path="../results/ensemble.csv", root_originalImage=root_originalImage, sep=",")