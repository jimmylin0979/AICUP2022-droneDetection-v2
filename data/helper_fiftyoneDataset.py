import fiftyone as fo

# # VisDrone2019
# dataset = fo.Dataset.from_dir(
#     dataset_dir="./VisDrone2019",
#     dataset_type=fo.types.YOLOv5Dataset, # COCODetectionDataset,
#     split="train",
#     name="VisDrone2019-train"
# )

### Training Dataset_v5 ###
# dataset = fo.Dataset.from_dir(
#     dataset_dir="./Training Dataset_v5 tiled",
#     dataset_type=fo.types.YOLOv5Dataset, # COCODetectionDataset,
#     split="train",
#     name="yolov5_train_tiled"
# )

# dataset = fo.Dataset.from_dir(
#     dataset_dir="./Training Dataset_v5 tiled",
#     dataset_type=fo.types.YOLOv5Dataset, # COCODetectionDataset,
#     split="val",
#     name="yolov5_valid_tiled"
# )

# dataset = fo.Dataset.from_dir(
#     dataset_dir="./Training Dataset_v5",
#     dataset_type=fo.types.YOLOv5Dataset, # COCODetectionDataset,
#     split="train",
#     name="yolov5_train"
# )
# dataset = fo.Dataset.from_dir(
#     dataset_dir="./Training Dataset_v5",
#     dataset_type=fo.types.YOLOv5Dataset, # COCODetectionDataset,
#     split="val",
#     name="yolov5_valid"
# )

### Training Dataset_v5 mmdet ###
# dataset = fo.Dataset.from_dir(
#     dataset_dir="./Training Dataset_v5 mmdet/train_coco_valid",
#     dataset_type=fo.types.COCODetectionDataset,
#     name="coco_valid"
# )

# dataset = fo.Dataset.from_dir(
#     dataset_dir="./Training Dataset_v5 mmdet/train_coco_valid_tiled",
#     dataset_type=fo.types.COCODetectionDataset,
#     name="coco_valid_tiled"
# )

### FusionDataset ###
dataset = fo.Dataset.from_dir(
    dataset_dir="./FusionDataset",
    dataset_type=fo.types.YOLOv5Dataset,
    split="train",
    name="FusionDataset_train"
)

dataset = fo.Dataset.from_dir(
    dataset_dir="./FusionDataset",
    dataset_type=fo.types.YOLOv5Dataset,
    split="val",
    name="FusionDataset_valid"
)

session = fo.launch_app()
session.wait()