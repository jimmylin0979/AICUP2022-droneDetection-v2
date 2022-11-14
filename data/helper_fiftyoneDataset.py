import fiftyone as fo

# # VisDrone2019
# dataset = fo.Dataset.from_dir(
#     dataset_dir="./VisDrone2019",
#     dataset_type=fo.types.YOLOv5Dataset, # COCODetectionDataset,
#     split="train",
#     name="VisDrone2019-train"
# )

# Training Dataset_v5
dataset = fo.Dataset.from_dir(
    dataset_dir="./Training Dataset_v5",
    dataset_type=fo.types.YOLOv5Dataset, # COCODetectionDataset,
    split="train",
    name="train"
)

session = fo.launch_app()
session.wait()