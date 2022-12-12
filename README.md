# AICUP2022 - droneDetection

## About the project

## Getting Start

### Installation

```bash
# clone the repository
git clone https://github.com/jimmylin0979/AICUP2022-droneDetection-v2.git

# go to repository folder
cd AICUP2022-droneDetection-v2

# pip install required packages
pip install -r requirements.txt
```
---

### Dataset

We package the complicated setup procedure into a bash script, you can simply complete setup by following steps below.  
To check what was happening during setting, please have a see in `./data/dataset.sh` 

1. Download dataset zip file and place them into below format:

    ```bash
    ./data
    ├── dataset.sh
    ├── Private Testing Dataset_v2.zip
    ├── Public Testing Dataset_v2.zip
    ├── Training Dataset_v5.zip
    ├── VisDrone2019-DET-train.zip
    ├── ...

    ```
2. Run the bash script to setup the dataset

    ```bash
    cd data
    bash dataset.sh
    ```

It will take about 20 ~ 30 minutes, depends on your CPU power, to complete the setup. (It took 20 minutes for i5-12400 to finish setup)   
The final directory structure should be like (we only list some important dataset folders here):

```
./data
├── FusionDataset
│   ├── images
|   └── labels
|
├── Private Testing Dataset_v2
│   ├── private
|   └── private_tiled
|
├── Public Testing Dataset_v2
│   ├── public
|   └── public_tiled
|
├── train
│   └── public
│       ├── img1001.png
|       ├── img1001.txt
|       ├── ....
|       ├── img1500.png
│       └── img1500.txt
|
├── Training Dataset_v5
│   ├── images
|   └── labels
|
├── Training Dataset_v5 tiled
│   ├── images
|   └── labels
|
├── ...
```

<!-- Second, run the command below to transform the dataset

```bash
# 1. Convert dataset format into YOLOv5Dataset
cd data
python helper_splitDataset_yolov5.py --root-src './Training Dataset_v5' --root-dst './Training Dataset_v5'

# 2. Tiling
cd ../TiledSet
python main.py --mode tile --root-src train_yolo_train --root-dst train_yolo_train

# 3. Take gamma correction on the dataset
``` -->

---

### Training

Be sure to start training after you finish all transformation on dataset (ex. tiling, gamma correction, and so on)

1. Download the pretrained weights of `yolov7-e6e_training.pt` in transfer learning section from https://github.com/WongKinYiu/yolov7,
    place the weights into folder `yolov7/weights`

2. Run the command below to start training, the results will be stored in `./results` folder,   
      the follow command will consume about 20G memory usage on GPU, you can modify the batch size depends on your personal device. 

    ```bash
    cd yolov7
    python train_aux.py --workers 4 --device 0 --batch-size 4 --data data/fusionDataset.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --weights weights/yolov7-e6e_training.pt --name yolov7-e6e-aug-tile-fusion --hyp data/hyp.scratch.custom.yaml --label-smoothing 0.1
    ```

---

### Evaluating

Be sure to start training after you finish all transformation on dataset (ex. tiling, gamma correction, and so on)

+ Weights (Private score 0.758381): https://drive.google.com/uc?export=download&id=1zGFK57FCeo-ylCeEoirI0ilJg1aFtAdE

1. Run the below command to start evluating, the predictions will be stored in `./results/yolov7/train/`
    ```bash
    cd yolov7

    # Public dataset (tiling)
    python detect.py --weights ../results/yolov7/train/yolov7-e6e-aug-tile-fusion/weights/best.pt --source ../data/Public\ Testing\ Dataset_v2/public_tiled/data/ --img-size 1280 --conf-thres 0.4 --device 0 --save-txt --save-conf --nosave --augment --name yolov7-e6e-aug-tile-fusion-public

    # Private dataset (tiling)
    python detect.py --weights ../results/yolov7/train/yolov7-e6e-aug-tile-fusion/weights/best.pt --source ../data/Private\ Testing\ Dataset_v2/private_tiled/data/ --img-size 1280 --conf-thres 0.4 --device 0 --save-txt --save-conf --nosave --augment --name yolov7-e6e-aug-tile-fusion-private
    ```

2. Merge tiling predictions
    ```bash
    cd TiledSet

    # Public dataset (merged)
    python main.py --mode merge --root-src ../data/'Public Testing Dataset_v2'/public --root-dst ../results/yolov7/detect/yolov7-e6e-aug-tile-fusion-public/predictionstile.csv
    mv predictions.csv predictions_public.csv
    mv predictions_public.csv ../results/yolov7/detect/yolov7-e6e-aug-tile-fusion-public/predictions_public.csv

    # Private dataset (merged)
    python main.py --mode merge --root-src ../data/'Private Testing Dataset_v2'/private --root-dst ../results/yolov7/detect/yolov7-e6e-aug-tile-fusion-private/predictions_tile.csv
    mv predictions.csv predictions_private.csv
    mv predictions_private.csv ../results/yolov7/detect/yolov7-e6e-aug-tile-fusion-private/predictions_private.csv
    ```
---

## Visualization

```bash
# Public visualization
python main_visualization.py

# Private visualization
python main_visualization_private.py
```

## Acknowledgments

* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [yolov7](https://github.com/WongKinYiu/yolov7)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)