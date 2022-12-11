# Please execute this bash script at the same directory level. ie, inside the ./data folder

# Data root folder
ROOT_FOLDER=data2

#######################################
### Official Dataset Public Testing ###

echo "Official Dataset Public Testing"

# Public Testing Dataset_v2
unzip 'Public Testing Dataset_v2.zip'

mkdir 'Public Testing Dataset_v2'
mv public 'Public Testing Dataset_v2'/public

cd ../TiledSet
python main.py --mode tile_test --root-src ../$ROOT_FOLDER/'Public Testing Dataset_v2'/public --root-dst ../$ROOT_FOLDER/'Public Testing Dataset_v2'/public_tiled

cd ../$ROOT_FOLDER
python helper_imageTransform.py --root 'Public Testing Dataset_v2'/public_tiled/data


#######################################
### Official Dataset Private Testing ###

echo "Official Dataset Private Testing"

# Private Testing Dataset_v2
mkdir 'Private Testing Dataset_v2'
mkdir 'Private Testing Dataset_v2'/private
unzip -P 9TrlWw7x7gW 'Private Testing Dataset_v2.zip' -d 'Private Testing Dataset_v2'/private

cd ../TiledSet
python main.py --mode tile_test --root-src ../$ROOT_FOLDER/'Private Testing Dataset_v2'/private --root-dst ../$ROOT_FOLDER/'Private Testing Dataset_v2'/private_tiled

cd ../$ROOT_FOLDER
python helper_imageTransform.py --root 'Private Testing Dataset_v2'/private_tiled/data


#######################################
### Official Dataset Training ###

echo "Official Dataset Training"

# unzip downloaded datset
unzip 'Training Dataset_v5.zip'

# convert dataset into YOLOv5 format
python helper_splitDataset_yolov5.py --root-src train --root-dst 'Training Dataset_v5'

# # Tiling 
cd ../TiledSet
python main.py --mode tile --root-src ../$ROOT_FOLDER/'Training Dataset_v5' --root-dst ../$ROOT_FOLDER/'Training Dataset_v5 tiled'

cd ../$ROOT_FOLDER
python helper_imageTransform.py --root 'Training Dataset_v5 tiled'/images/train
python helper_imageTransform.py --root 'Training Dataset_v5 tiled'/images/valid

#######################################
### FusionDataset ###

echo "FusionDataset"

# 
unzip VisDrone2019-DET-train.zip
# unzip VisDrone2019-DET-test-dev.zip

# mkdir VisDrone
# mkdir VisDrone/images
# mkdir VisDrone/labels

# mkdir VisDrone/images/train
# mkdir VisDrone/labels/train
# mkdir VisDrone/images/valid
# mkdir VisDrone/labels/valid

# Convert VisDrone into YOLOv5 format 
python helper_visdrone2yolov5.py

# Create folders in FusionData
mkdir FusionDataset
mkdir FusionDataset/images
mkdir FusionDataset/labels
mkdir FusionDataset/images/train
mkdir FusionDataset/images/valid
mkdir FusionDataset/labels/train
mkdir FusionDataset/labels/valid

# 
mv VisDrone2019-DET-train/images/* FusionDataset/images/train
mv VisDrone2019-DET-train/labels/* FusionDataset/labels/train
# Convert label from 10-class into 4-class
python helper_fusionDataset.py

# 
cp -a 'Training Dataset_v5 tiled'/images/train/. FusionDataset/images/train
cp -a 'Training Dataset_v5 tiled'/images/valid/. FusionDataset/images/valid
cp -a 'Training Dataset_v5 tiled'/labels/train/. FusionDataset/labels/train
cp -a 'Training Dataset_v5 tiled'/labels/valid/. FusionDataset/labels/valid