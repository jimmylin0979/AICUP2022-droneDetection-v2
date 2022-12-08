# 
import fiftyone as fo
import pandas as pd

import os
from tqdm import tqdm
import cv2

def main():

    #
    CLASSES = ('car', 'hov', 'person', 'motorcycle')
    root_sample = r'./data/Private Testing Dataset_v2/private/'
    # root_sample = r'./data/Public Testing Dataset_v2/public_tiled/data'
    root_detections = r'./results'

    #
    dfs = []
    model = []

    # ./results/mmdet
    blacklist = ['faster_rcnn', 'faster_rcnn_x_101']
    root_detections_mmdet = f"{root_detections}/mmdet"
    for result in os.listdir(root_detections_mmdet):
        # 
        if not os.path.isdir(f"{root_detections_mmdet}/{result}"):
            continue
        if result in blacklist:
            continue
        
        output_path = f"{root_detections_mmdet}/{result}/predictions_private.csv"
        print(output_path)
        if os.path.exists(output_path):
            df = pd.read_csv(output_path, header=None)
            dfs.append(df)
            model.append(result)
    print(model)

    # ./results/yolov7/train
    blacklist = ['']
    root_detections_yolov7 = f"{root_detections}/yolov7/detect"
    for result in os.listdir(root_detections_yolov7):
        # 
        if not os.path.isdir(f"{root_detections_yolov7}/{result}"):
            continue
        if result in blacklist:
            continue
        
        output_path = f"{root_detections_yolov7}/{result}/predictions_private.csv"
        print(output_path)
        if os.path.exists(output_path):
            df = pd.read_csv(output_path, header=None)
            dfs.append(df)
            model.append(result)
    print(model)

    #
    # Create a fiftyone dataset
    dataset = fo.Dataset(name="droneDetection_private")
    
    # 
    index = [0 for _ in range(len(dfs))]
    for filepath in tqdm(sorted(os.listdir(root_sample))):
        if filepath.endswith('.png'):
            # A dataset is consist of many samples 
            # create a sample

            sample = fo.Sample(filepath=f"{root_sample}/{filepath}")
            img = cv2.imread(f"{root_sample}/{filepath}")
            
            # (1080, 1920, 3)
            w, h = img.shape[1], img.shape[0]

            # 
            for i in range(len(dfs)):
                detections = []
                # 
                while index[i] < dfs[i].shape[0]:
                    # for obj in model_output[filepath]:
                    content = dfs[i].iloc[index[i], :].values
                    # print(content)

                    num_cols = dfs[i].shape[1]
                    content = [content[j] for j in range(num_cols)]
                    # content = content.strip(" \n")
                    # content = content.split(",")
                    # print(content)

                    #
                    if content[0] != filepath[:-4]:
                        break
                    index[i] += 1

                    # 
                    label = CLASSES[int(content[1])]
                    bounding_box_coor = [int(content[i]) for i in range(2, 6)]
                    bounding_box_coor[0] /= w
                    bounding_box_coor[1] /= h
                    bounding_box_coor[2] /= w
                    bounding_box_coor[3] /= h
                    bounding_box = bounding_box_coor

                    # 
                    confidence = None
                    if num_cols > 6:
                        confidence = float(content[6])

                    # print(filepath, i, label, bounding_box)

                    # 
                    detections.append(
                        # fo.Detection(label=str(int(content[1])), bounding_box=bounding_box, confidence=confidence)
                        fo.Detection(label=label, bounding_box=bounding_box, confidence=confidence)
                    )
                
                sample[f"{model[i]}"] = fo.Detections(detections=detections)
                # sample.save()
            
            dataset.add_sample(sample)
        
    session = fo.launch_app()
    session.wait()

"""
Reference : https://voxel51.com/docs/fiftyone/tutorials/evaluate_detections.html
<Detection: {
    'id': '6065d1e04976aab284081d83',
    'attributes': BaseDict({}),
    'tags': BaseList([]),
    'label': 'potted plant',
    'bounding_box': BaseList([
        0.37028125,
        0.3345305164319249,
        0.038593749999999996,
        0.16314553990610328,
    ]),
    'mask': None,
    'confidence': None,
    'index': None,
    'iscrowd': 0.0,
    'area': 531.8071000000001,
}>
"""

if __name__ == "__main__":
    main()