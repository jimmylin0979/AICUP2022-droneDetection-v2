"""
tph-yolov5
@InProceedings{Zhu_2021_ICCV,
    author    = {Zhu, Xingkui and Lyu, Shuchang and Wang, Xu and Zhao, Qi},
    title     = {TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-Captured Scenarios},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2021},
    pages     = {2778-2788}
}
Reference: https://github.com/cv516Buaa/tph-yolov5/blob/main/VisDrone2YOLO_lable.py
"""
import os
import pandas as pd
from PIL import Image

pwd = r"./VisDrone2019-DET-train"
YOLO_LABELS_PATH = f"{pwd}/labels"
VISANN_PATH = f"{pwd}/annotations/"
VISIMG_PATH = f"{pwd}/images/"

def convert(bbox, img_size):
    #将标注visDrone数据集标注转为yolov5
    #bbox top_left_x top_left_y width height
    dw = 1/(img_size[0])
    dh = 1/(img_size[1])
    x = bbox[0] + bbox[2]/2
    y = bbox[1] + bbox[3]/2
    x = x * dw
    y = y * dh
    w = bbox[2] * dw
    h = bbox[3] * dh
    return (x,y,w,h) 

def ChangeToYolo5():
    if not os.path.exists(YOLO_LABELS_PATH):
        os.makedirs(YOLO_LABELS_PATH)
    print(len(os.listdir(VISANN_PATH)))
    for file in os.listdir(VISANN_PATH):
        image_path = VISIMG_PATH + '/' + file.replace('txt', 'jpg')
        ann_file = VISANN_PATH + '/' + file
        out_file = open(YOLO_LABELS_PATH + '/' + file, 'w')
        bbox = pd.read_csv(ann_file,header=None).values
        img = Image.open(image_path)
        img_size = img.size
        for row in bbox:
            if(row[4]==1 and 0<row[5]<11):   
                label = convert(row[:4], img_size)
                out_file.write(str(row[5]-1) + " " + " ".join(str(f'{x:.6f}') for x in label) + '\n')
        out_file.close()

if __name__ == '__main__':
    ChangeToYolo5()