from ensemble_boxes import *
import os
import argparse
import pandas as pd

from tqdm import tqdm
import cv2

def xyxy2xywh(box, image_w, image_h):
    x1, y1, x2, y2 = box
    return [int(x1*image_w), int(y1*image_h), int((x2-x1)*image_w), int((y2-y1)*image_h)]



def ensemble(args):
    root_sample = r'./data/Public Testing Dataset_v2/public/'
    root_model = "runs/yolov7/detect"
    ensemble_path = "runs/yolov7/detect/ensemble/"


    if os.path.isdir(ensemble_path):
        print("ensemble folder name input: ")
        ensemble_name = input()
        ensemble_path = "runs/yolov7/detect/" + ensemble_name
        os.mkdir(ensemble_path)
    else:
        os.mkdir(ensemble_path)
        

    model = []
    data = {}
    # runs/yolov7/
    for result in os.listdir(root_model):
        # 
        if not os.path.isdir(f"{root_model}/{result}"):
            continue
        if "ensemble" in result:
            continue
    
        
        output_path = f"{root_model}/{result}/predictions.csv"
        print(output_path)
        if os.path.exists(output_path):
            model.append(result)
            data[result] = pd.read_csv(output_path, header=None)
    print(model)
    weights = []
    for i in range(len(model)):
        print("Weights input: ")
        weights.append(float(input()))
    
    # print(data)

    for filepath in tqdm(sorted(os.listdir(root_sample))):
        if filepath.endswith('.png'):
            img_results = {}
            # A dataset is consist of many samples 
            # create a sample

            img = cv2.imread(f"{root_sample}/{filepath}")
            # (1080, 1920, 3)
            w, h = img.shape[1], img.shape[0]

            ## get img result from all model
            img_name = filepath[:-4]
            for model_name in data:
                df = data[model_name]

                img_results[model_name] = df.loc[df.iloc[:, 0] == img_name, :]

            # weights = [0, 1, 1]
            box_list = []
            score_list = []
            label_list = []
            iou_thr = 0.5
            skip_box_thr = 0.0001
            sigma = 0.1

            for model_n in img_results:
                df = img_results[model_n]
                
                uni_model_box = []
                uni_model_score = []
                uni_model_label = []
                for idx, content in df.iterrows():
                    # print(content)
                    label = int(content[1])
                    score = float(content[6])
                    bounding_box_coor = [int(content[i]) for i in range(2, 6)]
                    # print(f"Before label: {bounding_box_coor}")
                    bounding_box_coor[2] += bounding_box_coor[0]
                    bounding_box_coor[3] += bounding_box_coor[1]
                    bounding_box_coor[0] /= w
                    bounding_box_coor[1] /= h
                    bounding_box_coor[2] /= w
                    bounding_box_coor[3] /= h
                    # print(f"after xyxy: {bounding_box_coor}")
                    # print(f"after label, score: {label}, {score}")


                    uni_model_box.append(bounding_box_coor)
                    uni_model_score.append(score)
                    uni_model_label.append(label)


                box_list.append(uni_model_box)
                label_list.append(uni_model_label)
                score_list.append(uni_model_score)

            # print(box_list)
            boxes, scores, labels = weighted_boxes_fusion(box_list, score_list, label_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            # print(boxes)

            with open(ensemble_path + "/predictions.csv", "a") as fw:
                
                for idx, box in enumerate(boxes):

                    content = []
                    content.append(img_name)
                    content.append(int(labels[idx]))

                    content.extend(xyxy2xywh(box, w, h))
                    content.append(float(scores[idx]))
                    content = [str(j) for j in content]
                    # print(f"{content}")
                    fw.write(','.join(content) + '\n')   # save predictions with coco format
        





                    
                
                    
                    




            
                















if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_method", type=str, default = 'wbf', help = "Choose ensemble methods")
    args = parser.parse_args()

    ensemble(args)
