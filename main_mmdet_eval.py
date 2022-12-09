# mmdetection 
import mmdet
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result_pyplot, init_detector

import mmcv
from mmcv import Config

# 
import os
import os.path as osp
from tqdm import tqdm

def main_eval(cfg, with_confidence=False):
    """
    """

    #     
    checkpoint = cfg.checkpoint # "./results/faster_rcnn/epoch_10.pth"
    model = init_detector(cfg, checkpoint, device="cuda:0")

    #
    if 'classes' not in cfg.data.train or 'classes' not in cfg.data.train.dataset:
        print("Fixing Class mislabeled")

    # img = mmcv.imread("./train_img0002.png")
    with open("predictions.csv", "w") as fw:
        for filename in tqdm(sorted(os.listdir(cfg.pwd_data_test))):
            if filename.endswith("png"):
                img = mmcv.imread(f"{cfg.pwd_data_test}/{filename}")
                result = inference_detector(model, img)
                # print(type(result))
                # print(result[0][0])

                # 
                for clas in range(4):          # For each category
                    predictions = result[clas]
                    for r in predictions:        # For each prediction in each category
                        # For each coodinate in each category
                        r = r.tolist()
                        r[2] = r[2] - r[0]
                        r[3] = r[3] - r[1]

                        # r[-1] is the condifence score
                        #   User should set with_confidence=True when they want to check the prediction with fiftyone,
                        #        should set with_confidence=False when just need a prediction to upload
                        if with_confidence:
                            confidence = float(r[4])
                        r = [round(r[i]) for i in range(len(r) - 1)]
                        if with_confidence:
                            r.append(confidence)

                        # w, h should be no lower than 1
                        r[2] = 1 if r[2] <= 0 else r[2]
                        r[3] = 1 if r[3] <= 0 else r[3]

                        # x, y should be inside the width, height of the image
                        # print(filename, img.shape)
                        if r[0] < 0 or r[0] > img.shape[1]:
                            r[0] = 0 if r[0] < 0 else img.shape[1]
                        if r[1] < 0 or r[1] > img.shape[0]:
                            r[1] = 0 if r[1] < 0 else img.shape[0]
                        
                        r = [str(r[i]) for i in range(len(r))]
                        output = ",".join(r)
                        
                        output_clas = clas
                        # Remember to define 'classes' attributes in your config file
                        if 'classes' not in cfg.data.train or 'classes' not in cfg.data.train.dataset:
                            # print('old style')
                            if clas == 2:
                                output_clas = 3
                            elif clas == 3:
                                output_clas = 2
                        fw.write(f"{filename[:-4]},{output_clas},{output}\n")
                
                # show_result_pyplot(model, img, result)

if __name__ == "__main__":
    # main_train(cfg)
    pass