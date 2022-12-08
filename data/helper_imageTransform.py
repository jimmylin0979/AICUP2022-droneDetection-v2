#
import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

#
def brighten(image):
    # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

    # 
    image = image.copy()
    # 
    alpha, beta = 1.3, 40
    gamma = 0.4
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    #             # # 1. correct with alpha & beta
    #             # image[y,x,c] = np.clip(alpha * image[y,x,c] + beta, 0, 255)
    #             # 2. correct with gamma
    #             image[y,x,c] = ((image[y,x,c] / 255) ** gamma) * 255
    image = ((image / 255) ** gamma) * 255
    return image

#
def log_transform(image):
    # Apply log transformation method
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    
    # Specify the data type so that
    # float value will be converted to int
    log_image = np.array(log_image, dtype = np.uint8)
    return log_image

# 
if __name__ == "__main__":

    # 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./Private Testing Dataset_v2/private_tiled", required=True)
    opts = parser.parse_args()

    #
    # root = "./test/public_tiled_brighten"
    # root = "./train_coco_train"
    # root = "./Private Testing Dataset_v2/private_tiled"
    root = opts.root

    for filename in tqdm(os.listdir(f"{root}/data")):
        # 
        if not filename.endswith('.png'):
            continue
        #  
        # filename = f"{root}/data/img0001_0_0.png"
        image = cv2.imread(f"{root}/data/{filename}")
        # cv2.imwrite("original.png", image)
        
        # Test for brighten
        image = brighten(image)
        cv2.imwrite(f"{root}/data/{filename}", image)
    
    # # Test for logt transform
    # image = log_transform(image)
    # cv2.imwrite("processed.png", image)