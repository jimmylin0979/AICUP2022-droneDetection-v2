import argparse
import os
import shutil
import cv2
from tqdm import tqdm
import random

def corner_to_center(content: str, h, w):
    """
    """

    # coordinate = [target, xmin, ymin, width, height]
    coordinates = content.split(",")
    coordinates = [int(coordinates[i]) for i in range(len(coordinates))]
    
    # the true meaning of each column in YOLOv4
    # YOLOv4 : <target> <x-center> <y-center> <width> <height>
    coordinates[1] = (coordinates[1] + coordinates[1] + coordinates[3]) / (2 * w)
    coordinates[2] = (coordinates[2] + coordinates[2] + coordinates[4]) / (2 * h)
    coordinates[3] = coordinates[3] / w
    coordinates[4] = coordinates[4] / h
    coordinates = [str(coordinates[i]) for i in range(len(coordinates))]

    return coordinates

def main(root_src, root_dst):

    # 
    root_src = f"{root_src}"
    
    # Output folder
    # Output folder should be in format of YOLOv4Dataset
    root_dst = f"{root_dst}"
    root_dst_images = f"{root_dst}/images"
    root_dst_labels = f"{root_dst}/labels"

    ##########################################################################
    # Create output folderm throw error if there alread exist a same-name folder 
    #   And also split into train/valid 
    if os.path.isdir(root_dst):
        raise NotImplemented
    else:
        # 
        os.mkdir(f"{root_dst}")
        os.mkdir(f"{root_dst}/data")
        os.mkdir(f"{root_dst_images}")
        os.mkdir(f"{root_dst_labels}")

        # train/valid 
        os.mkdir(f"{root_dst_images}/train")
        os.mkdir(f"{root_dst_labels}/train")
        os.mkdir(f"{root_dst_images}/valid")
        os.mkdir(f"{root_dst_labels}/valid")

    ##########################################################################
    # images.txt
    imgs_wh = []
    dataset_length = 0
    for filename in tqdm(sorted(os.listdir(root_src))):
        # 
        if filename.endswith(".png"):
            img = cv2.imread(f"{root_src}/{filename}")
            imgs_wh.append([img.shape[0], img.shape[1]])
            dataset_length += 1

    # random split (80/20 split ratio)
    index = [i for i in range(dataset_length)]
    random.Random(4).shuffle(index)     # Add random seed to make process re-producable
    split_index = int(dataset_length * 0.8)
    index_train = index[: split_index]
    index_valid = index[split_index: ]
    print(f"Train set has {split_index} imgs, valid set has {dataset_length - split_index} imgs")

    # # train/ valid
    # # 
    # _index = 0
    # for filename in tqdm(sorted(os.listdir(root_src))):
    #     if filename.endswith(".png"):
    #         output_dir = "train" if _index in index_train else "valid"
    #         src = f"{root_src}/{filename}"
    #         dst = f"{root_dst_labels}/{output_dir}/{filename}"
    #         if _index in index_train:
    #             shutil.copy(src, dst)
    #         _index += 1

    ##########################################################################
    # obj.names
    src = f"obj.names"
    dst = f"{root_dst}/obj.names"
    shutil.copy(src, dst)

    # dst = f"{root_dst_train}/obj.names"
    # shutil.copy(src, dst)
    # dst = f"{root_dst_valid}/obj.names"
    # shutil.copy(src, dst)

    ##########################################################################
    # data/XXXX.png
    i = 0
    for filename in tqdm(sorted(os.listdir(root_src))):
        # if the file is in format of txt, then rewrite the annotation into yolo one
        if filename.endswith(".txt"):
            # 
            with open(f"{root_dst}/data/{filename}", "w") as fw:
                # 
                with open(f"{root_src}/{filename}", "r") as fr:
                    for content in fr.readlines():
                        loc = corner_to_center(content, imgs_wh[i][0], imgs_wh[i][1])
                        anno = " ".join(loc) + "\n"
                        fw.write(anno)
            i += 1

        elif filename.endswith(".png"):
            # copy paste
            src = f"{root_src}/{filename}"
            dst = f"{root_dst}/data/{filename}"
            shutil.copy(src, dst)
    
    # train/ valid
    index_ann, index_img = 0, 0
    for filename in tqdm(sorted(os.listdir(root_src))):
        # if the file is in format of txt, then rewrite the annotation into yolo one
        if filename.endswith(".txt"):
            # 
            output_dir = "train" if index_ann in index_train else "valid"
            src = f"{root_dst}/data/{filename}"
            dst = f"{root_dst_labels}/{output_dir}/{filename}"
            shutil.copy(src, dst)
            index_ann += 1

        elif filename.endswith(".png"):
            # copy paste
            output_dir = "train" if index_ann in index_train else "valid"
            src = f"{root_dst}/data/{filename}"
            dst = f"{root_dst_images}/{output_dir}/{filename}"
            shutil.copy(src, dst)
            index_img += 1
    



if __name__ == "__main__":
    # 
    parser = argparse.ArgumentParser(description='AICUP2022 - droneDetection')
    parser.add_argument('--root_src', type=str, required=True,
                                help='The root_src folder path of dataset')
    parser.add_argument('--root_dst', type=str, required=True,
                                help='The root_dst folder path of dataset')
    args = parser.parse_args()

    # 
    main(args.root_src, args.root_dst)