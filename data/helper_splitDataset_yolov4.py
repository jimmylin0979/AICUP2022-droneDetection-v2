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
    
    # YOLOv4 : <target> <x-center> <y-center> <width> <height>
    coordinates[1] = (coordinates[1] + coordinates[1] + coordinates[3]) / (2 * w)
    coordinates[2] = (coordinates[2] + coordinates[2] + coordinates[4]) / (2 * h)
    coordinates[3] = coordinates[3] / w
    coordinates[4] = coordinates[4] / h
    coordinates = [str(coordinates[i]) for i in range(len(coordinates))]

    return coordinates

def main(root_dst, output_filename="images.txt"):

    # 
    root_src = f"{root_dst}"
    
    # Output folder
    # Output folder should be in format of YOLOv4Dataset
    root_dst = f"{root_dst}_yolo"
    root_dst_train = f"{root_dst}_train"
    root_dst_valid = f"{root_dst}_valid"

    ##########################################################################
    # Create output folderm throw error if there alread exist a same-name folder 
    #   And also split into train/valid 
    if os.path.isdir(root_dst):
        raise NotImplemented
    else:
        # 
        os.mkdir(f"{root_dst}")
        os.mkdir(f"{root_dst}/data")

        # train/valid 
        os.mkdir(f"{root_dst_train}")
        os.mkdir(f"{root_dst_train}/data")
        os.mkdir(f"{root_dst_valid}")
        os.mkdir(f"{root_dst_valid}/data")

    ##########################################################################
    # images.txt
    imgs_wh = []
    dataset_length = 0
    with open(f"{root_dst}/images.txt", "w") as file:
        for filename in tqdm(sorted(os.listdir(root_src))):
            # 
            if filename.endswith(".png"):
                file.write(f"data/{filename}\n")
                img = cv2.imread(f"{root_src}/{filename}")
                imgs_wh.append([img.shape[0], img.shape[1]])
                dataset_length += 1

    # random split (80/20 split ratio)
    index = [i for i in range(dataset_length)]
    random.Random(4).shuffle(index)
    split_index = int(dataset_length * 0.8)
    index_train = index[: split_index]
    index_valid = index[split_index: ]
    print(f"Train set has {split_index} imgs, valid set has {dataset_length - split_index} imgs")

    # train/ valid
    f_train = open(f"{root_dst_train}/images.txt", "w")
    f_valid = open(f"{root_dst_valid}/images.txt", "w")
    # 
    _index = 0
    for filename in tqdm(sorted(os.listdir(root_src))):
        if filename.endswith(".png"):
            file = f_train if _index in index_train else f_valid
            _index += 1
            file.write(f"data/{filename}\n")
    
    f_train.close()
    f_valid.close()

    ##########################################################################
    # obj.names

    # Generate obj.names file
    classes = []
    with open("classes.txt", "r") as fr:
        for content in fr.readlines():
            content = content.strip("\n ")
            content = content.split(" ")
            classes.append(content[-1])
    with open("obj.names", "w") as fw:
        for clas in classes:
            fw.write(f"{clas}\n")

    # 
    src = f"obj.names"
    dst = f"{root_dst}/obj.names"
    shutil.copy(src, dst)

    dst = f"{root_dst_train}/obj.names"
    shutil.copy(src, dst)
    dst = f"{root_dst_valid}/obj.names"
    shutil.copy(src, dst)

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
            output_dir = root_dst_train if index_ann in index_train else root_dst_valid
            with open(f"{output_dir}/data/{filename}", "w") as fw:
                # 
                with open(f"{root_src}/{filename}", "r") as fr:
                    for content in fr.readlines():
                        loc = corner_to_center(content, imgs_wh[index_ann][0], imgs_wh[index_ann][1])
                        anno = " ".join(loc) + "\n"
                        fw.write(anno)
            index_ann += 1

        elif filename.endswith(".png"):
            # copy paste
            src = f"{root_src}/{filename}"
            output_dir = root_dst_train if index_img in index_train else root_dst_valid
            dst = f"{output_dir}/data/{filename}"
            shutil.copy(src, dst)
            index_img += 1
    



if __name__ == "__main__":
    # 
    parser = argparse.ArgumentParser(description='AICUP2022 - droneDetection')
    parser.add_argument('--root_dst', type=str, required=True,
                                help='The root_dst folder path of dataset')
    args = parser.parse_args()

    # 
    main(args.root_dst)
