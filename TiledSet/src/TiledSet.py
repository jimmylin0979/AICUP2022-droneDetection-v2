#
import numpy as np
import cv2
import os
import shutil

# 
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# 
from src import Bbox
from src.Bbox import getOverlapPoint, isOverlap, NMS

# 
support_img_type = ["jpg", "png", "jpeg"]
support_ann_type = ["txt"]

class TiledSet(object):
    """
        This class defines the tiling code for dealin with input dataset (YoloV4Dataset)
    """

    def __init__(self, 
                 tileSize, 
                 stride,
                 *args, **kwargs) -> None:
        
        # 
        super(TiledSet, self).__init__()

        #
        self.tileSize = [int(i) for i in tileSize]          # [tileSize_wdith, tileSize_height]
        self.stride = [int(i) for i in stride]              # [stride_wdith, stride_height]

    ########################################################################
    # private

    def _find_anno(self, anns, tl_coord, bd_coord):
        """
        Find the bounding boxes that is located in the region (r: r+stride, c: c+stride), 
            which will be the new annotations of the tiled image
        
        Algo:
        1. (Brute-Force) Simply loop through all the bounding box in the whole image
            1.1 If the current bounding box is overlapping with window, 
                    => add the bounding box into the res, assign it as the ground truth of that tiled image

        Args:
            anns (List[Tuple]): the annotations (bounding boxes) in the whole images
            r (int): the left-up corner'x of the windows
            c (int): the left-up corner'y of the windows
        
        Returns:
            anns (List[Tuple]): the annotations (bounding boxes) that is located in the given area
        """

        # tl stands for top-left, while bd stands for bottom right
        r, c = tl_coord
        r_end, c_end = bd_coord
        h, w = r_end - r, c_end - c

        # 
        res = []
        for ann in anns:

            # window can also be seens as a bounding box
            bbox = Bbox(ann)
            window_coord = [0, c, r, w, h]
            window = Bbox(window_coord)

            # get overlap point of bbox & window
            # logging.debug("Input", ann, window_coord)
            if not isOverlap(bbox, window):
                continue

            overlap = getOverlapPoint(bbox, window)
            if overlap is None:
                continue

            # Condition 1. Two bounding boxes are overlapping, 
            # Condition 2. The overlapping area's w & h should larger than certain threshold
            # then put the overlap into the annotations of that window

            # Condition 2: Find a good ratio to filter out too small bounding boxes 
            # Trying to set higher threshold, making the detection focus more on the responsibile object 
            # v1 : 0.33, 0.33 
            # v2 : 0.50, 0.50 -> let model to be less sensitive to object boundary
            threshold_w, threshold_h = 0.33, 0.33
            if overlap.getArea() <= 0:
                continue
            if (overlap.w / bbox.w) < threshold_w or (overlap.h / bbox.h) < threshold_h:
                continue

            # Relative coordinate 
            overlap = Bbox([bbox.target, overlap.x1 - c, overlap.y1 - r, overlap.w, overlap.h])
            res.append(overlap.tolist())

        return res
    
    def isImage(self, filename):
        """
        """
        # 
        idx = filename.rfind(".")                           # Seperate the filename into filetype, filename 
        filetype = filename[idx + 1:]
        return filetype in support_img_type
    
    ########################################################################
    # public

    def visualize(self, img, anns):
        """
        Visulaize the tiled image with tiled annotations
        
        Args:
            img (cv2.):
            anns (List):
        """
        # 
        # img_demo = img.clone()
        img_demo = np.copy(img)
        for ann in anns:
            
            # 
            if isinstance(ann, Bbox):            
                if len(ann.tolist()) == 5:
                    target, x, y, w, h = ann.tolist()
                else:
                    target, x, y, w, h, _ = ann.tolist()
            else:
                if len(ann) == 5:
                    target, x, y, w, h = ann
                else:
                    target, x, y, w, h, _ = ann

            color = (0, 0, 255)
            if target == 1:
                color = (255, 0, 0)
            elif target == 2:
                color = (0, 255, 0)
            elif target == 3:
                color = (255, 255, 255)
            # print((int(x), int(y)), (int(x + w), int(y + h)))
            cv2.rectangle(img_demo, (int(x), int(y)), (int(x + w), int(y + h)), color, 2) 
        #
        # cv2.imwrite("visualize", img_demo)
        # cv2.imshow("visualize", img_demo)
        # cv2.waitKey(-1)
        plt.figure(figsize = (20, 18))
        plt.imshow(img_demo)
        plt.show()

    def tile(self, 
             root_src: str, 
             root_dst: str,
             with_annotations = True,
             *args, **kwargs):
        """
        Tile the image from root_src (should follow YOLOv4Dataset format), 
            including images and annotations. 

        Args:
            root_src (str):
            root_dst (str):

        Returns:
        """
        # Create root_dst dataset folder 
        # Assert error when there already exists a folder with same name  
        os.makedirs(root_dst, exist_ok=False)
        if with_annotations:
            os.makedirs(f"{root_dst}/images/train")
            os.makedirs(f"{root_dst}/images/valid")
            os.makedirs(f"{root_dst}/labels/train")
            os.makedirs(f"{root_dst}/labels/valid")
        else:
            os.mkdir(f"{root_dst}/data")

        # 
        images_listdir = []
        if with_annotations:
            # Copy obj.names file
            src = f"{root_src}/obj.names"
            dst = f"{root_dst}/obj.names"
            shutil.copyfile(src, dst)

            # Get images from images.txt
            images_listdir = []
            # with open(f"{root_src}/images.txt", "r") as fr:
            #     contents = fr.readlines()
            #     for content in contents:
            #         content = content.strip("\n ")
            #         # content = content[5:]
            #         images_listdir.append(content)
            
            # 
            for filename in os.listdir(f"{root_src}/images/train"):
                if filename.endswith(".png"):
                    images_listdir.append(f"train/{filename}")
            for filename in os.listdir(f"{root_src}/images/valid"):
                if filename.endswith(".png"):
                    images_listdir.append(f"valid/{filename}")
            
        else:
            images_listdir = sorted(os.listdir(f"{root_src}"))
        
        # Tiled image and write it into root_dst/data fodler
        for filename in tqdm(images_listdir):
            # 
            idx = filename.rfind(".")                           # Seperate the filename into filetype, filename 
            filetype = filename[idx + 1:]
            
            # If filename is in supported type of image  
            if self.isImage(filename):
                # Open image 
                img = None
                if not with_annotations:
                    img = cv2.imread(f"{root_src}/{filename}")
                else:
                    img = cv2.imread(f"{root_src}/images/{filename}")     # 
                h, w = img.shape[0], img.shape[1]                   # shape (Height, Width, Channels) (1080. 1920, 3)

                # Open annotation files corresponding to images 
                # After porcessing, 
                filename = filename[:idx]
                anns = []
                # if os.path.exists(f"{root_src}/data/{filename}.txt"):
                if with_annotations:
                    with open(f"{root_src}/labels/{filename}.txt", "r") as fr:
                        contents = fr.readlines()
                        for content in contents:
                            # Clear
                            content = content.strip("\n ,")
                            content = content.split(" ")
                            
                            # Transfer the (x_center, y_center, w, h)(yolo style) into (x_corner, y_corner, w, h)(COCO style)
                            multis = [w, h, w, h]
                            for i in range(1, len(content)):
                                content[i] = int(float(content[i]) * multis[i-1])
                            content[1] -= content[3] // 2
                            content[2] -= content[4] // 2
                            
                            anns.append(content)

                # # Visualize
                # self.visualize(img, anns)
                # break

                # r, c is the top-left coordinate of current windows 
                # (rowCount, colCount) 
                r, c = 0, 0
                rowCount, colCount = 0, 0

                # Loop from left to right, up to down 
                while r < h:
                    while c < w:
                        # Image & Annotation
                        r_end = r + self.tileSize[0] if r + self.tileSize[0] < h else h
                        c_end = c + self.tileSize[1] if c + self.tileSize[1] < w else w

                        # logging.debug(f"INFO: ")
                        # logging.debug(f"tiled_img.top-left coord {(r, c)} -> {(r_end, c_end)}")

                        tiled_img = img[r: r_end, c: c_end]
                        if with_annotations:
                            tiled_ann = self._find_anno(anns, (r, c), (r_end, c_end))

                        # # Logging
                        # logging.debug(f"tiled_img.shape : {tiled_img.shape}")
                        # logging.debug(f"tiled_ann : {tiled_ann}")
                        # logging.debug("=" * 80)

                        # # Visualization
                        # # Should comment after checking the code is running well 
                        # self.visualize(tiled_img, tiled_ann)

                        # Flush the tiled information back into disk 
                        # Image         will be save with name {original name}_{rowCount}_{colCount}.png
                        # Annotations   will be save with name {original name}_{rowCount}_{colCount}.txt
                        output_img_name = f"{root_dst}/data/{filename}_{rowCount}_{colCount}"
                        if with_annotations:
                            output_img_name = f"{root_dst}/images/{filename}_{rowCount}_{colCount}"

                        cv2.imwrite(f"{output_img_name}.{filetype}", tiled_img)
                        if with_annotations:
                            output_ann_name = f"{root_dst}/labels/{filename}_{rowCount}_{colCount}"
                            with open(f"{output_ann_name}.txt", "w") as fw:
                                for ann in tiled_ann:
                                    # Transfer annotation format back into (x_center, y_center, width, height)
                                    ann[1] = (ann[1] + ann[3] / 2) / (c_end - c)
                                    ann[2] = (ann[2] + ann[4] / 2) / (r_end - r)
                                    ann[3] /= (c_end - c)
                                    ann[4] /= (r_end - r)
                                    ann = [str(i) for i in ann]
                                    ann = " ".join(ann)
                                    fw.write(f"{ann}\n")

                        # Step & Terminal Case of column (width)
                        # Once the tiled image is already reach the boundary (weight of the input image), 
                        #   stop the inner loop
                        if c + self.tileSize[1] >= w:
                            c = 0
                            colCount = 0
                            break
                        else:
                            # If still not reach the boundary, then move right the tiled window by stride
                            c += self.stride[1]
                            colCount += 1

                    # Step & Terminal Case of row (height)
                    # Once the tiled image is already reach the boundary (height of the input image), 
                    #   stop the whole loop
                    if r + self.tileSize[0] >= h:
                        rowCount = 0
                        break
                    else:
                        # If still not reach the boundary, then move down the tiled window by stride
                        r += self.stride[0]
                        rowCount += 1
        
        # Write images.txt file
        if not with_annotations:
            with open(f"{root_dst}/images.txt", "w") as fw:
                for filename in sorted(os.listdir(f"{root_dst}/data")):
                    if self.isImage(filename):
                        fw.write(f"data/{filename}\n")
    
    def merge(self, prediction_path, root_originalImage=None, sep=" "):
        """
        Merge the annotations of each tiled image into one merged annotations

        Args:
            root_src (str):
        
        Returns:
        """

        # 
        i = 0
        contents = []
        with open(prediction_path, "r") as fr:
            contents = fr.readlines()
            
        # 
        fw = open("predictions.csv", "w")
        while i < len(contents):    
            
            # Get the current original image name
            # content: img1001_0_0,1,937,382,23,60
            content = contents[i]
            content = content.strip("\n ")
            content = content.split(",")
            current_originalName = content[0].split("_")[0]

            # the overal annotations that should store all annoations from tiled image that share the same mother image
            overall_anns = []
            # Loop through the tiled annotation within the same original image
            while i < len(contents):

                # Content clearing
                # content: img1001_0_0,1,937,382,23,60
                content = contents[i]
                content = content.strip("\n ")
                content = content.split(",")

                # 
                # img1001_0_0 => img1001, 0, 0
                originalName, rowCount, colCount = content[0].split("_")
                if originalName != current_originalName:
                    break
                rowCount = int(rowCount)
                colCount = int(colCount)                

                # 
                for j in range(2, 2 + 4):        # Data type transfering
                    content[j] = int(content[j])
                # content[1] -= content[3] // 2
                # content[2] -= content[4] // 2
                
                # Add offset depends on rowCount & colCount
                # Remember that the coordinate on each annotation is treated their tiled image's top-left corner as origin point
                # So every annotation files did not share the same origin
                content[2] += colCount * (self.stride[1])
                content[3] += rowCount * (self.stride[0])
                bbox = Bbox(content[1:])
                overall_anns.append(bbox)

                # step
                i += 1

            # # 
            # # Visualize (before merging)
            img = cv2.imread(f'{root_originalImage}/{current_originalName}.png')
            # self.visualize(img, overall_anns)
            
            # # Deal with overlapping bounding boxes with NMS
            # logging.debug("=" * 80)
            logging.debug(f"Before filter, nums of bbox: {len(overall_anns)}")
            # logging.debug(f"{current_originalName}")
            overall_anns = NMS(overall_anns, threshold=0.5, img=img, imgName = current_originalName)
            logging.debug(f"After filter, nums of bbox: {len(overall_anns)}")

            # # Visualize (after merging)
            # img = cv2.imread(f'{root_originalImage}/{current_originalName}.png')
            # self.visualize(img, overall_anns)

            # if current_originalName == "img1007":
            #     break

            # Write annotations back into output.csv 
            for bbox in overall_anns:
                content = bbox.tolist()
                content = [str(i) for i in content]
                content = sep.join(content)
                content = f"{current_originalName}{sep}{content}"
                fw.write(f"{content}\n")
            # fw.flush()

        fw.close()
        return None

    def ensemble(self, prediction_path, root_originalImage=None, sep=" "):
        """
        Merge the annotations of each tiled image into one merged annotations

        Args:
            root_src (str):
        
        Returns:
        """

        # 
        i = 0
        contents = []
        with open(prediction_path, "r") as fr:
            contents = fr.readlines()
        contents = sorted(contents)

        # 
        fw = open("predictions.csv", "w")
        while i < len(contents):    
            
            # Get the current original image name
            # content: img1001_0_0,1,937,382,23,60
            content = contents[i]
            content = content.strip("\n ")
            content = content.split(",")
            current_originalName = content[0].split("_")[0]

            # the overal annotations that should store all annoations from tiled image that share the same mother image
            overall_anns = []
            # Loop through the tiled annotation within the same original image
            while i < len(contents):

                # Content clearing
                # content: img1001_0_0,1,937,382,23,60
                content = contents[i]
                content = content.strip("\n ")
                content = content.split(",")   

                originalName = content[0]
                if originalName != current_originalName:
                    break             

                # 
                for j in range(2, 2 + 4):        # Data type transfering
                    content[j] = int(content[j])
                # content[1] -= content[3] // 2
                # content[2] -= content[4] // 2
                
                # Add offset depends on rowCount & colCount
                # Remember that the coordinate on each annotation is treated their tiled image's top-left corner as origin point
                # So every annotation files did not share the same origin
                bbox = Bbox(content[1:])
                overall_anns.append(bbox)

                # step
                i += 1

            # # 
            # # Visualize (before merging)
            # img = cv2.imread(f'{root_originalImage}/{current_originalName}.png')
            # self.visualize(img, overall_anns)
            
            # # Deal with overlapping bounding boxes with NMS
            # logging.debug("=" * 80)
            logging.debug(f"Before filter, nums of bbox: {len(overall_anns)}")
            # logging.debug(f"{current_originalName}")
            overall_anns = NMS(overall_anns, threshold=0.5, img=None, imgName = current_originalName)
            logging.debug(f"After filter, nums of bbox: {len(overall_anns)}")

            # # Visualize (after merging)
            # img = cv2.imread(f'{root_originalImage}/{current_originalName}.png')
            # self.visualize(img, overall_anns)

            # if current_originalName == "img1007":
            #     break

            # Write annotations back into output.csv 
            for bbox in overall_anns:
                content = bbox.tolist()
                content = [str(i) for i in content]
                content = sep.join(content)
                content = f"{current_originalName}{sep}{content}"
                fw.write(f"{content}\n")
            # fw.flush()

        fw.close()
        return None