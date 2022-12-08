# 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging

# 
class Bbox(object):
    """
        A class define the bounding box
    """
    def __init__(self, bbox) -> None:
        """
        Args:
            bbox (List[int]): a list of int that represents a bounding box instance 

        Return:
        """

        # origin (0, 0) is at the left-up corner in the original image 
        # A bounding box can be formated as
        #       1. [x1, y1,  w,  h]  => top-left corner + width & height 
        #       2. [x1, y1, x2, y2]  => top-left corner + bottom-right corner

        # unpack the bbox, and re-define the attributes in a easy-reading format
        # every element in bbox should be in format of int
        self.target = int(bbox[0])
        self.x1 = int(bbox[1])
        self.y1 = int(bbox[2])
        self.w = int(bbox[3])
        self.h = int(bbox[4])
        if len(bbox) >= 6:
            self.confidence = float(bbox[5])
        else:
            self.confidence = 1
        # 
        self.x2 = self.x1 + self.w
        self.y2 = self.y1 + self.h

    ###########################################################################################    
    def tolist(self):
        """
        Return Box as a list of int 
        """
        if self.confidence is not None:
            return [self.target, self.x1, self.y1, self.w, self.h, self.confidence]
        return [self.target, self.x1, self.y1, self.w, self.h]

    def toarray(self):
        """
        Return Box as a numpy.array of int 
        """
        if self.confidence is not None:
            return np.array([self.target, self.x1, self.y1, self.w, self.h, self.confidence])
        return np.array([self.target, self.x1, self.y1, self.w, self.h])
    
    def getArea(self):
        """
        Return the area of bounding box
        """
        return self.w * self.h
    
    ###########################################################################################
    # 
    def __str__ (self):
        return f"[({self.x1}, {self.y1}, {self.w}, {self.h}), {self.target}, {self.confidence}]\n"
    
    def __repr__ (self):
        # return f"Bbox ({self.x1}, {self.y1}, {self.w}, {self.h}), target {self.target}, {self.confidence}"
        return f"({self.x1}, {self.y1}, {self.w}, {self.h}), {self.target}, {self.confidence})\n"

###########################################################################################
# Utilization 

def isOverlap(bbox1, bbox2):
    """
    Check whether 2 bounding boxes are overlapping 

    Args:
        bbox1 (Box): bounding box 1, format [target, x_center, y_center, width, height]
        bbox2 (Box): bounding box 2

    Returns:
        bool: True if two bounding boxes are overlapping 
    """

    # 1. Check if any bounding box is empty area (Won't happen)
    # 2. Check if any bounding box is on the left side or right side of the other one
    if bbox1.x1 > bbox2.x2 or bbox2.x1 > bbox1.x2:
        return False
    # 3. Check if any bounding box is on the above of the other 
    if bbox1.y1 > bbox2.y2 or bbox2.y1 > bbox1.y2:
        return False

    return True

def getOverlapPoint(bbox1, bbox2):
    """
    Return the overlap point of 2 bounding boxes, return None if 2 arn't overlapping 

    Args:
        bbox1 (src.Box):
        bbox2 (src.Box):

    Returns:
        (List): a list represents the overlapping area, [x, y, w, h] (x, y are top-left corner of the area)
    """

    # Return None if 2 boxes are not overlapping 
    if not isOverlap(bbox1, bbox2):
        return None

    # 
    min_x = max(bbox1.x1, bbox2.x1)
    max_x = min(bbox1.x2, bbox2.x2)
    min_y = max(bbox1.y1, bbox2.y1)
    max_y = min(bbox1.y2, bbox2.y2)

    # return [min_x, min_y, max_x, max_y]
    return Bbox(['-1', min_x, min_y, max_x - min_x, max_y - min_y])

def visualize(img, anns):
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
        confidence = 0
        if isinstance(ann, Bbox):            
            if len(ann.tolist()) == 5:
                target, x, y, w, h = ann.tolist()
            else:
                target, x, y, w, h, confidence = ann.tolist()
        else:
            if len(ann) == 5:
                target, x, y, w, h = ann
            else:
                target, x, y, w, h, confidence = ann

        color = (0, 0, 0)
        if target == 1:
            color = (255, 0, 0)
        elif target == 2:
            color = (0, 255, 0)
        elif target == 3:
            color = (255, 255, 255)
        # print((int(x), int(y)), (int(x + w), int(y + h)))
        # if confidence > 0.55:
        cv2.rectangle(img_demo, (int(x), int(y)), (int(x + w), int(y + h)), color, 1) 
    #
    # cv2.imwrite("visualize", img_demo)
    # cv2.imshow("visualize", img_demo)
    # cv2.waitKey(-1)
    plt.figure(figsize = (20, 18))
    plt.imshow(img_demo)
    plt.show()

def NMS(bboxes, threshold = 0.50, img=None, imgName=""):
    """
    Perform No Maximal Suppression on input bboxes
    
    Args:
        bboxes (List):
        threshold (float): 
    
    Returns:
        res (List[Box]):
    """

    # 
    if len(bboxes) == 0:
        return []
    
    # Sort the bounding box by their top-left corner in descending order 
    # bboxes = sorted(bboxes, key=lambda bbox: (-bbox.getArea()))
    bboxes = sorted(bboxes, key=lambda bbox: (-bbox.confidence))

    # if img is not None:
    #     if imgName == "img1007":
    #         visualize(img, bboxes)

    res = []
    while len(bboxes) > 0:

        #
        anchor_bbox = bboxes[0]
        anchor_area = anchor_bbox.getArea()
        res.append(anchor_bbox)
        bboxes.remove(anchor_bbox)

        # NMS Algo 
        removed_boxes = []
        for bbox in bboxes:

            # If two bounding boxes have different target, continue
            # # TODO
            if anchor_bbox.target == 0:
                if bbox.target not in [0, 1]:
                    continue
            elif anchor_bbox.target == 1:
                if bbox.target not in [0, 1]:
                    continue
            else:
                if bbox.target != anchor_bbox.target:
                    continue
            
            # if bbox.target != anchor_bbox.target:
            #     continue
            
            # Calculate the overlapping area of anchor_box & bbox
            overlap_bbox = getOverlapPoint(anchor_bbox, bbox)
            if overlap_bbox is None:
                continue
            overlap_area = overlap_bbox.getArea()
            union_area = overlap_bbox.getArea() + anchor_area - overlap_area
            # Self-defined IoU
            overlap_ratio = max(overlap_area / bbox.getArea(), overlap_area / anchor_area)
            
            # # IoU
            # overlap_ratio = overlap_area / (union_area)

            # if bboxes[j] overlaps with anchor_bbox more than certain area
            if overlap_ratio > threshold:
                removed_boxes.append(bbox)

            # # If the overlap area is equal to bbox, 
            # #   it means that bbox is covered by anchor_area, then we should just remove it 
            # #   Since there should be some other tiled image be responble for recognizing that object  
            # elif overlap_area >= bbox.getArea():
            #     removed_boxes.append(bbox)

        # 
        # logging.debug(removed_boxes)
        for bbox in removed_boxes:
            bboxes.remove(bbox)
        
    # # 
    # if img is not None:
    #     if imgName == "img1007":
    #         visualize(img, res)

    return res

###########################################################################################
# Testing Area
if __name__ == "__main__":
    # 
    box = Bbox(['1', 0, 0, 512, 512])
    logging.debug(box)