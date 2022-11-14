import cv2

def main():

    filename_img = "./train/img0001.png"
    filename_ann = "./train/img0001.txt"

    # 
    img = cv2.imread(filename_img)

    # 
    with open(filename_ann, "r") as file:
        for content in file.readlines():
            content = content.replace("\n", "")
            coord = content.split(",")
            print(coord)
            coord = [int(i) for i in coord]
            target, x, y, w, h = coord
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1) 

    #
    cv2.imwrite("helper_bbox.png", img)

if __name__ == "__main__":
    main()