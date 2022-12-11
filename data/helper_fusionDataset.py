import os
from tqdm import tqdm

def main():
    
    # 
    # visdrone names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    # trainSet names: ["car", "hov", "person", "motorcycle"]
    fusion_dataset = "./FusionDataset"

    # 
    visdrone_dataset = "./VisDrone"
    cls_mapping = {
        0: 2,       # pedestrian -> person
        1: 2,       # people -> person
        2: 3,       # bicycle -> motorcycle
        3: 0,       # car -> car
        4: 0,       # van -> hov
        5: 1,       # truck -> hov
        6: 3,       # tricycle -> motorcycle
        7: 3,       # awning-tricycle -> motorcycle
        8: 1,       # bus -> hov
        9: 3,       # motor -> motrocycle
    }
    for filename in tqdm(os.listdir(f"{fusion_dataset}/labels/train")):
        # 
        if not filename.endswith('.txt'):
            continue
        
        # Read annotations from visdrone
        contents = []
        with open(f"{fusion_dataset}/labels/train/{filename}", "r") as fr:
            contents = fr.readlines()

        with open(f"{fusion_dataset}/labels/train/{filename}", "w") as fw:
            for content in contents:
                # Expected content of each line
                # 3 0.490104 0.466667 0.046875 0.037037
                content = content.strip(" \n")
                content = content.split(" ")
                cls = float(content[0])
                fusion_cls = cls_mapping[int(cls)]       # find the new class label 
                
                #
                fw.write(str(fusion_cls) + " " + " ".join(content[1:]) + "\n")



if __name__ == "__main__":
    main()