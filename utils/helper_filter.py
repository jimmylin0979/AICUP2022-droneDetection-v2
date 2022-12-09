import pandas as pd
import argparse

def main(args):
    """
    """

    # 
    df = pd.read_csv(args.file, header=None)
    drop_index = []
    # threshold = [0.4, 0.4, 0.30, 0.30]      # Car, Hov, Person, Motor
    threshold = args.threshold     # Car, Hov, Person, Motor
    for i in range(df.shape[0]):

        #         
        clas = df.iloc[i, 1]
        confidence = df.iloc[i, -1]

        # Filter out prediction with different threshold depends on its class
        if confidence < threshold:
                drop_index.append(i)
    
    # Drop the low probability row
    df = df.drop(drop_index)
    # print(df.columns)
    # Also drop the confidence col
    df = df.drop(6, axis=1)

    #
    df.to_csv("./predictions_filter_b.csv", index=False, header=False)

if __name__ == "__main__":

    """
    python3 filter_low_probability.py --file ../deformable_detr_v2/predictions.csv --threshold 0.4
    """

    # 
    parser = argparse.ArgumentParser(description='AICUP2022-droneDetection-v2')
    parser.add_argument('--file', type=str, required=True,
                                help='the file in which are going to filter out low probability bbox')
    parser.add_argument('--threshold', type=float, required=True,
                                help='threshold')
    args = parser.parse_args()

    # 
    main(args)