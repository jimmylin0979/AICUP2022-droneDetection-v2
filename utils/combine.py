import pandas as pd
import argparse



def combine(args):
    df1 = pd.read_csv(args.file_1, header = None)
    df2 = pd.read_csv(args.file_2, header = None)
    all = pd.concat([df1,df2],axis=0)
    
    all.to_csv("combine.csv", index = False, header = None)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_1", type = str)
    parser.add_argument("--file_2", type = str)
    args = parser.parse_args()

    combine(args)
