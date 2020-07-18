import sys
import json
import pandas as pd 

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: tojson.py <input_path> <output_path>')
        exit()
    infile = "/home/congyao/DuReader-master/dev_v2.1.json"
    outfile = "/home/congyao/DuReader-master/dev_v2.1_temp1.json"
    df = pd.read_json(infile)
    with open(outfile, 'w') as f:
        for row in df.iterrows():
            f.write(row[1].to_json() + '\n')