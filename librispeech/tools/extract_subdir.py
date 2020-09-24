import argparse
import os
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('root_dir')
    parser.add_argument('-n', '--num_subdir', type=int)
    args = parser.parse_args()
    filename = args.filename
    root_dir = args.root_dir
    num_subdir = args.num_subdir

    subdir_list = []
    with open(filename) as f:
        for line in f:
            line = os.path.dirname(line.strip())
            line = line.split('/')[num_subdir:]
            path_name = '/'.join(line)
            if path_name not in subdir_list:
                # print(path_name)
                subdir_list.append(path_name)
    
    for path in subdir_list:
        print(os.path.join(root_dir, path))

