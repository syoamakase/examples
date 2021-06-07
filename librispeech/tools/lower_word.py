import argparse
import os
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    filename = args.filename

    with open(filename) as f:
        for line in f:
            file_name, line = line.strip().split(' ', 1)
            print(f'{file_name} {line.lower()}')
