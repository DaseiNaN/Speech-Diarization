#!/usr/bin/env python3

# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

import argparse
import sys

import torch

sys.path.append("/home/dasein/Projects/Speech-Diarization")


def average_model_pytorch(ifiles, ofile):
    omodel = {}
    for path in ifiles:
        print(path)
        state_dict = torch.load(path)["state_dict"]
        for key in state_dict.keys():
            val = state_dict[key]
            key = key.replace("net.", "")
            if key not in omodel:
                omodel[key] = val
            else:
                omodel[key] += val
    for key in omodel.keys():
        omodel[key] /= len(ifiles)
    torch.save(dict(model=omodel), ofile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ofile")
    parser.add_argument("ifiles", nargs="+")
    args = parser.parse_args()
    average_model_pytorch(args.ifiles, args.ofile)
