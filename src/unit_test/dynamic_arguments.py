#!/usr/bin/python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-args_file", default="args.txt")

data = {"f": 0, "g": "cat"}

for k, v in data.items():
    parser.add_argument("--%s" % k, default=None, type=type(v), help="data")

dynamic_args = parser.parse_args()
dynamic_args = vars(dynamic_args)

import pdb

pdb.set_trace()

print(dynamic_args)
