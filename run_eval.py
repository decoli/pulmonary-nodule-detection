import os
import csv
import argparse

parser = argparse.ArgumentParser(description='Run eval.py')
parser.add_argument('--start_iter', required=True, type=int, help='set the iter of loop startting.')
parser.add_argument('--range', required=True, type=int, help='set the range value.')
args = parser.parse_args()

start_iter = args.start_iter

while True:
    path_weights = '.\weights\ssd_luna16_{start_iter}.pth'.format(
        start_iter=start_iter)
    if not os.path.exists(path_weights):
        break

    command = 'python .\eval.py --trained_model {path_weights}'.format(
        path_weights=path_weights)
    os.system(command)

    start_iter = start_iter + 500

print('eval end.')
