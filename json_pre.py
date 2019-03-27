from data_utils.json2dir import json2dir as json2dir
from data_utils.data_aug import data_aug as data_aug
from data_utils.segmentation import segmentation as segmentation
from data_utils.gtcolormap import gtcolormap as gtcolormap
from data_utils.flip_merge import flip_merge as flip_merge
import argparse
import math

parser = argparse.ArgumentParser(description = 'Process json to png')
parser.add_argument('--read_path',type=str,help='set the dirname to be processed in rawdata_dir',nargs='?',default=None)
parser.add_argument('--out_path',type=str,help='set the dirname to save trainset and valset')
parser.add_argument('--aug_times',type=int,nargs='?',default=0,help='set the factor os data augment')
parser.add_argument('--mode',type=str,help='set how to produce the label: leftfront,rightbehind ..',
                        choices=['rightfront','rightbehind','behind','right','front','d01','d02','d05'])
parser.add_argument('--need_zoom',type=str,nargs='?',default='no')
parser.add_argument('--extra_path',type=str,nargs='?',help='set the dataset to be merged to read_path',default=None)

args = parser.parse_args()

dir_json2dir_in = args.read_path
aug_times = int(args.aug_times)
mode = args.mode
extra_dir_in = args.extra_path
dir_seg_in = args.out_path #+'_'+mode+'_'+str(aug_times)
need_zoom = args.need_zoom
if dir_json2dir_in == None:
    print('do not process json')
    dir_seg_in = args.out_path
else:
    json2dir(dir_json2dir_in,dir_seg_in,mode)
    if extra_dir_in == None:
        print('do not merge mutil dataset')
    else:
        temp_seg_in = dir_seg_in+'_temp'
        mode_temp = mode.replace('right','left')
        json2dir(extra_dir_in,temp_seg_in,mode_temp)
        flip_merge(dir_seg_in,temp_seg_in)
segmentation(dir_seg_in,mode)

if aug_times >0:
    print('augmentation has some error temporarily')
    data_aug(dir_seg_in,aug_times,need_zoom)
else:
    print('do not augment the data')