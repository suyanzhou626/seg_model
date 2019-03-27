import os
import numpy as np
from PIL import Image as img
import math
import random

def segmentation(dir_seg,mode):
    print('choose the trainset and valset')
    read_path = os.path.join(dir_seg,'labelraw')
    seg_path = os.path.join(dir_seg,'segmentation')
    if not os.path.exists(seg_path):
        os.mkdir(seg_path)
    information_path = os.path.join(dir_seg,'information.txt')

    files = os.listdir(read_path)
        
    fi = open(information_path,'w')
    selectset = []
    num_havenodamage = 0
    num_havelargesize = 0
    for key in files:
        filename = os.path.join(read_path,key)
        label = np.array(img.open(filename))
        if mode[0] == 'd':
            if np.max(label)==1 or (np.max(label)==255 and len(np.unique(label))>=3):
                if label.shape[0]<=600 and label.shape[1]<=600:
                    selectset.append(key)
                else:
                    num_havelargesize +=1
            else:
                num_havenodamage += 1
        else:
            if 1<=np.max(label)<=25 or (np.max(label)==255 and len(np.unique(label))>=3):
                if label.shape[0]<=600 and label.shape[1]<=600:
                    selectset.append(key)
                else:
                    num_havelargesize +=1
            else:
                num_havenodamage += 1
    print('         the num of have large size picture in %s: ' % mode,num_havelargesize)
    print('         the num of have no damage in %s: ' % mode,num_havenodamage)
    print('         the num of damage label in %s: ' % mode,len(selectset))
    x = len(selectset)
#    fi.writelines('the num of have large size picture in %s: ' % mode+str(num_havelargesize)+'\n')
    fi.writelines('the num of damage label in %s: ' % mode+str(x)+'\n')
    y = math.ceil(x*0.7)
    fi.writelines('the num of train label in %s: ' % mode + str(y) + '\n')
    print('         the num of train label in %s: ' % mode + str(y))
    fi.writelines('the num of val label in %s: ' % mode + str(x-y) + '\n')
    print('         the num of val label in %s: ' % mode + str(x-y))
    trainset = selectset[0:y]
    valset = set(selectset) - set(trainset)
    assert((len(valset)+len(trainset)) == x)
    train_path = os.path.join(seg_path,'train.txt')
    val_path = os.path.join(seg_path,'val.txt')
    f1 = open(train_path,'w')
    f2 = open(val_path,'w')
    for i in trainset:
        f1.writelines('image/' + i + ' ' + 'labelraw/' + i + '\n')
    for j in valset:
        f2.writelines('image/' + j + ' ' + 'labelraw/' + j + '\n')
    f1.close()
    f2.close()
    fi.close()