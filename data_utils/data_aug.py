import Augmentor
import os
import time
import cv2
import numpy as np

def data_aug(data_path,aug_times,need_zoom):
    print('aug data')
    if need_zoom:
        print('     apply zoom operation!')
    else:
        print('     dont execute zoom!')
    batch_num = aug_times
    train_path = os.path.join(data_path,'segmentation','train.txt')
    f1 = open(train_path,'r')
    image_files = [i.strip('\n') for i in f1]
    f1.close()

    count = 1
    start_time = time.time()
    for file in image_files:
        stop_time = time.time()
        label_map = {}
        if count % 100 == 0:
            print('     augment the data, %d/%d, aug operation cost time is: ' % (count,len(image_files)),stop_time-start_time)
            start_time = stop_time
        count += 1
        image_path = os.path.join(data_path,'image',file+'.png')
        label_path = image_path.replace('image','labelraw')
        image_save_path = []
        label_save_path = []
        for k in range(batch_num):
            image_save_path.append(os.path.join(data_path,'image',file+'_aug'+str(k+1)+'.png'))
            label_save_path.append(os.path.join(data_path,'labelraw',file+'_aug'+str(k+1)+'.png'))
        image_array = cv2.imread(image_path)
        raw_array = [image_array]
        label_array = cv2.imread(label_path,0)
        label_key = np.unique(label_array).tolist()
        for i in range(len(label_key)):
            label_map[i] = label_key[i]
            label_array[label_array==label_key[i]] = i
        raw_array.append(label_array)
            
        p = Augmentor.DataPipeline([raw_array])
        p.skew(0.5,magnitude=0.5)
        p.random_distortion(0.5,4,4,5)
        p.rotate(0.5,20,20)
        p.shear(0.5,10,10)
        if need_zoom:
            p.zoom(1,1.2,1.6)
        new_array = p.sample(batch_num)
        num_new_sample = len(new_array)
#        num_new = len(new_array[0])
        f2 = open(train_path,'a+')
        for m in range(num_new_sample):
            f2.writelines(image_save_path[m].split('.')[0].split('/')[-1]+'\n')
            save_temp_array = new_array[m]
            for j in range(len(label_key)):
                save_temp_array[save_temp_array==j] = label_map[j]
            cv2.imwrite(image_save_path[m],save_temp_array[0])
            cv2.imwrite(label_save_path[m],save_temp_array[1])
        f2.close()