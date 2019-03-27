import os
import cv2

SAVE_NAME = '%d.png'
def flip_merge(input_path1,input_path2):
    print('flip the image and label to increase the dataset')
    dirnames_path1 = os.listdir(input_path1)
    dirnames_path1.remove('tfrecord')
    dirnames_path2 = os.listdir(input_path2)
    dirnames_path2.remove('tfrecord')
    assert(dirnames_path1 == dirnames_path2)
    raw_image_num1 = len(os.listdir(os.path.join(input_path1,dirnames_path1[0])))
    raw_image_num2 = len(os.listdir(os.path.join(input_path2,dirnames_path2[1])))
    filenames = os.listdir(os.path.join(input_path2,dirnames_path2[0]))
    filenames_temp = sorted(os.listdir(os.path.join(input_path1,dirnames_path1[0])))
    name_base = '0'
    for i in filenames_temp:
        temp_base = i.split('.')[0]
        if int(temp_base) >= int(name_base):
            name_base = temp_base

    new_name_base = (int(name_base) + 1)
    for file in filenames:
        for i in range(len(dirnames_path2)):
            read_name = os.path.join(input_path2,dirnames_path2[i],file)
            save_name = os.path.join(input_path1,dirnames_path2[i],SAVE_NAME % new_name_base)
            raw_array = cv2.imread(read_name)
            new_array = cv2.flip(raw_array,1)
            cv2.imwrite(save_name,new_array)
        new_name_base += 1
    new_image_num = len(os.listdir(os.path.join(input_path1,dirnames_path1[-1])))
    assert(new_image_num == raw_image_num1+raw_image_num2)
    os.system('rm -rf %s' % input_path2)    
    print('finish the flip and merge')