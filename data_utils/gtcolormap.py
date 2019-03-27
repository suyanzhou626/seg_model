import os
import cv2

def _addImage(img1_path,img2_path):
    img1 = cv2.imread(img1_path)
    img = cv2.imread(img2_path)
    h,w,_ = img1.shape
    #input('make sure the shape of img: {} {}'.format(str(h),str(w)))
    img2 = cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
    alpha = 1
    beta = 0.5
    gamma = 0
    img_add = cv2.addWeighted(img1,alpha,img2,beta,gamma)
    return img_add

def gtcolormap(dir_in):
    print('process the ground_truth to add color')
    read_path = os.path.join(dir_in,'labelclass')
    out_path = os.path.join(dir_in,'colormap')
    image_path = os.path.join(dir_in,'image')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    filenames = os.listdir(read_path)
#    input('make sure the num of gt: '+str(len(filenames))+str(filenames))
    for filens in filenames:
        base = filens.split('.')[0]
        image_name = os.path.join(image_path,base+'.png')
        label_name = os.path.join(read_path,filens)
        out_name = os.path.join(out_path,filens)
        out_image = _addImage(image_name,label_name)
        cv2.imwrite(out_name,out_image)