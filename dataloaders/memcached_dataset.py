import mc
from torch.utils.data import Dataset
import numpy as np
from . import custom_transforms as tr
from torchvision import transforms
import io
from PIL import Image
import os
import cv2

import linklink as link

def pil_loader(img_str,bgr_mode=False,gray_mode=False):
    buff = io.BytesIO(img_str)
    
    img = Image.open(buff)
    img = img.convert('RGB')
    img = np.array(img).astype(dtype=np.float32)
    if bgr_mode:
        img = img[:,:,::-1]  #convert to BGR
    elif gray_mode:
        img = img[:,:,::-1]
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = np.dstack([img,img,img])
    return img

def pil_loader_label(img_str,bgr_mode=False):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    if (img.mode != 'L' and img.mode != 'P'):
            temp = np.unique(np.array(img))
            if np.max(temp)<15:
                img = img.convert('L')
            else:
                raise 'error'
    # img = img.convert('L')
    img = np.array(img).astype(dtype=np.float32)
    return img
 
class McDataset(Dataset):
    def __init__(self, args,meta_file,split='train'):
        self.args = args
        self.root_dir = args.data_dir
        self.rank = link.get_rank()
        self.split = split
        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        if self.rank == 0:
            print("building dataset from %s, num of images: %d" % (meta_file,self.num))
        self.metas = []
        for line in lines:
            img_path, label_path = line.rstrip().split()
            self.metas.append((img_path, label_path))
        self.initialized = False
 
    def __len__(self):
        return self.num

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True
 
    def __getitem__(self, idx):
        image_filename = os.path.join(self.root_dir,self.metas[idx][0])
        label_filename = os.path.join(self.root_dir,self.metas[idx][1])
        ## memcached
        self._init_memcached()
        image_value = mc.pyvector()
        label_value = mc.pyvector()
        self.mclient.Get(image_filename, image_value)
        self.mclient.Get(label_filename, label_value)
        image_value_str = mc.ConvertBuffer(image_value)
        label_value_str = mc.ConvertBuffer(label_value)
        img = pil_loader(image_value_str,bgr_mode=self.args.bgr_mode,gray_mode=self.args.gray_mode)
        label = pil_loader_label(label_value_str,bgr_mode=self.args.bgr_mode)
        sample = {'image':img,'label':label}
        if self.split == 'train':
            sample = self.transform_tr(sample)
        else:
            sample = self.transform_val(sample)

        return sample

    def transform_tr(self, sample):
        temp = []
        if self.args.rotate > 0:
            temp.append(tr.RandomRotate(self.args.rotate))
        temp.append(tr.RandomScale(rand_resize=self.args.rand_resize))
        temp.append(tr.RandomCrop(self.args.input_size))
        temp.append(tr.RandomHorizontalFlip())
        temp.append(tr.Normalize(mean=self.args.normal_mean,std=self.args.normal_std))
        if self.args.noise_param is not None:
            temp.append(tr.GaussianNoise(mean=self.args.noise_param[0],std=self.args.noise_param[1]))
        temp.append(tr.ToTensor())
        composed_transforms = transforms.Compose(temp)

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(self.args.test_size,shrink=self.args.shrink),
            tr.Normalize(mean=self.args.normal_mean,std=self.args.normal_std),
            tr.ToTensor()])

        return composed_transforms(sample)
