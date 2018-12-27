import mc
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dataloaders import custom_transforms as tr
from torchvision import transforms
import io
from PIL import Image
import os

import linklink as link

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    
    img = Image.open(buff)
    img = img.convert('RGB')
    return img

def pil_loader_label(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    assert(img.mode=='L' or img.mode=='P')
    # img = img.convert('L')
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
        img = pil_loader(image_value_str)
        label = pil_loader_label(label_value_str)
        sample = {'image':img,'label':label}
        if self.split == 'train':
            sample = self.transform_tr(sample)
        else:
            sample = self.transform_val(sample)

        return sample

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomScaleCrop(crop_size=self.args.crop_size),
            tr.RandomHorizontalFlip(),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=self.args.normal_mean),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=self.args.normal_mean),
            tr.ToTensor()])

        return composed_transforms(sample)
