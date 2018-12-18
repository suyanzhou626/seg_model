import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GenDataset(Dataset):
    def __init__(self,args,data_list,split='train'):
        super().__init__()
        self._base_dir = args.data_dir
        self._data_list = data_list
        self.args = args
        self.split = split
        self.images = []
        self.categories = []
        with open(self._data_list,'r') as f:
            lines = f.readlines()
        temp_discard = 0
        temp_all = 0
        for ii, line in enumerate(lines):
            _image = os.path.join(self._base_dir, line.split()[0])
            _cat = os.path.join(self._base_dir, line.split()[1])
            temp_all += 1
            if not os.path.isfile(_image):
                temp_discard += 1
                continue
            if not os.path.isfile(_cat):
                temp_discard += 1
                continue
            self.images.append(_image)
            self.categories.append(_cat)
        print('all images have: %d, discard %d' % (temp_all,temp_discard))
        assert (len(self.images) == len(self.categories))
        if not 'rank' in args:
            print('Number of images in {}: {:d}'.format(self._data_list.split('/')[-1], len(self.images)))
        elif args.rank == 0:
            print('Number of images in {}: {:d}'.format(self._data_list.split('/')[-1], len(self.images)))

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'vis':
            temp =  self.transform_vis(sample)
            temp['name'] = self.images[index].split('/')[-1].split('.')[0]
            return temp

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        if _target.mode == 'RGB':
            _target = _target.convert('L')
        return _img, _target

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

    def transform_vis(self,sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=self.args.normal_mean),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return args.dataset + '  split=' + str(self.split)



if __name__ == "__main__":
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.crop_size = 513


    # for ii, sample in enumerate(dataloader):
    #     for jj in range(sample["image"].size()[0]):
    #         img = sample['image'].numpy()
    #         gt = sample['label'].numpy()
    #         tmp = np.array(gt[jj]).astype(np.uint8)
    #         segmap = decode_segmap(tmp, dataset='coco')
    #         img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
    #         img_tmp *= (0.229, 0.224, 0.225)
    #         img_tmp += (0.485, 0.456, 0.406)
    #         img_tmp *= 255.0
    #         img_tmp = img_tmp.astype(np.uint8)
    #         plt.figure()
    #         plt.title('display')
    #         plt.subplot(211)
    #         plt.imshow(img_tmp)
    #         plt.subplot(212)
    #         plt.imshow(segmap)

    #     if ii == 1:
    #         break

    # plt.show(block=True)