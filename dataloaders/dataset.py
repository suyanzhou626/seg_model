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
        print('Number of images in {}: {:d}'.format(self._data_list.split('/')[-1], len(self.images)))

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _img = np.array(_img).astype(dtype=np.float32)
        if self.args.bgr_mode:
            _img = _img[:,:,::-1].copy()  #convert to BGR
        _target = Image.open(self.categories[index])
        if (_target.mode != 'L' and _target.mode != 'P'):
            temp = np.unique(np.array(_target))
            if np.max(temp)<self.args.num_classes:
                _target = _target.convert('L')
            else:
                raise 'error in %s' % self.categories[index]
        _target = np.array(_target).astype(dtype=np.float32)
        return _img, _target

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