import argparse
import os
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from modeling import network_map
from dataloaders.utils import decode_seg_map_sequence
from utils.metrics import Evaluator
from collections import OrderedDict
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.)):
        self.mean = mean
    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img -= self.mean
#        img /= 255.0

        return img
class Resize(object):
    def __init__(self,target_size):
        self.size = target_size[0] if isinstance(target_size,list) else target_size
    
    def __call__(self,img):
        w, h = img.size
        if h < w:
            ow = self.size
            oh = int(1.0 * h * ow / w)
        else:
            oh = self.size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        padh = self.size - oh if oh < self.size else 0
        padw = self.size - ow if ow < self.size else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        return {'image': img,
                'ow':ow,'oh':oh}
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()

        return img


class VideoDataset(Dataset):
    def __init__(self,args,video_path):
        super().__init__()
        cap = cv2.VideoCapture(video_path)
        self.wid = int(cap.get(3))
        self.hei = int(cap.get(4))
        self.framerate = int(cap.get(5))
        self.framenum = int(cap.get(7))
        self.images = []
        self.args = args
        cnt = 0
        while(cap.isOpened()):
            a,b=cap.read()
            if not a:
                break
            self.images.append(b)
            cnt+=1
        cap.release()

        print('this video({}) has %d frame(%d)'.format(video_path) % (self.framenum,cnt))
    def __getitem__(self, index):
        _img = self.images[index]
        temp = Image.fromarray(_img,mode='RGB')
        input_image = self.transform_val(temp)
        _img = torch.from_numpy(_img)
        sample = {'input':input_image['image'],'ori':_img,'ow':input_image['ow'],'oh':input_image['oh']}
        return sample

    def transform_val(self, sample):
        pre_trans = Resize(self.args.crop_size)
        temp = pre_trans(sample)
        composed_transforms = transforms.Compose([
            Normalize(mean=self.args.normal_mean),
            ToTensor()
        ])
        res = composed_transforms(temp['image'])
        return {'image':res,'ow':temp['ow'],'oh':temp['oh']}

    def __len__(self):
        return self.framenum

class Valuator(object):
    def __init__(self, args):
        self.args = args
        self.args.batchnorm_function = torch.nn.BatchNorm2d
        # Define network
        model = self.args.network(self.args)
        self.model = model

        # Using cuda
        if self.args.cuda:
            self.model = self.model.cuda()

        # Resuming checkpoint
        if not os.path.isfile(self.args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(self.args.resume))
        checkpoint = torch.load(self.args.resume)
        new_state_dict = OrderedDict()
        for k,v in checkpoint['state_dict'].items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(self.args.resume, checkpoint['epoch']))
    



    def visual(self,video_path):
        self.model.eval()
        print('\nvisualizing')
        vis_set = VideoDataset(self.args,video_path)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') #opencv3.0
        save_name = os.path.join(self.args.save_dir,video_path.split('/')[-1].split('.')[0])
        if not os.path.exists(save_name):
            os.mkdir(save_name)
        videoWriter = cv2.VideoWriter(save_name + '.avi', fourcc, float(vis_set.framerate), (vis_set.wid,vis_set.hei))
        vis_loader = DataLoader(vis_set, batch_size=self.args.batch_size, shuffle=False,drop_last=False)
        num_img_tr = len(vis_loader)
        print('=====>[frames: %5d]' % (num_img_tr * self.args.batch_size))
        for i, sample in enumerate(vis_loader):
            image, ori, ow,oh = sample['input'],sample['ori'],sample['ow'], sample['oh']
            image = image.cuda()
            with torch.no_grad():
                if self.args.backbone == 'dbl':
                    _,output = self.model(image)
                else:
                    output = self.model(image)
            output = output[:,:,0:oh[0].item(),0:ow[0].item()]
            output = torch.nn.functional.interpolate(output,size=ori.size()[1:3],mode='bilinear',align_corners=True)
            pred = output.data.cpu().numpy()
            ori = ori.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred = np.ones(pred.shape) - pred
            label = decode_seg_map_sequence(pred).cpu().numpy().transpose([0,2,3,1])
            label = label[:,:,:,::-1] # convert to BGR
            pred = np.stack([pred,pred,pred],axis=3)
            ori[pred==1] = 0
            label[pred==0] = 0
            temp = ori + label
            temp = temp.astype(np.uint8)
            cv2.imwrite(os.path.join(save_name,str(i)+'.jpg'),temp[0])
            videoWriter.write(temp[0])
        print('write %d frame' % (i+1))
        videoWriter.release()

def main():
    parser = argparse.ArgumentParser(description="PyTorch vnet Training")
    parser.add_argument('--backbone',type=str,default=None,help='choose the network') 
    parser.add_argument('--test_path',type=str,default=None,help='path to val.txt')
    parser.add_argument('--num_classes',type=int,default=None,help='the number of classes')
    parser.add_argument('--crop_size', type=int, default=225,
                        help='crop image size')
    parser.add_argument('--normal_mean',type=float, nargs='*',default=[104.008,116.669,122.675])
    # training hyper params

    parser.add_argument('--save_dir',type=str,default=None,help='path to save model')

    # cuda, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')


    args = parser.parse_args()
    args.batch_size = 1
    args.network = network_map[args.backbone]
    args.cuda = not args.no_cuda and torch.cuda.is_available()    
    args.gpus = torch.cuda.device_count()
    print("torch.cuda.device_count()=",args.gpus)
    # if args.test_batch_size is None:
    #     args.test_batch_size = args.batch_size
    print(args)
    valuator = Valuator(args)
    video_paths = os.listdir(args.test_path)
    for i in video_paths:
        path = os.path.join(args.test_path,i)
        valuator.visual(path)

if __name__ == "__main__":
   main()
