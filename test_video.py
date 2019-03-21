import argparse
import os
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from modeling.generatenet import generate_net
from dataloaders.utils import decode_seg_map_sequence
from utils.metrics import Evaluator
from collections import OrderedDict
from utils.load import load_pretrained_mode
from utils import post_utils
from dataloaders import custom_transforms as tr

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
        _img_bgr = _img.copy()
        if self.args.gray_mode:
            _img = cv2.cvtColor(_img,cv2.COLOR_BGR2GRAY).copy()
        elif not self.args.bgr_mode:
            _img = _img[:,:,::-1].copy()
        sample = {'image':_img}
        
        sample = self.transform_val(sample)
        sample['ori'] = torch.from_numpy(np.array(_img_bgr))
        return sample

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(self.args.test_size,shrink=self.args.shrink),
            tr.Normalize(mean=self.args.normal_mean,std=self.args.normal_std,bgr_mode=self.args.bgr_mode,gray_mode=self.args.gray_mode),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def __len__(self):
        return self.framenum

class Valuator(object):
    def __init__(self, args):
        self.args = args
        self.args.batchnorm_function = torch.nn.BatchNorm2d
        # Define network
        model = generate_net(self.args)
        self.model = model

        # Using cuda
        if self.args.cuda:
            self.model = self.model.cuda()

        # Resuming checkpoint
        _,_,_ = load_pretrained_mode(self.model,checkpoint_path=self.args.resume)

        if self.args.twoframeavg > 0:
            print('averaging the two frame output mask')
            self.avgframe = post_utils.AverageFrame(self.args.twoframeavg)

    def visual(self,video_path):
        self.model.eval()
        print('\nvisualizing')
        vis_set = VideoDataset(self.args,video_path)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') #opencv3.0
        save_name = os.path.join(self.args.save_dir,video_path.split('/')[-1].split('.')[0])
        os.makedirs(self.args.save_dir,exist_ok=True)
        videoWriter = cv2.VideoWriter(save_name + '.avi', fourcc, float(vis_set.framerate), (vis_set.wid,vis_set.hei))
        vis_loader = DataLoader(vis_set, batch_size=self.args.batch_size, shuffle=False,drop_last=False)
        num_img_tr = len(vis_loader)
        print('=====>[frames: %5d]' % (num_img_tr * self.args.batch_size))
        for i, sample in enumerate(vis_loader):
            image, ori = sample['image'],sample['ori']
            image = image.cuda()
            with torch.no_grad():
                output = self.model(image)
                if isinstance(output,(tuple,list)):
                    output = output[0]
            output = torch.nn.functional.interpolate(output,size=ori.size()[1:3],mode='bilinear',align_corners=True)
            pred = output.data.cpu().numpy()
            ori = ori.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            if self.args.twoframeavg > 0:
                pred[0] = self.avgframe(pred[0])
            if self.args.blurlevel > 0:
                pred[0] = post_utils.bluranderode(pred[0],blurlevel=self.args.blurlevel)
            # pred = np.ones(pred.shape) - pred
            label = decode_seg_map_sequence(pred).cpu().numpy().transpose([0,2,3,1])
            label = label[:,:,:,::-1] # convert to BGR
            pred = np.stack([pred,pred,pred],axis=3)
            ori = ori.astype(dtype=np.uint8)
            label = label.astype(dtype=np.uint8)
            # ori *= pred.astype(dtype=np.uint8)
            # label[pred==0] = 0
            temp = self.addImage(ori,label)
            temp[pred == 0] = 0
            temp = temp.astype(np.uint8)
            # temp = temp[:,:,:,::-1]
            # cv2.imwrite(os.path.join(save_name,str(i)+'.jpg'),temp[0])
            videoWriter.write(temp[0])
        print('write %d frame' % (i+1))
        videoWriter.release()

    def addImage(self,img1_path,img2_path):
        alpha = 1
        beta = 0.7
        gamma = 0
        img_add = cv2.addWeighted(img1_path,alpha,img2_path,beta,gamma)
        return img_add

def main():
    from utils import parse_args
    args = parse_args.parse()
    args.batch_size = 1
    args.ft = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()    
    args.gpus = torch.cuda.device_count()
    print("torch.cuda.device_count()=",args.gpus)
    print(args)
    valuator = Valuator(args)
    video_paths = os.listdir(args.test_path)
    for i in video_paths:
        path = os.path.join(args.test_path,i)
        valuator.visual(path)

if __name__ == "__main__":
   main()
