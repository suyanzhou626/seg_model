import argparse
import os
import numpy as np
import torch
import PIL.Image as img
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from modeling.v23 import V23_4x
from modeling.vnet3_360 import Vnet3_360
from modeling.dbl import Dbl
from modeling.msc import MSC
from dataloaders.utils import decode_seg_map_sequence
from utils.metrics import Evaluator

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
        img /= 255.0

        return img


class VideoDataset(Dataset):
    def __init__(self,video_path):
        super().__init__()
        cap = cv2.VideoCapture(video_path)
        self.wid = int(cap.get(3))
        self.hei = int(cap.get(4))
        self.framerate = int(cap.get(5))
        self.framenum = int(cap.get(7))
        self.images = []
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
        input_image = self.transform_val(_img)
        _img = torch.from_numpy(_img)
        sample = {'input':input_image,'ori':_img}
        return sample

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            Normalize([104.008,116.669,122.675]),
            transforms.ToTensor()
            ])

        return composed_transforms(sample)

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
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(self.args.resume, checkpoint['epoch']))
    



    def visual(self,video_path):
        self.model.eval()
        print('\nvisualizing')
        vis_set = VideoDataset(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') #opencv3.0
        save_name = os.path.join(self.args.save_dir,video_path.split('/')[-1].split('.')[0])
        if not os.path.exists(save_name):
            os.mkdir(save_name)
        videoWriter = cv2.VideoWriter(save_name + '.avi', fourcc, float(vis_set.framerate), (vis_set.wid,vis_set.hei))
        vis_loader = DataLoader(vis_set, batch_size=self.args.batch_size, shuffle=False,drop_last=False)
        num_img_tr = len(vis_loader)
        print('=====>[frames: %5d]' % (num_img_tr * self.args.batch_size))
        for i, sample in enumerate(vis_loader):
            image, ori = sample['input'],sample['ori']
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                if self.args.backbone == 'dbl':
                    _,output = self.model(image)
                else:
                    output = self.model(image)

            pred = output.data.cpu().numpy()
            ori = ori.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred = np.ones(pred.shape) - pred
            label = decode_seg_map_sequence(pred).cpu().numpy().transpose([0,2,3,1])
            label = label[:,:,:,::-1]
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

    # training hyper params

    parser.add_argument('--save_dir',type=str,default=None,help='path to save model')

    # cuda, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')


    network_map = {'v23_4x':V23_4x,'vnet3_360':Vnet3_360,'dbl':Dbl,'msc':MSC}
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
