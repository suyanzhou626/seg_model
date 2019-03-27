import argparse
import os
import torch
from modeling.generatenet import generate_net
from collections import OrderedDict
from utils.load import load_pretrained_mode

class ToModel(object):
    def __init__(self, args):
        self.args = args
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir,exist_ok=True)
        self.args.out_name = os.path.join(self.args.save_dir,self.args.dataset+'.pth')
        self.args.batchnorm_function = torch.nn.BatchNorm2d
        # Define Dataloader
        self.nclass = self.args.num_classes
        # Define network
        model = generate_net(self.args)
        self.model = model
        self.model = self.model.cuda()
        # Resuming checkpoint
        _,_,_ = load_pretrained_mode(self.model,checkpoint_path=self.args.resume)

    def convert(self):
        torch.save(self.model,self.args.out_name)

def main():
    from utils import parse_args
    args = parse_args.parse()
    print(args)
    tomodel = ToModel(args)
    tomodel.convert()

if __name__ == "__main__":
   main()
