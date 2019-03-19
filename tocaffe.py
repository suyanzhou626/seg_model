import argparse
import nart_tools.pytorch as pytorch
import nart_tools
import os
import torch
from modeling.generatenet import generate_net
from collections import OrderedDict
from utils.load import load_pretrained_mode

class ToCaffe(object):
    def __init__(self, args):
        self.args = args
        self.args.out_name = os.path.join(self.args.save_dir,'model')
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
        nart_tools.update_interp=True
        with pytorch.convert_mode():
            pytorch.convert(self.model,[self.args.input_shape],self.args.out_name,input_names=["data"],output_names=["out"])

def main():
    from .utils import parse_args
    args = parse_args.parse()
    print(args)
    tocaffe = ToCaffe(args)
    tocaffe.convert()

if __name__ == "__main__":
   main()
