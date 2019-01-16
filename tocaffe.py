import argparse
import nart_tools.pytorch as pytorch
import os
import torch
from modeling import network_map
from collections import OrderedDict

class ToCaffe(object):
    def __init__(self, args):
        self.args = args
        self.args.batchnorm_function = torch.nn.BatchNorm2d
        # Define Dataloader
        self.nclass = self.args.num_classes
        # Define network
        model = self.args.network(self.args)
        self.model = model

        # Resuming checkpoint
        if not os.path.isfile(self.args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(self.args.resume))
        checkpoint = torch.load(self.args.resume,map_location=lambda storage, loc: storage)
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

    def convert(self):
        with pytorch.convert_mode():
            pytorch.convert(self.model,[self.args.input_shape],self.args.out_name,input_names=["data"],output_names=["out"])

def main():
    parser = argparse.ArgumentParser(description="PyTorch vnet Training")

    parser.add_argument('--backbone',type=str,default=None,help='choose the network') 

    parser.add_argument('--num_classes',type=int,default=None,help='the number of classes')
    parser.add_argument('--input_shape', type=int, nargs='*',default=(3,225,225),
                        help='input image size')

    parser.add_argument('--out_name',type=str,default=None,help='path to save model')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')


    args = parser.parse_args()
    args.network = network_map[args.backbone]

    # if args.test_batch_size is None:
    #     args.test_batch_size = args.batch_size
    print(args)
    tocaffe = ToCaffe(args)
    tocaffe.convert()

if __name__ == "__main__":
   main()
