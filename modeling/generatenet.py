from .deeplabv3plus import DeepLabv3plus
from .hourglass import Hourglass
from .msc import MSC
from .v23 import V23_4x
from .vnet3_360 import Vnet3_360
def generate_net(args):
    if args.backbone == 'deeplabv3plus' or args.backbone == 'deeplabv3+':
        return DeepLabv3plus(args)
    elif args.backbone == 'hourglass':
        return Hourglass(args)
    elif args.backbone == 'msc':
        return MSC(args)
    elif args.backbone == 'v23':
        return V23_4x(args)
    elif args.backbone == 'vnet':
        return Vnet3_360(args)
    else:
        raise ValueError('generateNet.py: network %s is not support yet' % args.backbone)