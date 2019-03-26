from .deeplabv3plus import DeepLabv3plus
from .dbl import Dbl
from .hourglass import Hourglass
from .msc import MSC
from .v23 import V23_4x
from .v23_aug import V23_aug
from .vnet3_360 import Vnet3_360
from .vnet_pruning_1 import VnetPrun1
from .vnet_pruning_2 import VnetPrun2
from .vnet_pruning_3 import VnetPrun3
from .vnet_mloss import VnetMloss
from .v23g import V23_G
from .aacn import AACN
def generate_net(args):
    if args.backbone == 'deeplabv3plus' or args.backbone == 'deeplabv3+':
        return DeepLabv3plus(args)
    elif args.backbone == 'dbl':
        return Dbl(args)
    elif args.backbone == 'hourglass':
        return Hourglass(args)
    elif args.backbone == 'msc':
        return MSC(args)
    elif args.backbone == 'v23':
        return V23_4x(args)
    elif args.backbone == 'v23aug':
        return V23_aug(args)
    elif args.backbone == 'vnet':
        return Vnet3_360(args)
    elif args.backbone == 'vnetprun1':
        return VnetPrun1(args)
    elif args.backbone == 'vnetprun2':
        return VnetPrun2(args)
    elif args.backbone == 'vnetprun3':
        return VnetPrun3(args)
    elif args.backbone == 'vnetmloss':
        return VnetMloss(args)
    elif args.backbone == 'v23g':
        return V23_G(args)
    elif args.backbone == 'aacn':
        return AACN(args)
    else:
        raise ValueError('generateNet.py: network %s is not support yet' % args.backbone)