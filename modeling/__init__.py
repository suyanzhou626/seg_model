from .dbl import Dbl
from .hed import HED_vgg16
from .msc import MSC
from .v23 import V23_4x
from .vnet3_360 import Vnet3_360
from .vnet_pruning_1 import VnetPrun1

network_map = {'v23_4x':V23_4x,'vnet3_360':Vnet3_360,'dbl':Dbl,'msc':MSC,'hed':HED_vgg16,'vnetprun1':VnetPrun1}