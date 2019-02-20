from .dbl import Dbl
from .msc import MSC
from .v23 import V23_4x
from .vnet3_360 import Vnet3_360
from .vnet_pruning_1 import VnetPrun1
from .vnet_pruning_2 import VnetPrun2
from .vnet_pruning_3 import VnetPrun3
from .deeplab import DeepLab
from .v23_aug import V23_aug

network_map = {'v23_4x':V23_4x,'vnet3_360':Vnet3_360,'dbl':Dbl,'msc':MSC,'vnetprun1':VnetPrun1}
network_map['v23aug'] = V23_aug
network_map['vnetprun2'] = VnetPrun2
network_map['vnetprun3'] = VnetPrun3
network_map['deeplab'] = DeepLab