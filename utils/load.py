import torch
import os

def load_pretrained_mode(model,checkpoint_path=None):
    assert(checkpoint_path is not None)
    model_state = model.state_dict()
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    start_epoch = 0
    optimizer = None
    best_pred = 0
    if 'state_dict' in checkpoint.keys():
        start_epoch = checkpoint['epoch']
        optimizer = checkpoint['optimizer']
        state_dict = checkpoint['state_dict']
        best_pred = checkpoint['best_pred']
        state_dict = {k.replace('module.',''): v for k,v in state_dict.items()}
    else:
        state_dict = {k.replace('module.',''): v for k,v in checkpoint.items()}
    model_params = len(model_state.keys())
    checkpoint_params = len(state_dict.keys())
    print('this model has {} params; this checkpoint has {} params'.format(model_params,checkpoint_params))
    if model_params > checkpoint_params:
        for i in model_state.keys():
            if i not in state_dict.keys():
                print('this param of the model dont in the checkpoint: {}'.format(i))
    num = 0
    total = 0
    for k,v in state_dict.items():
        total += 1
        if k in model_state.keys():
            right_flag = True
            if (len(v.size()) != len(model_state[k].size())):
                continue
            for i in range(len(v.size())):
                if v.size()[i] != model_state[k].size()[i]:
                    right_flag = False
                    break
            if right_flag:
                model_state[k] = v
                num += 1
        else:
            print('this param of the checkpoint dont in the model:{}'.format(k))
    model.load_state_dict(model_state)
    print('success for loading pretrained model params {}/{} from {}! (epoch: {})'.format(str(num),str(total),checkpoint_path,str(start_epoch)))
    return optimizer,start_epoch,best_pred