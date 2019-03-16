import torch

def get_parameters(model,key):
    for m in model.named_parameters():
        if key in m[0]:
            yield m[1]