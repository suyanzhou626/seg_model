from dataloaders import dataset
from torch.utils.data import DataLoader
import glob
import os
def make_data_loader(args, **kwargs):
    data_dir = args.data_dir
    dataset_name = args.dataset
    train_list = glob.glob(os.path.join(data_dir,'*train_list_'+dataset_name+'.txt')) if args.train_list is None else [args.train_list]
    if len(train_list)> 1:
        print('train_list: ',train_list)
        inx = input('please choose one index to use as train_list:')
        train_list=train_list[int(inx)]
    elif len(train_list) < 1:
        train_list = None
    else:
        train_list = train_list[0]
    val_list = glob.glob(os.path.join(data_dir,'*val_list_'+dataset_name+'.txt')) if args.val_list is None else [args.val_list]
    if len(val_list) > 1:
        print('val_list: ',val_list)
        inx = input('please choose one index to use as val_list:')
        val_list=[val_list[int(inx)]]
    elif len(val_list) < 1:
        val_list = None
    else:
        val_list = val_list[0]
    test_list = glob.glob(os.path.join(data_dir,'*test_list_'+dataset_name+'.txt'))
    if len(test_list) > 1:
        print('test_list: ',test_list)
        inx = input('please choose one index to use as test_list:')
        test_list=[test_list[int(inx)]]
    elif len(test_list) < 1:
        test_list=None
    else:
        test_list = test_list[0]
    if train_list is not None:
        train_set = dataset.GenDataset(args,train_list,split='train')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,drop_last = True, **kwargs)
    else:
        train_set = None
        train_loader = None
        print('have no trainset')
    
    if val_list is not None:
        val_set = dataset.GenDataset(args,val_list,split='val')
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,drop_last = True, **kwargs)
    else:
        val_set = None
        val_loader = None
        print('have no valset')
    
    if test_list is not None:
        test_set = dataset.GenDataset(args,test_list,split='val')
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,drop_last = True, **kwargs)
    else:
        test_set = None
        test_loader = None
    if 'rank' in args:
        return train_set,val_set,test_set
    else:
        return train_loader,val_loader,test_loader
