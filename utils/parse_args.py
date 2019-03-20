import argparse

def parse():
    parser = argparse.ArgumentParser(description="PyTorch vnet Training")
    # necessary param about: model,dataset
    parser.add_argument('--backbone',type=str,default=None,help='choose the network') 
    parser.add_argument('--dataset', type=str, default=None,help='dataset name (default: pascal)')
    parser.add_argument('--data_dir',type=str,default=None,
                        help='path to dataset which add the *.txt is the image path')
    parser.add_argument('--train_list',type=str,default=None,help='path to train.txt')
    parser.add_argument('--val_list',type=str,default=None,help='path to val.txt')

    # necessary train param
    parser.add_argument('--input_size', type=int, default=None,help='crop image size')
    parser.add_argument('--test_size',type=int,default=None)
    parser.add_argument('--shrink',type=int,default=None)
    parser.add_argument('--num_classes',type=int,default=None,help='the number of classes')

    # optional train param
    parser.add_argument('--bgr_mode',action='store_true', default=False,help='input image is bgr but rgb')
    parser.add_argument('--gray_mode',action='store_true',default=False,help='input image conveted to gray')
    parser.add_argument('--normal_mean',type=float, nargs='*',default=[104.008,116.669,122.675])
    parser.add_argument('--normal_std',type=float,default=1.0)
    parser.add_argument('--rand_resize',type=float, nargs='*',default=[0.75,1.25])
    parser.add_argument('--rotate',type=int,default=0)
    parser.add_argument('--noise_param',type=float,nargs='*',default=None)
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--foreloss_weight',type=float,default=1)
    parser.add_argument('--seloss_weight',type=float,default=1)

    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--batch_size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--bn_var_mode',type=str,default='L2')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--save_dir',type=str,default=None,help='path to save model')

    # optimizer params
    parser.add_argument('--optim_method',type=str,default='sgd')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    # cuda, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')

    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')

    #tocaffe necessary params
    parser.add_argument('--input_shape', type=int, nargs='*',default=None,help='input image size')

    #video test necessary params
    parser.add_argument('--test_path',type=str,default=None,help='the dir including video used to test model')

    args = parser.parse_args()
    return args