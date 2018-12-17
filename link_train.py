import argparse
import os
import numpy as np
import torch
import linklink as link
from torch.utils.data import DataLoader

from modeling.v23 import V23_4x
from modeling.vnet3_360 import Vnet3_360
from modeling.dbl import Dbl
from dataloaders import make_data_loader
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.distributed_utils import *
from utils.utils import DistributedSampler,simple_group_split

class Trainer(object):
    def __init__(self, args):
        rank, world_size = dist_init()
        if args.bn_group_size is None:
            args.bn_group_size = world_size
        args.bn_var_mode = (link.syncbnVarMode_t.L1 
                                       if args.bn_var_mode == 'L1' 
                                       else link.syncbnVarMode_t.L2)
        args.rank = rank
        args.world_size = world_size
        args.bn_group = simple_group_split(world_size, rank, world_size//args.bn_group_size)
        self.args = args
        if rank == 0:
            print("torch.cuda.device_count()=",args.gpus)
            print(args)
        # Define Saver
        self.saver = Saver(args)
        if rank == 0:
            self.saver.save_experiment_config()
            # Define Tensorboard Summary
            self.summary = TensorboardSummary(self.saver.experiment_dir)
            self.writer = self.summary.create_summary()
        
        kwargs = {'num_workers': args.gpus, 'pin_memory': True}
        self.train_set, self.val_set, self.test_set = make_data_loader(args, **kwargs)
        self.train_sampler = DistributedSampler(self.train_set)
        self.val_sampler = DistributedSampler(self.val_set)
        self.train_loader = DataLoader(self.train_set,batch_size=args.batch_size,drop_last=True,sampler=self.train_sampler)
        self.val_loader = DataLoader(self.val_set,batch_size=args.batch_size,drop_last=True,sampler=self.val_sampler)
        self.nclass = args.num_classes
        weight = torch.from_numpy(np.zeros((self.nclass,)))

        if args.use_balanced_weights:
            classes_weights_path = os.path.join(args.save_dir, args.dataset,'classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
                weight = torch.from_numpy(weight.astype(np.float32))
            elif rank == 0:
                weight = calculate_weigths_labels(args.save_dir,args.dataset, self.train_loader, self.nclass)
                weight = torch.from_numpy(weight.astype(np.float32)).type(torch.FloatTensor)
            link.broadcast(weight,root=0)
                    
        else:
            weight = None
        
        # Define network
        model = args.network(self.args)

        train_params = [{'params': model.get_conv_weight_params(), 'lr': args.lr,'weight_decay':args.weight_decay},
                        {'params': model.get_conv_bias_params(), 'lr': args.lr * 2,'weight_decay':0},
                        {'params': model.get_bn_prelu_params(),'lr': args.lr,'weight_decay':0}]
        # train_params = [{'params':model.parameters(),'lr':args.lr}]

        # Define Optimizer
        if args.optim_method == 'sgd':
            optimizer = torch.optim.SGD(train_params, momentum=args.momentum, lr=args.lr,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        else:
            pass

        # Define Criterion
        # whether to use class balanced weights
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        self.evaluator_inner = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))
        self.model = self.model.cuda()
        model = DistModule(model,sync=False)
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume) and rank == 0:
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            if rank == 0:
                print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            self.args.start_epoch = 0
        if rank == 0:
            print('Starting Epoch:', self.args.start_epoch)
            print('Total Epoches:', self.args.epochs)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        num_img_tr = len(self.train_loader)
        self.evaluator_inner.reset()
        if self.args.rank == 0:
            print('Training')
            print('=====>[Epoch: %d, numImages: %5d   previous best=%.4f]' % (epoch, num_img_tr * self.args.batch_size*self.args.world_size,self.best_pred))
        for i,sample in enumerate(self.train_loader):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            current_lr = self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if self.args.backbone == 'dbl':
                output1,output = self.model(image)
                loss1 = self.criterion(output1, target)
                loss2 = self.criterion(output, target)
                loss = loss1+loss2
            else:
                output = self.model(image)
                loss = self.criterion(output,target)
            loss.backward()
            reduce_gradients(self.model,sync=False)
            self.optimizer.step()
            link.allreduce(loss)
            train_loss += loss.item()
            pred = output.data.cpu().numpy()
            target_array = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator_inner.add_batch(target_array,pred)
            if i % 5 == 0 or i == num_img_tr -1:
                Acc_train = torch.Tensor([self.evaluator_inner.Pixel_Accuracy()])
                Acc_class_train = torch.Tensor([self.evaluator_inner.Pixel_Accuracy_Class()])
                mIoU_train = torch.Tensor([self.evaluator_inner.Mean_Intersection_over_Union()])
                FWIoU_train = torch.Tensor([self.evaluator_inner.Frequency_Weighted_Intersection_over_Union()])
                link.allreduce(Acc_train)
                link.allreduce(Acc_class_train)
                link.allreduce(mIoU_train)
                link.allreduce(FWIoU_train)
                Acc_train = Acc_train.item()/self.args.world_size
                Acc_class_train = Acc_class_train.item()/self.args.world_size
                mIoU_train = mIoU_train.item()/self.args.world_size
                FWIoU_train = FWIoU_train.item()/self.args.world_size
                if self.args.rank == 0:
                    print('\n===>Iteration  %d/%d    learning_rate: %.6f   metric:' % (i,num_img_tr,current_lr))
                    print('=>Train loss: %.4f    acc: %.4f     m_acc: %.4f     miou: %.4f     fwiou: %.4f' 
                                % (loss.item(),Acc_train,Acc_class_train,mIoU_train,FWIoU_train))
                self.evaluator_inner.reset()
            if self.args.rank == 0:
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if num_img_tr > 10:
                if i % (num_img_tr // 10) == 0:
                    global_step = i + num_img_tr * epoch
                    if self.args.rank == 0:
                        self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
            else:
                global_step = i + num_img_tr * epoch
                if self.args.rank == 0:
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
        if self.args.rank == 0:
            self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        test_loss = 0.0
        num_img_tr = len(self.val_loader)
        if self.args.rank == 0:
            print('\nValidation')
            print('=====>[Epoch: %d, numImages: %5d]' % (epoch, num_img_tr * self.args.batch_size*self.args.world_size))
        for i, sample in enumerate(self.val_loader):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                if self.args.backbone == 'dbl':
                    output1,output = self.model(image)
                    loss = self.criterion(output1,target) + self.criterion(output,target)
                else:
                    output = self.model(image)
                    loss = self.criterion(output, target)
            link.allreduce(loss)
            test_loss += loss.item()
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            # print('===>Iteration  %d/%d' % (i,num_img_tr))
            # print('test loss: %.3f' % (test_loss / (i + 1)))

        # Fast test during the training
        Acc = torch.Tensor([self.evaluator.Pixel_Accuracy()])
        Acc_class = torch.Tensor([self.evaluator.Pixel_Accuracy_Class()])
        mIoU = torch.Tensor([self.evaluator.Mean_Intersection_over_Union()])
        FWIoU = torch.Tensor([self.evaluator.Frequency_Weighted_Intersection_over_Union()])
        link.allreduce(Acc)
        link.allreduce(Acc_class)
        link.allreduce(mIoU)
        link.allreduce(FWIoU)
        Acc = Acc.item()/self.args.world_size
        Acc_class = Acc_class.item()/self.args.world_size
        mIoU = mIoU.item()/self.args.world_size
        FWIoU = FWIoU.item()/self.args.world_size
        if self.args.rank == 0:
            self.writer.add_scalar('val/total_loss_epoch', test_loss/num_img_tr, epoch)
            self.writer.add_scalar('val/mIoU', mIoU, epoch)
            self.writer.add_scalar('val/Acc', Acc, epoch)
            self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
            self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
            print("Loss: %.3f  Acc: %.4f,  Acc_class: %.4f,  mIoU: %.4f,  fwIoU: %.4f\n\n" % (test_loss,Acc, Acc_class, mIoU, FWIoU))

        new_pred = mIoU
        if new_pred > self.best_pred and self.args.rank == 0:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch vnet Training")

    # parser.add_argument('--out_stride', type=int, default=8,
    #                     help='network output stride (default: 8)')
    parser.add_argument('--backbone',type=str,default=None,help='choose the network') 
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset name (default: pascal)')
    parser.add_argument('--data_dir',type=str,default=None,
                        help='path to dataset which add the *.txt is the image path')
    parser.add_argument('--train_list',type=str,default=None,help='path to train.txt')
    parser.add_argument('--val_list',type=str,default=None,help='path to val.txt')
    parser.add_argument('--crop_size', type=int, default=225,
                        help='crop image size')
    parser.add_argument('--normal_mean',type=float, nargs='*',default=[104.008,116.669,122.675])
    parser.add_argument('--num_classes',type=int,default=None,help='the number of classes')
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--optim_method',type=str,default='sgd')
    parser.add_argument('--bn_group_size',type=int,default=None)
    parser.add_argument('--bn_var_mode',type=str,default='L2')
    # parser.add_argument('--test_batch_size', type=int, default=None,
    #                     metavar='N', help='input batch size for \
    #                             testing (default: auto)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--save_dir',type=str,default=None,help='path to save model')
    # optimizer params
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
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')

    network_map = {'v23_4x':V23_4x,'vnet3_360':Vnet3_360,'dbl':Dbl}
    args = parser.parse_args()
    args.network = network_map[args.backbone]
    args.cuda = not args.no_cuda and torch.cuda.is_available()    
    args.gpus = torch.cuda.device_count()
    if args.cuda and args.gpus > 1:
        args.sync_bn = True
    else:
        args.sync_bn = False
    # if args.test_batch_size is None:
    #     args.test_batch_size = args.batch_size
    if args.checkname is None:
        args.checkname = args.backbone
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
    if trainer.args.rank == 0:
        trainer.writer.close()

if __name__ == "__main__":
   main()
