import os
import numpy as np
import torch
import time
import gc
import linklink as link
from torch.utils.data import DataLoader
from collections import OrderedDict
from modeling.generatenet import generate_net
from dataloaders.memcached_dataset import McDataset
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.distributed_utils import *
from utils.utils import DistributedSampler,simple_group_split,DistributedGivenIterationSampler
from utils.load import load_pretrained_mode

def get_params(model, key):
	for m in model.named_modules():
		if key == '1x':
			if 'backbone' in m[0] and isinstance(m[1], torch.nn.Conv2d):
				for p in m[1].parameters():
					yield p
		elif key == '10x':
			if 'backbone' not in m[0] and isinstance(m[1], torch.nn.Conv2d):
				for p in m[1].parameters():
					yield p

class Trainer(object):
    def __init__(self, args):
        rank, world_size = dist_init()
        args.bn_var_mode = (link.syncbnVarMode_t.L1 
                                       if args.bn_var_mode == 'L1' 
                                       else link.syncbnVarMode_t.L2)
        args.rank = rank
        args.world_size = world_size
        args.bn_group = simple_group_split(world_size, rank, 1)
        args.gpus = torch.cuda.device_count()

        self.args = args
        def BNFunc(*args, **kwargs):
            return link.nn.SyncBatchNorm2d(*args, group=self.args.bn_group, sync_stats=True, var_mode=self.args.bn_var_mode, **kwargs)
        self.args.batchnorm_function = BNFunc
        if rank == 0:
            print(self.args)
        # Define Saver
            self.saver = Saver(self.args)
        # Define Tensorboard Summary
            self.summary = TensorboardSummary()
            self.writer = self.summary.create_summary(self.saver.experiment_dir)
        
        kwargs = {'num_workers': self.args.gpus, 'pin_memory': True}
        self.train_set = McDataset(self.args,self.args.train_list,split='train')
        self.val_set = McDataset(self.args,self.args.val_list,split='val')

        self.train_sampler = DistributedSampler(self.train_set)
        self.val_sampler = DistributedSampler(self.val_set,round_up=True)

        self.train_loader = DataLoader(self.train_set,batch_size=self.args.batch_size,sampler=self.train_sampler)
        self.val_loader = DataLoader(self.val_set,batch_size=1,sampler=self.val_sampler)
        self.nclass = self.args.num_classes
        weight = torch.from_numpy(np.zeros((self.nclass,))).type(torch.FloatTensor)

        if self.args.use_balanced_weights:
            classes_weights_path = os.path.join(self.args.save_dir, self.args.dataset,'classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
                weight = torch.from_numpy(weight.astype(np.float32))
            elif rank == 0:
                temp_train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True,drop_last = True, **kwargs)
                weight = calculate_weigths_labels(self.args.save_dir,self.args.dataset, temp_train_loader, self.nclass)
                weight = torch.from_numpy(weight.astype(np.float32)).type(torch.FloatTensor)
            link.broadcast(weight,root=0)
                    
        else:
            weight = None
        
        # Define network
        model = generate_net(self.args)
        train_params = [{'params': model.get_conv_weight_params(), 'lr': self.args.lr,'weight_decay':self.args.weight_decay},
                        {'params': model.get_conv_bias_params(), 'lr': self.args.lr * 2,'weight_decay':0}]

        # for fituning the xception deeplab from xception pretrained model
        if 'deeplab' in self.args.backbone and self.args.ft and self.args.resume is None:
            train_params = [
			{'params': get_params(model,key='1x'), 'lr': self.args.lr,'weight_decay':self.args.weight_decay},
			{'params': get_params(model,key='10x'), 'lr': 10*self.args.lr,'weight_decay':self.args.weight_decay}
		]
        # train_params = [{'params':model.parameters(),'lr':self.args.lr}]

        # Define Optimizer
        if self.args.optim_method == 'sgd':
            optimizer = torch.optim.SGD(train_params, momentum=self.args.momentum, lr=self.args.lr,
                                    weight_decay=0, nesterov=self.args.nesterov)
        elif self.args.optim_method == 'adagrad':
            optimizer = torch.optim.Adagrad(train_params,lr=self.args.lr,weight_decay=self.args.weight_decay)
        else:
            pass

        # Define Criterion
        # whether to use class balanced weights
        self.criterion = SegmentationLosses(weight=weight, cuda= not self.args.no_cuda,foreloss_weight=args.foreloss_weight,seloss_weight=args.seloss_weight).build_loss(mode=self.args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        self.evaluator_inner = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr,
                                            self.args.epochs, len(self.train_loader))
        self.model = self.model.cuda()
        self.args.start_epoch = 0
        # Resuming checkpoint
        self.best_pred = 0.0
        if self.args.resume is not None:
            optimizer,start_epoch,best_pred = load_pretrained_mode(self.model,checkpoint_path=self.args.resume)
            if not self.args.ft and optimizer is not None:
                self.optimizer.load_state_dict(optimizer)
                self.args.start_epoch = start_epoch
            self.best_pred = best_pred
        #     if not os.path.isfile(self.args.resume) and rank == 0:
        #         raise RuntimeError("=> no checkpoint found at '{}'" .format(self.args.resume))
        #     checkpoint = torch.load(self.args.resume)
        #     self.args.start_epoch = checkpoint['epoch']
        #     new_state_dict = OrderedDict()
        #     for k,v in checkpoint['state_dict'].items():
        #         if 'module' in k:
        #             name = k[7:]
        #         else:
        #             name = k
        #         new_state_dict[name] = v
        #     self.model.load_state_dict(new_state_dict,strict=False)
        #     if not self.args.ft:
        #         self.optimizer.load_state_dict(checkpoint['optimizer'])
        #     self.best_pred = checkpoint['best_pred']
        #     if rank == 0:
        #         print("=> loaded checkpoint '{}' (epoch {})"
        #           .format(self.args.resume, checkpoint['epoch']))
        #     del checkpoint,new_state_dict,k,v,name
        #     gc.collect
        #     torch.cuda.empty_cache()
        self.model = DistModule(self.model,sync=False)

        # Clear start epoch if fine-tuning
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
            start_time = time.time()
        for i,sample in enumerate(self.train_loader):
            image, target = sample['image'], sample['label']
            if not self.args.no_cuda:
                image, target = image.cuda(), target.cuda()
            current_lr = self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss,output = self.criterion(output,target)
            loss = loss/self.args.world_size
            loss.backward()
            reduce_gradients(self.model,sync=False)
            self.optimizer.step()
            link.allreduce(loss)
            train_loss += loss.item()
            pred = output.data.cpu().numpy()
            target_array = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator_inner.add_batch(target_array,pred)
            if i % 10 == 0:
                confusion_matrix = torch.Tensor(self.evaluator_inner.confusion_matrix)
                link.allreduce(confusion_matrix)
                self.evaluator_inner.confusion_matrix = confusion_matrix.numpy()
                Acc_train = self.evaluator_inner.Pixel_Accuracy()
                Acc_class_train = self.evaluator_inner.Pixel_Accuracy_Class()
                mIoU_train,IoU_train = self.evaluator_inner.Mean_Intersection_over_Union()
                FWIoU_train = self.evaluator_inner.Frequency_Weighted_Intersection_over_Union()
                # Acc_train = torch.Tensor([self.evaluator_inner.Pixel_Accuracy()])
                # Acc_class_train = torch.Tensor([self.evaluator_inner.Pixel_Accuracy_Class()])
                # mIoU_train,IoU_train = self.evaluator_inner.Mean_Intersection_over_Union()
                # mIoU_train = torch.Tensor([mIoU_train])
                # IoU_train = torch.Tensor(IoU_train)
                # FWIoU_train = torch.Tensor([self.evaluator_inner.Frequency_Weighted_Intersection_over_Union()])
                # link.allreduce(IoU_train)
                # link.allreduce(Acc_train)
                # link.allreduce(Acc_class_train)
                # link.allreduce(mIoU_train)
                # link.allreduce(FWIoU_train)
                # IoU_train = IoU_train.numpy()
                # IoU_train = IoU_train/self.args.world_size
                # Acc_train = Acc_train.item()/self.args.world_size
                # Acc_class_train = Acc_class_train.item()/self.args.world_size
                # mIoU_train = mIoU_train.item()/self.args.world_size
                # FWIoU_train = FWIoU_train.item()/self.args.world_size
                if self.args.rank == 0:
                    print('\n===>Iteration  %d/%d    learning_rate: %.6f   metric:' % (i,num_img_tr,current_lr))
                    print('=>Train loss: %.4f    acc: %.4f     m_acc: %.4f     miou: %.4f     fwiou: %.4f' 
                                % (loss.item(),Acc_train,Acc_class_train,mIoU_train,FWIoU_train))
                    print("IoU per class: ",IoU_train)
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
            stop_time = time.time()
            print('=====>[Epoch: %d, numImages: %5d   time_consuming: %d]' % 
            (epoch, num_img_tr * self.args.batch_size*self.args.world_size,stop_time-start_time))
            self.writer.add_scalar('train/total_loss_epoch', train_loss/(num_img_tr), epoch)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        test_loss = 0.0
        num_img_tr = len(self.val_loader)
        if self.args.rank == 0:
            print('\nValidation')
            start_time = time.time()
        for i, sample in enumerate(self.val_loader):
            image, target = sample['image'], sample['label']
            if not self.args.no_cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                loss,output = self.criterion(output, target)
            loss = loss/self.args.world_size
            link.allreduce(loss)
            test_loss += loss.item()
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            # print('===>Iteration  %d/%d' % (i,num_img_tr))
            # print('test loss: %.3f' % (test_loss / (i + 1)))
        stop_time = time.time()
        # Fast test during the training
        Acc = torch.Tensor([self.evaluator.Pixel_Accuracy()])
        Acc_class = torch.Tensor([self.evaluator.Pixel_Accuracy_Class()])
        mIoU,IoU = self.evaluator.Mean_Intersection_over_Union()
        mIoU = torch.Tensor([mIoU])
        IoU = torch.Tensor(IoU)
        FWIoU = torch.Tensor([self.evaluator.Frequency_Weighted_Intersection_over_Union()])
        link.allreduce(Acc)
        link.allreduce(Acc_class)
        link.allreduce(mIoU)
        link.allreduce(FWIoU)
        link.allreduce(IoU)
        IoU = IoU.numpy()
        IoU = IoU/self.args.world_size
        Acc = Acc.item()/self.args.world_size
        Acc_class = Acc_class.item()/self.args.world_size
        mIoU = mIoU.item()/self.args.world_size
        FWIoU = FWIoU.item()/self.args.world_size
        if self.args.rank == 0:
            self.writer.add_scalar('val/total_loss_epoch', test_loss/(self.args.world_size*num_img_tr), epoch)
            self.writer.add_scalar('val/mIoU', mIoU, epoch)
            self.writer.add_scalar('val/Acc', Acc, epoch)
            self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
            self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
            print('=====>[Epoch: %d, numImages: %5d   previous best=%.4f    time_consuming: %d]' % (epoch, num_img_tr*self.args.world_size,self.best_pred,stop_time-start_time))
            print("Loss: %.3f  Acc: %.4f,  Acc_class: %.4f,  mIoU: %.4f,  fwIoU: %.4f\n\n" % (test_loss/(num_img_tr),Acc, Acc_class, mIoU, FWIoU))
            print("IoU per class: ",IoU)

        new_pred = mIoU
        if self.args.rank == 0:
            if new_pred > self.best_pred:
                is_best = True
                self.best_pred = new_pred
            else:
                is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': new_pred,
            }, is_best)

def main():
    import argparse
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

    args = parser.parse_args()
    if args.test_size is None:
        args.test_size = args.input_size
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
    if trainer.args.rank == 0:
        trainer.writer.close()

if __name__ == "__main__":
   main()
