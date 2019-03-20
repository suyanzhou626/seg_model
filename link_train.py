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
        self.model = DistModule(self.model,sync=False)

        # Clear start epoch if fine-tuning
        if rank == 0:
            print('Starting Epoch:', self.args.start_epoch)
            print('Total Epoches:', self.args.epochs)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        num_img_tr = len(self.train_loader)
        self.evaluator.reset()
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
            pred = output.data.clone()
            loss = loss/self.args.world_size
            loss.backward()
            reduce_gradients(self.model,sync=False)
            self.optimizer.step()
            link.allreduce(loss)
            train_loss += loss.item()
            pred = pred.data.cpu().numpy()
            target_array = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator_inner.add_batch(target_array,pred)
            self.evaluator.add_batch(target_array,pred)
            if i % 10 == 0:
                confusion_matrix_iter = torch.Tensor(self.evaluator_inner.confusion_matrix)
                link.allreduce(confusion_matrix_iter)
                self.evaluator_inner.confusion_matrix = confusion_matrix_iter.numpy()
                Acc_train = self.evaluator_inner.Pixel_Accuracy()
                Acc_class_train = self.evaluator_inner.Pixel_Accuracy_Class()
                mIoU_train,IoU_train = self.evaluator_inner.Mean_Intersection_over_Union()
                FWIoU_train = self.evaluator_inner.Frequency_Weighted_Intersection_over_Union()
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
        confusion_matrix_epoch = torch.Tensor(self.evaluator.confusion_matrix)
        link.allreduce(confusion_matrix_epoch)
        self.evaluator.confusion_matrix = confusion_matrix_epoch.numpy()
        Acc_train_epoch = self.evaluator.Pixel_Accuracy()
        Acc_class_train_epoch = self.evaluator.Pixel_Accuracy_Class()
        mIoU_train_epoch,IoU_train_epoch = self.evaluator.Mean_Intersection_over_Union()
        FWIoU_train_epoch = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        if self.args.rank == 0:
            stop_time = time.time()
            print('=====>[Epoch: %d, numImages: %5d   time_consuming: %d]' % 
            (epoch, num_img_tr * self.args.batch_size*self.args.world_size,stop_time-start_time))
            print("Loss: %.3f  Acc: %.4f,  Acc_class: %.4f,  mIoU: %.4f,  fwIoU: %.4f\n\n" % (train_loss/(num_img_tr),
                Acc_train_epoch, Acc_class_train_epoch, mIoU_train_epoch, FWIoU_train_epoch))
            print("IoU per class: ",IoU_train_epoch)
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
        stop_time = time.time()
        # Fast test during the training
        confusion_matrix_val = torch.Tensor(self.evaluator.confusion_matrix)
        link.allreduce(confusion_matrix_val)
        self.evaluator.confusion_matrix = confusion_matrix_val.numpy()
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU,IoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        if self.args.rank == 0:
            self.writer.add_scalar('val/total_loss_epoch', test_loss/(num_img_tr), epoch)
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
    from utils import parse_args
    args = parse_args.parse()
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
