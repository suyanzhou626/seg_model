import argparse
import os
import numpy as np
import torch
import time
import gc
from collections import OrderedDict
from modeling.generatenet import generate_net
from dataloaders import make_data_loader
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from utils.load import load_pretrained_mode

class Trainer(object):
    def __init__(self, args):
        self.args = args
        if self.args.sync_bn:
            self.args.batchnorm_function = SynchronizedBatchNorm2d
        else:
            self.args.batchnorm_function = torch.nn.BatchNorm2d
        print(self.args)
        # Define Saver
        self.saver = Saver(self.args)
        # Define Tensorboard Summary
        self.summary = TensorboardSummary()
        self.writer = self.summary.create_summary(self.saver.experiment_dir)
        
        # Define Dataloader
        kwargs = {'num_workers': self.args.gpus, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(self.args, **kwargs)
        self.nclass = self.args.num_classes

        # Define network
        model = generate_net(self.args)
        train_params = [{'params': model.get_conv_weight_params(), 'lr': self.args.lr,'weight_decay':self.args.weight_decay},
                        {'params': model.get_conv_bias_params(), 'lr': self.args.lr * 2,'weight_decay':0}]

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
        if self.args.use_balanced_weights:
            classes_weights_path = os.path.join(self.args.save_dir, self.args.dataset,'classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(self.args.save_dir,self.args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32)).type(torch.FloatTensor)
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=self.args.cuda,foreloss_weight=args.foreloss_weight,seloss_weight=args.seloss_weight).build_loss(mode=self.args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        self.evaluator_inner = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr,
                                            self.args.epochs, len(self.train_loader))
                            
        
        self.model = self.model.cuda()
        # Resuming checkpoint
        self.args.start_epoch = 0
        self.best_pred = 0.0
        if self.args.resume is not None:
            optimizer,start_epoch,best_pred = load_pretrained_mode(self.model,checkpoint_path=self.args.resume)
            if not self.args.ft and optimizer is not None:
                self.optimizer.load_state_dict(optimizer)
                self.args.start_epoch = start_epoch
                self.best_pred = best_pred
        # Using cuda
        if self.args.cuda:
            self.model = torch.nn.DataParallel(self.model)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        num_img_tr = len(self.train_loader)
        self.evaluator.reset()
        self.evaluator_inner.reset()
        print('Training')
        start_time = time.time()
        for i,sample in enumerate(self.train_loader):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            current_lr = self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss,output = self.criterion(output,target)
            pred = output.data.clone()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            pred = pred.data.cpu().numpy()
            target_array = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator_inner.add_batch(target_array,pred)
            self.evaluator.add_batch(target_array,pred)
            if i % 10 == 0:
                Acc_train = self.evaluator_inner.Pixel_Accuracy()
                Acc_class_train = self.evaluator_inner.Pixel_Accuracy_Class()
                mIoU_train,IoU_train = self.evaluator_inner.Mean_Intersection_over_Union()
                FWIoU_train = self.evaluator_inner.Frequency_Weighted_Intersection_over_Union()
                print('\n===>Iteration  %d/%d    learning_rate: %.6f   metric:' % (i,num_img_tr,current_lr))
                print('=>Train loss: %.4f    acc: %.4f     m_acc: %.4f     miou: %.4f     fwiou: %.4f' % (loss.item(),
                                                                                    Acc_train,Acc_class_train,mIoU_train,FWIoU_train))
                print("IoU per class: ",IoU_train)
                self.evaluator_inner.reset()
            
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if num_img_tr > 10:
                if i % (num_img_tr // 10) == 0:
                    global_step = i + num_img_tr * epoch
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
            else:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
        Acc_train_epoch = self.evaluator.Pixel_Accuracy()
        Acc_class_train_epoch = self.evaluator.Pixel_Accuracy_Class()
        mIoU_train_epoch,IoU_train_epoch = self.evaluator.Mean_Intersection_over_Union()
        FWIoU_train_epoch = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        stop_time = time.time()
        self.writer.add_scalar('train/total_loss_epoch', train_loss/num_img_tr, epoch)
        print('=====>[Epoch: %d, numImages: %5d   time_consuming: %d]' % 
        (epoch, num_img_tr * self.args.batch_size,stop_time-start_time))
        print("Loss: %.3f  Acc: %.4f,  Acc_class: %.4f,  mIoU: %.4f,  fwIoU: %.4f\n\n" % (train_loss/(num_img_tr),
                Acc_train_epoch, Acc_class_train_epoch, mIoU_train_epoch, FWIoU_train_epoch))
        print("IoU per class: ",IoU_train_epoch)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        test_loss = 0.0
        print('\nValidation')
        num_img_tr = len(self.val_loader)
        start_time = time.time()
        for i, sample in enumerate(self.val_loader):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                loss,output = self.criterion(output, target)
            test_loss += loss.item()
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
        stop_time = time.time()
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU,IoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss/num_img_tr, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('=====>[Epoch: %d, numImages: %5d   previous best=%.4f    time_consuming: %d]' % (epoch, num_img_tr * self.args.gpus,self.best_pred,(stop_time-start_time)))
        print("Loss: %.3f  Acc: %.4f,  Acc_class: %.4f,  mIoU: %.4f,  fwIoU: %.4f\n\n" % (test_loss/(num_img_tr),Acc, Acc_class, mIoU, FWIoU))
        print("IoU per class: ",IoU)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        else:
            is_best = False
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': new_pred,
        }, is_best)

def main():
    from utils import parse_args
    args = parse_args.parse()
    if args.test_size is None:
        args.test_size = args.input_size
    args.sync_bn = True
    args.cuda = True
    args.gpus = torch.cuda.device_count()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
