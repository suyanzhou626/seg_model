import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(args.save_dir, args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir,exist_ok=True)
        print('experiment_{}'.format(str(run_id)))

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk
            is_best: help to judge if it's the best state,if true,backup to the directory
        """
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if not os.path.exists(os.path.join(self.directory,'best_pred.txt')):
                with open(os.path.join(self.directory,'best_pred.txt'),'w') as f:
                    f.write(str(best_pred))
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                with open(os.path.join(self.directory,'best_pred.txt'),'r') as f:
                    max_iou = float(f.readline())
                if best_pred > max_iou:
                    with open(os.path.join(self.directory,'best_pred.txt'),'w') as f:
                        f.write(str(best_pred))
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            # if self.runs:
            #     previous_miou = [0.0]
            #     for run in self.runs:
            #         run_id = run.split('_')[-1]
            #         path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
            #         if os.path.exists(path):
            #             with open(path, 'r') as f:
            #                 miou = float(f.readline())
            #                 previous_miou.append(miou)
            #         else:
            #             continue
            #     max_miou = max(previous_miou)
            #     if best_pred > max_miou:
            #         shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            # else:
            #     shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['backbone'] = self.args.backbone
        p['datset'] = self.args.dataset
        p['lr'] = self.args.lr
        p['optim'] = self.args.optim_method
        p['nesterov'] = self.args.nesterov
        p['momentum'] = self.args.momentum
        p['weight_decay'] = self.args.weight_decay
        p['fine tune'] = str(self.args.ft)
        p['nclasses'] = self.args.num_classes
        p['batchsize'] = self.args.batch_size
        p['class_balance'] = str(self.args.use_balanced_weights)
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['sync_bn'] = str(self.args.sync_bn)
        p['epoch'] = self.args.epochs
        p['crop_size'] = self.args.
        p['link'] = str(self.args.use_link)

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
