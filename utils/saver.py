import os
import shutil
import torch
import glob

class Saver(object):

    def __init__(self, args):
        self.directory = os.path.join(args.save_dir, args.dataset, args.backbone)
        runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir,exist_ok=True)
        print('experiment_{}'.format(str(run_id)))

        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = args.__dict__
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk
            is_best: help to judge if it's the best state,if true,backup to the directory
        """
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            filename_best = os.path.join(self.experiment_dir,'best.pth.tar')
            torch.save(state,filename_best)
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