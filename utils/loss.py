import torch
import torch.nn as nn

def get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = torch.Tensor(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.loss_weight1 = 1
        self.loss_weight2 = 1

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                            reduction='elementwise_mean' if self.size_average else 'sum')
        if self.cuda:
            criterion = criterion.cuda()
        if not isinstance(logit,list):
            # n, c,h,w = logit.size()
            logit = nn.functional.interpolate(logit,size=target.data.size()[1:],mode='bilinear',align_corners=True)
            loss = criterion(logit, target.long())
            return loss,logit
        elif len(logit) == 2:
            pred1,pred2 = tuple(logit)
            pred1 = nn.functional.interpolate(pred1,size=target.data.size()[1:],mode='bilinear',align_corners=True)
            target2 = torch.clone(target)
            target2 = torch.unsqueeze(target2,1).float()
            target2 = nn.functional.interpolate(target2,size=pred2.data.size()[2:],mode='bilinear',align_corners=True)
            target2 = torch.squeeze(target2,1).long()
            target2[target2>1] = 1
            loss1 = criterion(pred1,target)
            loss2 = criterion(pred2,target2)
            loss = loss1 + self.loss_weight1*loss2
            return loss,pred1
        elif len(logit) == 3:
            semantic_pred,se_pred,fore_pred = tuple(logit)
            semantic_pred = nn.functional.interpolate(semantic_pred,size=target.data.size()[1:],mode='bilinear',align_corners=True)
            # print(semantic_pred,semantic_pred.size())
            # print(fore_pred,fore_pred.size())
            # print(se_pred,se_pred.size())
            n_classes = semantic_pred.size()[1]
            target2 = torch.clone(target)
            target2 = torch.unsqueeze(target2,1).float()
            target2 = nn.functional.interpolate(target2,size=fore_pred.data.size()[2:],mode='bilinear',align_corners=True)
            target2 = torch.squeeze(target2,1).long()
            target2[target2>1] = 1
            se_target = get_batch_label_vector(target,n_classes).type_as(se_pred)
            target = target.long()
            # print(target,target.size())
            # print(target2,target2.size())
            # print(se_target,se_target.size())
            loss1 = criterion(semantic_pred,target)
            loss2 = criterion(fore_pred,target2)
            se_loss = nn.BCELoss(self.weight,reduction='elementwise_mean' if self.size_average else 'sum')(torch.sigmoid(se_pred),se_target)
            loss = loss1 + self.loss_weight1*loss2 + self.loss_weight2*se_loss
            return loss,semantic_pred
        # if self.batch_average:
        #     loss /= n

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c,h,w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='elementwise_mean' if self.size_average else 'sum')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3,2,2).cuda()
    c = torch.rand(1, 2,2,2).cuda()
    d = torch.rand(1,3).cuda()
    pred = [a,d,c]
    b = torch.argmax(a,dim=1)
    print(a.type())
    print(b.type())
    print(loss.CrossEntropyLoss(pred, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




