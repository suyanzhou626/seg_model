import cv2
import numpy as np
import sys   
sys.setrecursionlimit(500000)

def blur(logit,kernel_size=3):
    new_logit = cv2.blur(logit,(kernel_size,kernel_size),anchor=(kernel_size-1,kernel_size-1))
    return new_logit

def _doflooding(classmap,i,j,width,height,compid,classid,compmap,compsize):
    if (i < 0 or i > (height -1) or j < 0 or j > (width -1)):
        return
    if compmap[i,j] != -1:
        return
    if classmap[i,j] != classid:
        return
    compmap[i,j] = compid
    compsize['cnt'] += 1

    _doflooding(classmap,i - 1,j,width,height,compid,classid,compmap,compsize)
    _doflooding(classmap,i,j + 1,width,height,compid,classid,compmap,compsize)
    _doflooding(classmap,i + 1,j,width,height,compid,classid,compmap,compsize)
    _doflooding(classmap,i,j-1,width,height,compid,classid,compmap,compsize)

def removehole(logit,m_hole_ratio):
    assert(len(logit.shape) == 3)
    height,width = logit.shape[0:2]
    classmap = np.argmax(logit,axis=2)
    compmap = np.zeros_like(classmap)
    compmap.fill(-1)
    compid_to_classid = {}

    compid = 0
    for i in range(height):
        for j in range(width):
            if compmap[i,j] != -1:
                continue
            compsize = {'cnt':0}
            classid = classmap[i,j]
            _doflooding(classmap,i,j,width,height,compid,classid,compmap,compsize)

            to_class = classmap[i,j]
            if compsize['cnt'] < int(m_hole_ratio * width * height):
                if i > 0:
                    to_class = classmap[i-1,j]
                elif j > 0:
                    to_class = classmap[i,j-1]
            compid_to_classid[compid] = to_class
            compid += 1
    
    for i in range(height):
        for j in range(width):
            compid_temp = compmap[i,j]
            from_class = classmap[i,j]
            to_class_temp = compid_to_classid[compid_temp]
            temp = logit[i,j,from_class]
            logit[i,j,from_class] = logit[i,j,to_class_temp]
            logit[i,j,to_class_temp] = temp   
    
    return logit

class Deflicker():
    def __init__(self,m_score_diff_threshold):
        super().__init__()
        self.threshold = m_score_diff_threshold
        self.pre_scoremap = np.zeros((1,1))

    def __call__(self,logit):
        assert(len(logit.shape)==3)
        if self.pre_scoremap.shape != logit.shape:
            self.pre_scoremap = logit.copy()
            return logit
        else:
            conf_map_temp = np.sort(logit,axis=2)
            conf_map = conf_map_temp[:,:,-1].copy() - conf_map_temp[:,:,-2].copy()
            # conf_map = np.stack([conf_map]*logit.shape[-1],axis=-1)
            logit[conf_map <= self.threshold] = self.pre_scoremap[conf_map <= self.threshold].copy()
            self.pre_scoremap[conf_map > self.threshold] = logit[conf_map > self.threshold].copy()
            return logit



if __name__ == "__main__":
    test_mask = np.random.randint(0,high=2,size=(20,20))
    test_mask = test_mask.astype(np.uint8)
    print(test_mask)
    processed = bluranderode(test_mask)
    print(processed)