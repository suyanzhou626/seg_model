import cv2
import numpy as np

def bluranderode(pre_mask,blurlevel=0.02):
    assert(pre_mask.shape[0] > 1 and len(pre_mask.shape) >=2)
    mask_fore_num = pre_mask.sum(1).max()
    blur_size = max(3,int(mask_fore_num*blurlevel//10))
    erode_size = max(3,blur_size//2)
    blur_size = (blur_size,blur_size)
    new_mask = pre_mask.astype(np.uint8)
    for i in range(erode_size,2,-2):
        open_kernel = cv2.getStructuringElement(cv2.MORPH_OPEN,(i,i))
        new_mask = cv2.morphologyEx(new_mask,cv2.MORPH_OPEN,open_kernel)
    for i in range(erode_size,2,-2):
        close_kernel = cv2.getStructuringElement(cv2.MORPH_OPEN,(i,i))
        new_mask = cv2.morphologyEx(new_mask,cv2.MORPH_CLOSE,close_kernel)
    new_mask = cv2.blur(new_mask,blur_size,anchor=(blur_size-1,blur_size-1))
    return new_mask

class AverageFrame(object):
    def __init__(self,level=0.05):
        super().__init__()
        self.level = level
        self.pre_pred = np.zeros((1,1))

    def __call__(self,curr_mask):
        assert len(curr_mask.shape) == 2
        height,width = curr_mask.shape
        if self.pre_pred.shape != (height,width):
            self.pre_pred = curr_mask.copy()
            return curr_mask
        threshold = int(height * width * self.level)
        diffpixel = np.abs(self.pre_pred - curr_mask).sum()
        if diffpixel < threshold:
            new_mask = curr_mask.copy()
            new_mask[self.pre_pred != curr_mask] = 1
            self.pre_pred = new_mask.copy()
            return new_mask
        else:
            self.pre_pred = curr_mask.copy()
            return curr_mask

        


if __name__ == "__main__":
    test_mask = np.random.randint(0,high=2,size=(20,20))
    test_mask = test_mask.astype(np.uint8)
    print(test_mask)
    processed = bluranderode(test_mask)
    print(processed)