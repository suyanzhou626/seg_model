import os
import numpy as np

def calculate_weigths_labels(save_dir,dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    print('Calculating classes weights')
    length = len(dataloader)
    for i,sample in enumerate(dataloader):
        if i % (length // 10) == 0:
            print('===>%d/%d' % (i,length))
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l

    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(save_dir,dataset,'classes_weights.npy')
    np.save(classes_weights_path, ret)

    return ret