# seg_model
a general construct using different network to train and test

there are two branches including pytorch_train and linklink_train respectively used in official pytorch and linklink enviroment.

the utils folder have all function necessary for trainging

the modeling folder contains all network code and you can add any network you need only modify generatenet.py

the dataloaders folder implements a general dataset 

link_train.py is the training file generally and vis.py is used to eval val set (the groud truth is necessary)

if you want to use this framwork, just copy a shell file included in experiment and modify corresponding parameters

here are some necessary parameters(and the details of other optional parameters can be seen in the link_train.py):
    backbone: choose what network you want to use in the modeling
    dataset: the name of the dataset you use in this experiment
    data_dir: the path of the dataset used which combining the *list.txt can obtain the complete path
    train_list/val_list: the txt files save the name of images and labels in the dataset
    input_size: the size of the image the network accept
    shrink: the param to control the output size of resize operation with the purpose of general using
    num_classes: the channels of the output layers
    epochs: how many epochs you want to train
    batch_size: how many image you want to use in one iteration