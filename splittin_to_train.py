import os
import numpy as np
import shutil
import random

# Creating dataset with disturbition: Train(%70),,Validation(%15),Test(%15)
# # Creating Train / Val / Test folders (One time use)
root_dir = 'Categories'
classes_dir = ['/akiec', '/bcc', '/bkl', '/df','/mel','/nv','/vasc']

val_ratio = 0.15
test_ratio = 0.15

for cls in classes_dir:
    if not os.path.exists(root_dir +'/train' + cls):
        os.makedirs(root_dir +'/train' + cls)
        os.makedirs(root_dir +'/val' + cls)
        os.makedirs(root_dir +'/test' + cls)


    # Creating partitions of the data after shuffeling
    src = root_dir + cls # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])

    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir +'/train' + cls)

    for name in val_FileNames:
        shutil.copy(name, root_dir +'/val' + cls)

    for name in test_FileNames:
        shutil.copy(name, root_dir +'/test' + cls)
        