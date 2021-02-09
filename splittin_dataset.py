import pandas as pd
from shutil import copy
import os

name_file1 ="HAM10000_images_part_1"
name_file2="HAM10000_images_part_2"
df = pd.read_csv('HAM10000_metadata.csv')
all_images_1 = os.listdir('HAM10000_images_part_1')
all_images_2 = os.listdir('HAM10000_images_part_2')

def creating_labeled_dataset(all_images,name_file):
    if not os.path.exists('categories'):
        os.mkdir('categories')
    k = 0
    for image in all_images:
        print(image)
        can_type = df[df['image_id'] == image.replace(".jpg","")]['dx']
        can_type = str(list(can_type)[0])
        print(can_type)

        if not os.path.exists(os.path.join('categories', can_type)):
            os.mkdir(os.path.join('categories', can_type))

        path_from = os.path.join(name_file, image)
        path_to = os.path.join('categories', can_type, image)

        copy(path_from, path_to)
        print('Moved {} to {}'.format(image, path_to))
        k += 1

    print('Moved {} images.'.format(k))

creating_labeled_dataset(all_images_1,name_file1)
creating_labeled_dataset(all_images_2,name_file2)
