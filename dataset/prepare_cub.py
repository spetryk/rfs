import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import pickle

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser()

    parser.add_argument('--datadir', type=str, help='Path to CUB data folder')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--size', type=int, default=84, help='Image size (for square)')

    args = parser.parse_args()

    random = np.random.RandomState(args.seed)

    #filelist_path = './filelists/CUB/'
    #data_path = 'CUB_200_2011/images'
    data_path = os.path.join(args.datadir, 'images')
    dataset_list = ['train', 'val', 'test']

    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort()
    label_dict = dict(zip(folder_list, range(0, len(folder_list))))

    classfile_list_all = []

    # Load attributes
    attrs = pd.read_csv(os.path.join(args.datadir, 'attributes/image_attribute_labels.txt'),
                        sep=' ',
                        header=None,
                        names=['image_id', 'attribute_id', 'is_present', 'certainty_id', 'time'])
    # Zero out attributes with certainty < 3
    attrs['is_present'] = np.where(attrs['certainty_id'] < 3, 0, attrs['is_present'])
    # Get image names
    image_names = pd.read_csv(os.path.join(args.datadir, 'images.txt'), sep=' ',
                              header=None,
                              names=['image_id', 'image_name'])
    attrs = attrs.merge(image_names, on='image_id')
    attrs['is_present'] = attrs['is_present'].astype(str)
    attrs = attrs.groupby('image_name')['is_present'].apply(lambda col: ''.join(col))
    attrs = dict(zip(attrs.index, attrs))
    attrs = {os.path.basename(k): v for k, v in attrs.items()}

    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list_all.append([
            join(args.datadir, folder_path, cf) for cf in listdir(folder_path)
            if (isfile(join(folder_path, cf)) and cf[0] != '.' and not cf.endswith('.npz'))
        ])
        random.shuffle(classfile_list_all[i])

    for dataset in dataset_list:
        file_list = []
        label_list = []
        for i, classfile_list in enumerate(classfile_list_all):
            if 'train' in dataset:
                if (i % 2 == 0):
                    file_list.extend(classfile_list)
                    label_list.extend(np.repeat(
                        i, len(classfile_list)).tolist())
            if 'val' in dataset:
                if (i % 4 == 1):
                    file_list.extend(classfile_list)
                    label_list.extend(np.repeat(
                        i, len(classfile_list)).tolist())
            if 'test' in dataset:
                if (i % 4 == 3):
                    file_list.extend(classfile_list)
                    label_list.extend(np.repeat(
                        i, len(classfile_list)).tolist())

        # Get attributes
        attribute_list = [
            attrs[os.path.basename(f)] for f in file_list if not f.endswith('.npz')
        ]

        djson = {
            'label_names': folder_list,
            'image_names': file_list,
            'image_labels': label_list,
            'image_attributes': attribute_list,
        }

        os.makedirs(os.path.join(args.datadir, 'fsl_splits'), exist_ok=True)
        with open(os.path.join(args.datadir, 'fsl_splits', dataset + '.json'), 'w') as fout:
            json.dump(djson, fout)

        imgs = []
        transform = transforms.Resize((args.size, args.size))
        for filename in tqdm(file_list):
            img = Image.open(filename).convert('RGB')
            img = transform(img)
            img = np.asarray(img)
            imgs.append(img)

        imgs = np.stack(imgs)
        print(imgs.shape)

        np.save(os.path.join(args.datadir, 'fsl_splits', dataset + '.npy'), imgs)

        print("%s -OK" % dataset)


    # for bird_class in tqdm(os.listdir(data_path), desc="Classes"):
    #     bird_imgs_np = {}
    #     class_dir = os.path.join(data_path, bird_class)
    #     bird_imgs = sorted([x for x in os.listdir(class_dir) if x != "img.npz"])
    #     for bird_img in bird_imgs:
    #         bird_img_fname = os.path.join(class_dir, bird_img)
    #         img = Image.open(bird_img_fname).convert("RGB")
    #         img_np = np.asarray(img)

    #         full_bird_img_fname = os.path.join(
    #             data_path, bird_class, bird_img
    #         )

    #         bird_imgs_np[full_bird_img_fname] = img_np

    #     np_fname = os.path.join(class_dir, "img.npz")
    #     np.savez_compressed(np_fname, **bird_imgs_np)


