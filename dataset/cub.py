import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import pandas as pd

from collections import defaultdict
from itertools import chain
import os
from PIL import Image
import json
from tqdm import tqdm

import sys
sys.path.append('..')
from lsl.birds.data import lang_utils
import torchfile
import glob

class CUB2011(Dataset):
    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None, vocab=None):
        super(Dataset, self).__init__()
        self.args = args
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        self.mean = [0.4856074, 0.4994159, 0.4323767]
        self.std  = [0.1817421, 0.1811020, 0.1927458]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain
        self.lsl = args.lsl
        self.lang_dir = args.lang_dir

        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        with open(os.path.join(self.data_root, 'fsl_splits', partition + '.json'), 'rb') as f:
            self.data = json.load(f)

        self.imgs = np.load(os.path.join(self.data_root, 'fsl_splits', partition + '.npy'))

        labels = self.data['image_labels']
        # adjust sparse labels to labels from 0 to n.
        cur_class = 0
        label2label = {}
        for idx, label in enumerate(labels):
            if label not in label2label:
                label2label[label] = cur_class
                cur_class += 1
        new_labels = []
        for idx, label in enumerate(labels):
            new_labels.append(label2label[label])
        self.labels = new_labels

        # pre-process for contrastive sampling
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            self.labels = np.asarray(self.labels)
            self.labels = self.labels - np.min(self.labels)
            num_classes = np.max(self.labels) + 1

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                self.cls_positive[self.labels[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

        if self.args.lsl:
            # Load language and mapping from image names -> lang idx
            self.lang = {}           # items shape Nx10x32 (N varies, checked a few and saw 60, 60, 58)
            self.lang_lengths = {}   # items are shape Nx10
            self.lang_masks = {}     # items shape Nx10x32
            self.image_name_idx = {} # 11788 keys (len of CUB dataset)
            for cln, label_name in enumerate(self.data["label_names"]):
                # Use the numeric class id instead of label name due to
                # inconsistencies
                digits = label_name.split(".")[0]
                matching_names = [
                    x
                    for x in os.listdir(os.path.join(self.lang_dir, "word_c10"))
                    if x.startswith(digits)
                ]
                assert len(matching_names) == 1, matching_names
                label_file = os.path.join(self.lang_dir, "word_c10", matching_names[0])
                lang_tensor = torch.from_numpy(torchfile.load(label_file)).long()
                # Make words last dim
                lang_tensor = lang_tensor.transpose(2, 1)
                lang_tensor = lang_tensor - 1  # XXX: Decrement language by 1 upon load

                if (
                    self.args.language_filter == "color"
                    or self.args.language_filter == "nocolor"
                ):
                    lang_tensor = lang_utils.filter_language(
                        lang_tensor, self.args.language_filter, vocab
                    )

                if self.args.shuffle_lang:
                    lang_tensor = lang_utils.shuffle_language(lang_tensor)

                lang_lengths = lang_utils.get_lang_lengths(lang_tensor)

                # Add start and end of sentence tokens to language
                lang_tensor, lang_lengths = lang_utils.add_sos_eos(
                    lang_tensor, lang_lengths, vocab
                )
                lang_masks = lang_utils.get_lang_masks(
                    lang_lengths, max_len=lang_tensor.shape[2]
                )

                self.lang[label_name] = lang_tensor
                self.lang_lengths[label_name] = lang_lengths
                self.lang_masks[label_name] = lang_masks

                # Give images their numeric ids according to alphabetical order
                img_dir = os.path.join(self.lang_dir, "text_c10", label_name, "*.txt")
                sorted_imgs = sorted(
                    [
                    os.path.splitext(os.path.basename(i))[0]
                        for i in glob.glob(img_dir)
                    ]
                )
                for i, img_fname in enumerate(sorted_imgs):
                    self.image_name_idx[img_fname] = i

            self.captions = []
            self.lengths  = []
            self.masks    = []
            for x, y in zip(self.data["image_names"], self.data["image_labels"]):
                if y in label2label.keys(): # is this ever false?
                    #self.sub_meta[y].append(x) # image filenames, under each class label key
                    label_name = self.data["label_names"][y] # string classname

                    image_basename = os.path.splitext(os.path.basename(x))[0]
                    image_lang_idx = self.image_name_idx[image_basename]

                    captions = self.lang[label_name][image_lang_idx]
                    lengths = self.lang_lengths[label_name][image_lang_idx]
                    masks = self.lang_masks[label_name][image_lang_idx]

                    self.captions.append(captions)
                    self.lengths.append(lengths)
                    self.masks.append(masks)

                else:
                    breakpoint()
                    assert self.max_class is not None


    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)
        target = self.labels[item] - min(self.labels)


        if self.lsl:
            caption = self.captions[item]
            length  = self.lengths[item]
            mask    = self.masks[item]

            # sample one random caption out of the 10
            rand_idx = np.random.randint(len(caption))
            caption = caption[rand_idx]
            length  = length[rand_idx]
            mask    = mask[rand_idx]

            return img, target, item, (caption, length, mask)
        elif not self.is_sample:
            return img, target, item
        else:
            pos_idx = item
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, item, sample_idx


    def __len__(self):
        return len(self.labels)




class MetaCUB2011(CUB2011):
    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True,
                 vocab=None):
        super(MetaCUB2011, self).__init__(args, partition, False, vocab=vocab)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples
        if train_transform is None:
            self.train_transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(len(self.imgs)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way, ))

        support_xs = support_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1, )), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs



if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 1
    args.n_queries = 12
    args.data_root = '/shared/spetryk/data/rfs/CUB_200_2011'
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    args.lsl = True
    args.lang_dir = '/home/spetryk/rfs/lsl/birds/reed-birds'
    args.language_filter = 'all'
    args.shuffle_lang = False
    vocab = lang_utils.load_vocab(args.lang_dir)
    cub_dataset = CUB2011(args, 'val', vocab=vocab)
    print(len(cub_dataset))
    print(cub_dataset.__getitem__(500)[0].shape)

    metacub = MetaCUB2011(args, vocab=vocab)
    print(len(metacub))
    print(metacub.__getitem__(500)[0].size())
    print(metacub.__getitem__(500)[0].dtype)
    print(metacub.__getitem__(500)[1].shape)
    print(metacub.__getitem__(500)[2].size())
    print(metacub.__getitem__(500)[2].dtype)
    print(metacub.__getitem__(500)[3].shape)
    breakpoint()



