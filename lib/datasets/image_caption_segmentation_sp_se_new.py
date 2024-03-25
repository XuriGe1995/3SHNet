"""COCO dataset loader"""
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import os.path as osp
import numpy as np
from imageio import imread
import random
import json
import cv2
from collections import Counter
import logging

logger = logging.getLogger(__name__)



class PrecompRegionDataset(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, data_split, tokenizer, opt, train):
        self.tokenizer = tokenizer
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name

        self.base_target_size = 64
        
        if 'coco' in data_name:
#             print(data_name)
#             loc_cap = osp.join('../Data/data',data_name+'_precomp')
#             loc_image = osp.join('../Data/data',data_name+'_precomp')
            loc_cap = '../Data/data/coco_precomp'
            loc_image = '../Data/data/coco_precomp'
        else:
            loc_cap = '../Data/data/f30k_precomp'
            loc_image = '../Data/data/f30k_precomp'
        # Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        # Image features
        if 'coco'in data_name:
            self.images = []
            
            self.image_segmentations = np.load(os.path.join(loc_image, '%s_segmentations.npy' % data_split))
            self.image_segmaps = np.load(os.path.join(loc_image, '%s_seg_maps.npy' % data_split))
            self.image_cat_onehots = np.load(os.path.join(loc_image, '%s_cat_onehots.npy' % data_split))
            
            ## image_IDs
            self.image_ids = []
            with open(osp.join(loc_cap, '%s_ids.txt' % data_split), 'r') as f:
                for line in f:
                    self.image_ids.append(line.strip())
            self.length = len(self.captions)
            # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
            if len(self.image_ids) != self.length:
                self.im_div = 5
            else:
                self.im_div = 1
        else:
            self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % data_split))
            print(os.path.join(loc_image, '%s_ims.npy' % data_split))
            self.image_segmentations = np.load(os.path.join(loc_image, '%s_segmentations.npy' % data_split))
            self.image_segmaps = np.load(os.path.join(loc_image, '%s_seg_maps.npy' % data_split))
            self.image_cat_onehots = np.load(os.path.join(loc_image, '%s_cat_onehots.npy' % data_split))
            # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
            num_images = len(self.images)
            self.length = len(self.captions)
            if num_images != self.length:
                self.im_div = 5
            else:
                self.im_div = 1
        
        
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_index = index // self.im_div
        caption = self.captions[index]
        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)

        # Convert caption (string) to word ids (with Size Augmentation at training time)
        target = process_caption(self.tokenizer, caption_tokens, self.train)
        if len(self.images) > 0:
            image = self.images[img_index]
            Unet_seg = self.image_segmentations[img_index]
            segmentation = self.image_segmaps[img_index]
            cat_onehot = self.image_cat_onehots[img_index]
        else:
            img_pth = self.image_ids[img_index]
#             print(img_pth)
            image = torch.Tensor(np.load('/nfs/Image-text-matching/Data/data/coco_precomp/coco_region_features/' + img_pth +'.npy'))
            Unet_seg = self.image_segmentations[img_index]
            segmentation = self.image_segmaps[img_index]
            cat_onehot = self.image_cat_onehots[img_index]
#             segmentation = np.load('/nfs/Image-text-matching/Data/data/coco_precomp/coco_seg_results/'+ img_pth + '.npy')
    
        if self.train:  # Size augmentation for region feature
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image = image[np.where(rand_list > 0.20)]
        image = torch.Tensor(image)
        
        ## two types of segmentation results
        

        segmentation = torch.Tensor(segmentation) ## shape like: 64*64
        cat_onehot = torch.Tensor(cat_onehot)
#         segmentation = torch.Tensor(segmentation) ## shape like: 460*680
        
        Unet_seg = torch.Tensor(Unet_seg) ## shape like: 
        
        return image, segmentation, cat_onehot, Unet_seg, target, index, img_index

    def __len__(self):
        return self.length
    

    def mask_to_onehot(self, seg, num_classes=133):
        _mask = [seg==i for i in range(num_classes)]
        return np.array(_mask).astype('float32')
    def _seg_norm(self, im_in): 
        im_in -= 65.0
        return im_in

    @staticmethod
    def _hori_flip(im):
        im = np.fliplr(im).copy()
        return im

def process_caption(tokenizer, tokens, train=True):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)
    return target


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, segmentations, cat_onehots, Unet_segs, captions, ids, img_ids = zip(*data)
    if len(images[0].shape) == 2:  # region feature
        # Sort a data list by caption length
        # Merge images (convert tuple of 3D tensor to 4D tensor)
        # images = torch.stack(images, 0)
        img_lengths = [len(image) for image in images]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        img_lengths = torch.Tensor(img_lengths)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
            
        ## Segmentation results
        segmentations = torch.stack(segmentations, 0)
        cat_onehots = torch.stack(cat_onehots, 0)
        Unet_segs = torch.stack(Unet_segs, 0)

        return all_images, img_lengths, segmentations, cat_onehots, Unet_segs, targets, lengths, ids
    else:  # raw input image
        # Merge images (convert tuple of 3D tensor to 4D tensor)
        images = torch.stack(images, 0)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
            
        ## Segmentation results
        segmentations = torch.stack(segmentations, 0)
        Unet_segs = torch.stack(Unet_segs, 0)
        
        return images, segmentations, Unet_segs, targets, lengths, ids


def get_loader(data_path, data_name, data_split, tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        drop_last = True
    else:
        drop_last = False
    if opt.precomp_enc_type == 'basic':
        dset = PrecompRegionDataset(data_path, data_name, data_split, tokenizer, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
    else:
        dset = RawImageDataset(data_path, data_name, data_split, tokenizer, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn)
    return data_loader


def get_loaders(data_path, data_name, tokenizer, batch_size, workers, opt):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, True, workers)
    val_loader = get_loader(data_path, data_name, 'dev', tokenizer, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_train_loader(data_path, data_name, tokenizer, batch_size, workers, opt, shuffle):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, shuffle, workers)
 
    return train_loader


def get_test_loader(split_name, data_name, tokenizer, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, data_name, split_name, tokenizer, opt,
                             batch_size, False, workers, train=False)
    return test_loader
