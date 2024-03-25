import os
import argparse
import logging
from lib import evaluation

import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
 
default_collate_func = dataloader.default_collate
 

def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)
 
setattr(dataloader, 'default_collate', default_collate_override)



logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco',
                        help='coco or f30k')
    parser.add_argument('--data_path', default='/tmp/data/coco')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--evaluate_cxc', action='store_true')
    opt = parser.parse_args()

    if opt.dataset == 'coco':
        weights_bases = [
            'runs/coco_butd_region_bert',
#             'runs/release_weights/coco_butd_grid_bert',
#             'runs/release_weights/coco_wsl_grid_bert',
        ]
    elif opt.dataset == 'f30k':
        weights_bases = [
             'runs/f30k_butd_region_bert',
            #'runs/f30k_butd_grid_bert'
#             'runs/release_weights/f30k_butd_grid_bert',
#             'runs/release_weights/f30k_wsl_grid_bert',
        ]
    else:
        raise ValueError('Invalid dataset argument {}'.format(opt.dataset))

    for base in weights_bases:
        print(base)
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        if opt.save_results:  # Save the final results for computing ensemble results
            save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))
        else:
            save_path = None
#         print(opt.save_results)
#         opt.dataset = 'f30k'
#         print(opt.data_path)
        if opt.dataset == 'coco':
            if not opt.evaluate_cxc:
                print('xxxxxxxxxxx')
                # Evaluate COCO 5-fold 1K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
                # Evaluate COCO 5K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)
            else:
                print('yyyyyyyyyyyyyyyyy')
                # Evaluate COCO-trained models on CxC
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True, cxc=True)
        elif opt.dataset == 'f30k':
            # Evaluate Flickr30K
            print('zzzzzzzzzzzzzzzzzzzzzzz')
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)


if __name__ == '__main__':
    main()
