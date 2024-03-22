# The official web of the 3SHNet project. 
[《3SHNet: Boosting Image-Sentence Retrieval via Visual Semantic-Spatial Self-Highlighting》]([https://arxiv.org/abs/2003.08813](https://www.sciencedirect.com/science/article/pii/S0306457324000761)])

by Xuri Ge*, Songpei Xu*, Fuhai Chen#, Jie Wang, Guoxin Wang, Shan An, Joemon M. Jose#

Information Processing and Management (IP&M 2024)

## Introduction
In this paper, we propose a novel visual \textbf{S}emantic-\textbf{S}patial \textbf{S}elf-\textbf{H}ighlighting \textbf{Net}work (termed \textbf{\textit{3SHNet}}) for high-precision, high-efficiency and high-generalization image-sentence retrieval. 3SHNet highlights the salient identification of prominent objects and their spatial locations within the visual modality, thus allowing the integration of visual semantics-spatial interactions and maintaining independence between two modalities. This integration effectively combines object regions with the corresponding semantic and position layouts derived from segmentation to enhance the visual representation. And the modality-independence guarantees efficiency and generalization. Additionally, 3SHNet utilizes the structured contextual visual scene information from segmentation to conduct the local (region-based) or global (grid-based) guidance and achieve accurate hybrid-level retrieval. Extensive experiments conducted on MS-COCO and Flickr30K benchmarks substantiate the superior performances, inference efficiency and generalization of the proposed 3SHNet when juxtaposed with contemporary state-of-the-art methodologies. Specifically, on the larger MS-COCO 5K test set, we achieve 16.3\%, 24.8\%, and 18.3\% improvements in terms of rSum score, respectively, compared with the state-of-the-art methods using different image representations, while maintaining optimal retrieval efficiency.


## Prerequisites
Basic env: python=3.7; pytorch=1.8.0(cuda11); tensorflow=2.11.0; tensorboard, etc. You can directly run:
```
conda env create -n 3SHNet --file env.yaml
```

## Data Preparation
To run the code, annotations and region-level and global-level image features with corresponding segmentation results for the MSCOCO and Flickr30K datasets are needed.

First, the basic annotations and region-level image features can be downloaded from [SCN](https://github.com/kuanghuei/SCAN#download-data).

For the global-level image features, we implemented the pre-trained REXNet-101 in [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa) to extract all images in MSCOCO and Flickr30K. After that, we stored them in independent .npy files (mainly for MScoco due to the large images).

About segmentation extractions, features are computed with the code provided by [UPSNet](https://github.com/uber-research/UPSNet). They include three types, i.e. segmentation semantic features (`%s_segmentations.npy`, dims=(N,133,7,7)), segmentation maps (`%s_seg_maps.npy`, dims=(N, 64, 64), downsampleing to reduce the calculations) and category-one-hot(`%s_cat_onehots.npy`, dims=(N,133) a little different from paper, which is embedded by a linear layer, it doesn't influence the conclusion).
Here we provided the segmentation results of the [test set](https://drive.google.com/drive/folders/1lU3I7J8XIquCtQ2lIML9w_9tDdRZWXm4?usp=drive_link) as examples, which can also be used to obtain our reported results.

## Training
We separate the global (grid-based) and local (region-based) training processes.
Training the region-level based 3SHNet model please run ` train_rgn_seg_sp_se_coco.sh`  under the main folder to start training:
```
sh train_rgn_seg_sp_se_coco.sh
```
Training the global-level based 3SHNet model please run ` train_rgn_seg_sp_se_coco.sh`  under the main folder to start training:
```
sh train_grid_seg_sp_se_coco.sh
```

Similar training process can be applied to Flickr30K.

## Testing the model
To test the trained models, you can directly run the eval scripts as:
```
sh eval_rgn_seg_sp_se_coco.sh
```
or
```
sh eval_gird_seg_sp_se_coco.sh
```
And to obtain the ensemble results, you can refer to the code in `eval_ensemble.py`
And to obtain the cross-dataset testing results, you can refer to the code in `eval_cross_ensemble.py`

To obtain the reported results, we have released the single region-level and grid-level pre-trained models in [Google Drive](https://drive.google.com/drive/folders/1lU3I7J8XIquCtQ2lIML9w_9tDdRZWXm4?usp=drive_link).
You should modify the pre-trained model paths in the evaluation codes and then follow the above testing processes.
To ensure reproducibility, we ran the code again and got similar or even higher results than reported in the paper!



## Citation
```
  @article{ge20243shnet,
  title={3SHNet: Boosting image-sentence retrieval via visual semantic-spatial self-highlighting},
  author={Ge, Xuri and Xu, Songpei and Chen, Fuhai and Wang, Jie and Wang, Guoxin and An, Shan and Jose, Joemon M},
  journal={Information Processing and Management},
  year={2024},
  publisher={Elsevier}
  }
```
Acknowledgement: We referred to the implementations of [vse_infty](https://github.com/woodfrog/vse_infty) and [SCAN](https://github.com/kuanghuei/SCAN) to build up our codebase.
