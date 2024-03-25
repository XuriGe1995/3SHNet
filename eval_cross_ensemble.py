import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# # Evaluate model ensemble
# paths = ['save_results/coco_butd_rgn_seg_sp_Attention4_se_Attention4_fusion3_bert_01_results_coco.npy',
#          'save_results/coco_butd_rgn_seg_sp_Attention4_se_Attention4_fusion3_bert_03_results_coco.npy']

# evaluation.eval_ensemble(results_paths=paths, save_path = None, fold5=True)
# evaluation.eval_ensemble(results_paths=paths, save_path = None, fold5=False) #'save_results/coco_5k_butd_rgn_seg_sp_Attention4_se_Attention4_fusion3_ensemble.npy'

paths = ['save_results/Region_trainedcoco_valf30k_crossdataset_results_f30k.npy',
         'save_results/Region_trainedcoco_valf30k_crossdataset_results_01_f30k.npy']
evaluation.eval_ensemble(results_paths=paths, save_path = None, fold5=False)
#'save_results/f30k_butd_rgn_seg_sp_Attention4_se_Attention4_fusion3_ensemble.npy'
