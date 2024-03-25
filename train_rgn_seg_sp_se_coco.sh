######### 
##To ensure reproducibility, we ran the code again and got similar or even higher results than reported in the paper!
#########
DATASET_NAME='coco'
DATA_PATH='./data/'

CUDA_VISIBLE_DEVICES=0 python train_rgn_seg_sp_se.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} \
  --logger_name runs/${DATASET_NAME}_butd_rgn_seg_sp_se_reproduced/log \
  --model_name runs/${DATASET_NAME}_butd_rgn_seg_sp_se_reproduced \
  --num_epochs=25 --lr_update=15 --learning_rate=.0005 --precomp_enc_type basic --workers 3 \
  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1 --batch_size 256
