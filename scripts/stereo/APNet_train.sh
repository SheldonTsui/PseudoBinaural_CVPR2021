gpus=${1}
split_id=${2}
exp_name=APNet

CUDA_VISIBLE_DEVICES=${gpus} python train.py --name ${exp_name}_${split_id} \
    --splitPath ./new_splits/split${split_id} \
    --save_epoch_freq 50 \
    --display_freq 10 \
    --save_latest_freq 100 \
    --batchSize 192\
    --learning_rate_decrease_itr 5 \
    --niter 400 \
    --lr_visual 0.0001 \
    --lr_audio 0.001 \
    --nThreads 32 \
    --gpu_ids 0,1 \
    --validation_on \
    --validation_freq 50 \
    --validation_batches 50 \
    --fusion_model APNet \
    --val_return_key stereo_loss_fusion \
    --checkpoints_dir checkpoints/stereo \
    --tensorboard True |& tee -a logs/${exp_name}_${split_id}.log
