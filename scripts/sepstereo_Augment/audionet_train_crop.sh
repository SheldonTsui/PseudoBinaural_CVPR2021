gpus=${1}
split_id=${2}
exp_name=Augment_AudioNet_sepstereo_crop

CUDA_VISIBLE_DEVICES=${gpus} python train_diffusion.py --name ${exp_name}_${split_id} \
    --hdf5FolderPath ./dataset/cleaned_splits/crop/split${split_id} \
    --dataset_mode Augment_sepstereo \
    --datalist FAIR_data \
    --save_epoch_freq 20 \
    --display_freq 10 \
    --save_latest_freq 100 \
    --batchSize 256 \
    --learning_rate_decrease_itr 5 \
    --niter 300 \
    --lr_visual 5e-5 \
    --lr_audio 5e-4 \
    --nThreads 32 \
    --gpu_ids 0,1,2,3 \
    --validation_on \
    --validation_freq 50 \
    --validation_batches 50 \
    --audio_normal \
    --norm_mode in \
    --not_use_background \
    --val_return_key stereo_loss \
    --checkpoints_dir checkpoints/sepstereo_Augment \
    --tensorboard True |& tee -a logs/${exp_name}_${split_id}.log
