gpus=${1}
split_id=${2}
test_mode=${3}
exp_name=AudioNet

CUDA_VISIBLE_DEVICES=${gpus} python demo_stereo.py --input_audio_length 10 \
    --hop_size 0.1 \
    --weights_visual checkpoints/stereo/${exp_name}_${split_id}/visual_${test_mode}.pth \
    --weights_audio checkpoints/stereo/${exp_name}_${split_id}/audio_${test_mode}.pth \
    --output_dir_root eval_demo/stereo/${exp_name}_${split_id}_${test_mode} \
    --hdf5FolderPath ./dataset/cleaned_splits/split${split_id}/test.h5 
