gpus=${1}
split_id=${2}
test_mode=${3}
exp_name=Augment_AudioNet_sepstereo_blending

CUDA_VISIBLE_DEVICES=${gpus} python demo_stereo.py --input_audio_length 10 \
    --hop_size 0.1 \
    --norm_mode in \
    --weights_visual checkpoints/sepstereo_Augment/${exp_name}_${split_id}/visual_${test_mode}.pth \
    --weights_audio checkpoints/sepstereo_Augment/${exp_name}_${split_id}/audio_${test_mode}.pth \
    --output_dir_root eval_demo/sepstereo_Augment/${exp_name}_${split_id}_${test_mode} \
    --splitPath ./dataset/cleaned_splits/split${split_id}/test.txt
