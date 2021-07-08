import os
import librosa
import argparse
import numpy as np
import mmcv
import pdb
from math import pi
from numpy import linalg as LA
from scipy.signal import hilbert
from data.stereo_dataset import generate_spectrogram
import statistics as stat

def get_content(value_list, text):
    if len(value_list) == 1:
        content_res = "{}: {}".format(text, value_list[0])
    else:
        content_res = "{}: {}, {}, {}".format(
            text,
            stat.mean(value_list),
            stat.stdev(value_list),
            stat.stdev(value_list) / np.sqrt(len(value_list))
        )

    return content_res

def normalize(samples):
    return samples / np.maximum(1e-20, np.max(np.abs(samples)))

def STFT_L2_distance(predicted_spect_channel1, gt_spect_channel1, predicted_spect_channel2, gt_spect_channel2):
    #channel1
    real = np.expand_dims(np.real(predicted_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(predicted_spect_channel1), axis=0)
    predicted_realimag_channel1 = np.concatenate((real, imag), axis=0)
    real = np.expand_dims(np.real(gt_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(gt_spect_channel1), axis=0)
    gt_realimag_channel1 = np.concatenate((real, imag), axis=0)
    channel1_distance = np.mean(np.power((predicted_realimag_channel1 - gt_realimag_channel1), 2))

    #channel2
    real = np.expand_dims(np.real(predicted_spect_channel2), axis=0)
    imag = np.expand_dims(np.imag(predicted_spect_channel2), axis=0)
    predicted_realimag_channel2 = np.concatenate((real, imag), axis=0)
    real = np.expand_dims(np.real(gt_spect_channel2), axis=0)
    imag = np.expand_dims(np.imag(gt_spect_channel2), axis=0)
    gt_realimag_channel2 = np.concatenate((real, imag), axis=0)
    channel2_distance = np.mean(np.power((predicted_realimag_channel2 - gt_realimag_channel2), 2))

    #sum the distance between two channels
    stft_l2_distance = channel1_distance + channel2_distance
    return float(stft_l2_distance)

def Envelope_distance(predicted_binaural, gt_binaural):
    #channel1
    pred_env_channel1 = np.abs(hilbert(predicted_binaural[0,:]))
    gt_env_channel1 = np.abs(hilbert(gt_binaural[0,:]))
    channel1_distance = np.sqrt(np.mean((gt_env_channel1 - pred_env_channel1)**2))

    #channel2
    pred_env_channel2 = np.abs(hilbert(predicted_binaural[1,:]))
    gt_env_channel2 = np.abs(hilbert(gt_binaural[1,:]))
    channel2_distance = np.sqrt(np.mean((gt_env_channel2 - pred_env_channel2)**2))

    #sum the distance between two channels
    envelope_distance = channel1_distance + channel2_distance
    return float(envelope_distance)

def SNR(predicted_binaural, gt_binaural):
    mse_distance = np.mean(np.power((predicted_binaural - gt_binaural), 2))
    snr = 10. * np.log10((np.mean(gt_binaural**2) + 1e-4) / (mse_distance + 1e-4))

    return float(snr)

def Magnitude_distance(predicted_spect_channel1, gt_spect_channel1, predicted_spect_channel2, gt_spect_channel2):
    stft_mse1 = np.mean(np.power(np.abs(predicted_spect_channel1 - gt_spect_channel1), 2))
    stft_mse2 = np.mean(np.power(np.abs(predicted_spect_channel2 - gt_spect_channel2), 2))

    return float(stft_mse1 + stft_mse2)

def Angle_Diff_distance(predicted_binaural, gt_binaural):
    gt_diff = gt_binaural[0] - gt_binaural[1]
    pred_diff = predicted_binaural[0] - predicted_binaural[1]
    gt_diff_spec = librosa.core.stft(gt_diff, n_fft=512, hop_length=160, win_length=400, center=True)
    pred_diff_spec = librosa.core.stft(pred_diff, n_fft=512, hop_length=160, win_length=400, center=True)
    _, pred_diff_phase = librosa.magphase(pred_diff_spec)
    _, gt_diff_phase = librosa.magphase(gt_diff_spec)
    pred_diff_angle = np.angle(pred_diff_phase)
    gt_diff_angle = np.angle(gt_diff_phase)
    angle_diff_init_distance = np.abs(pred_diff_angle - gt_diff_angle)
    angle_diff_distance = np.mean(np.minimum(angle_diff_init_distance, np.clip(2 * pi - angle_diff_init_distance, a_min=0, a_max=2*pi))) 
    
    return float(angle_diff_distance)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_root', type=str, help="the demo path")
    parser.add_argument('--audio_sampling_rate', default=16000, type=int, help='audio sampling rate')
    parser.add_argument('--real_mono', default=False, type=bool, help='whether the input predicted binaural audio is mono audio')
    parser.add_argument('--normalization', default=True, type=bool)
    parser.add_argument('--audioNames_file', type=str, default='', help='audioNames file')
    args = parser.parse_args()
    stft_distance_list = []
    envelope_distance_list = []
    angle_distance_list = []
    snr_list = []
    magnitude_distance_list = []

    if len(args.audioNames_file) > 0:
        audioNames = mmcv.list_from_file(args.audioNames_file)
    elif args.results_root[-6:-4] == '00':
        audioNames = ['']
    else:
        audioNames = sorted(os.listdir(args.results_root))
    print("# folders:", len(audioNames))
    index = 1
    for audio_name in audioNames:
        #if audio_name[0] not in ['0', '1']:
        #    continue
        if index % 10 == 0:
            print("Evaluating testing example " + str(index) + " :", audio_name)
        #check whether input binaural is mono, replicate to two channels if it's mono
        if args.real_mono:
            mono_sound, audio_rate = librosa.load(os.path.join(args.results_root, audio_name, 'mixed_mono.wav'), sr=args.audio_sampling_rate)
            predicted_binaural = np.repeat(np.expand_dims(mono_sound, 0), 2, axis=0)
            if args.normalization:
                predicted_binaural = normalize(predicted_binaural)
        else:
            predicted_binaural, audio_rate = librosa.load(os.path.join(args.results_root, audio_name, 'predicted_binaural.wav'), sr=args.audio_sampling_rate, mono=False)
            if args.normalization:
                predicted_binaural = normalize(predicted_binaural)
        gt_binaural, audio_rate = librosa.load(os.path.join(args.results_root, audio_name, 'input_binaural.wav'), sr=args.audio_sampling_rate, mono=False)
        if args.normalization:
            gt_binaural = normalize(gt_binaural)
        
        # channel1 spectrogram
        predicted_spect_channel1 = librosa.core.stft(np.asfortranarray(predicted_binaural[0,:]), n_fft=512, hop_length=160, win_length=400, center=True)
        gt_spect_channel1 = librosa.core.stft(np.asfortranarray(gt_binaural[0,:]), n_fft=512, hop_length=160, win_length=400, center=True)
        # channel2 spectrogram
        predicted_spect_channel2 = librosa.core.stft(np.asfortranarray(predicted_binaural[1,:]), n_fft=512, hop_length=160, win_length=400, center=True)
        gt_spect_channel2 = librosa.core.stft(np.asfortranarray(gt_binaural[1,:]), n_fft=512, hop_length=160, win_length=400, center=True)

        #get results for this audio
        stft_distance_list.append(STFT_L2_distance(predicted_spect_channel1, gt_spect_channel1, predicted_spect_channel2, gt_spect_channel2))
        envelope_distance_list.append(Envelope_distance(predicted_binaural, gt_binaural))
        cur_angle_dist = Angle_Diff_distance(predicted_binaural, gt_binaural)
        angle_distance_list.append(cur_angle_dist)
        cur_snr = SNR(predicted_binaural, gt_binaural)
        snr_list.append(cur_snr)
        magnitude_distance_list.append(Magnitude_distance(predicted_spect_channel1, gt_spect_channel1, predicted_spect_channel2, gt_spect_channel2))
        index = index + 1

    #print the results
    stft_res = get_content(value_list=stft_distance_list, text='STFT L2 Distance')
    env_res = get_content(envelope_distance_list, text='Average Envelope Distance')
    magnitude_res = get_content(magnitude_distance_list, text='Magnitude Distance')
    angle_res = get_content(angle_distance_list, text='Phase Distance')
    snr_res = get_content(snr_list, text='SNR')

    print(stft_res)
    print(env_res)
    print(magnitude_res)
    print(angle_res)
    print(snr_res)

    store_content = [args.results_root]
    store_content.append(stft_res.split(',')[0])
    store_content.append(env_res.split(',')[0])
    store_content.append(magnitude_res.split(',')[0])
    store_content.append(angle_res.split(',')[0])
    store_content.append(snr_res.split(',')[0])
    with open('output/six_metrics.txt', 'a') as cur_file:
        cur_file.writelines('\n'.join(store_content) + '\n\n')

if __name__ == '__main__':
    main()
