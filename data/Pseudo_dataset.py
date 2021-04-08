import os
import os.path as osp
import librosa
import numpy as np
from glob import glob
import random
import mmcv
import natsort
from math import pi
import pdb
import cv2

import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch.utils.data as data

#from ambisonics.binauralizer import SourceBinauralizer
from .ambisonics.common import spherical_harmonics_matrix
from .ambisonics.hrir import CIPIC_HRIR
from .ambisonics.position import PositionalSource, Position

def audio_normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return rms / desired_rms, samples

def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel

def process_image(image, augment, square=False):
    if square:
        iH, iW = 240, 240
        H, W = 224, 224
    else:
        iH, iW = 240, 480
        H, W = 224, 448
    image = mmcv.imresize(image, (iW,iH))
    h,w,_ = image.shape
    w_offset = w - W 
    h_offset = h - H 
    left = random.randrange(0, w_offset + 1)
    upper = random.randrange(0, h_offset + 1)
    image = mmcv.imcrop(image, np.array([left, upper, left+W-1, upper+H-1]))

    if augment:
        enhancer = ImageEnhance.Brightness(Image.fromarray(image))
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image

class PseudoDataset(data.Dataset):
    def __init__(self, opt, list_sample_file):
        super().__init__()

        self.opt = opt
        self.total_samples = mmcv.list_from_file(list_sample_file)

        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)

        random.seed(self.opt.seed)

        # load background, just one large-sizeimage
        self.bkg_img = mmcv.imread('./data/bkg.png')

        # build binauralizer
        hrtf_dir = "./data/subject03" 
        # binauralizer = SourceBinauralizer(use_hrtfs=True, cipic_dir=hrtf_dir)
        self.hrir_db = CIPIC_HRIR(hrtf_dir)
        # encode to ambisonics, and then stereo
        speakers_phi = (2. * np.arange(2*4) / float(2*4) - 1.) * np.pi
        self.speakers_pos = [Position(phi, 0, 1, 'polar') for phi in speakers_phi]
        self.sph_mat = spherical_harmonics_matrix(self.speakers_pos, max_order=1) # shape: [N_array_speakers, 4]

        # parameter of loading audio 
        self.exp_audio_len = int(self.opt.audio_length * self.opt.audio_sampling_rate)
        #print("Now load {} box info".format(self.opt.mode))
        #self.box_info = mmcv.load('./dataset/ASMR/scripts/results/{}_box_info.pkl'.format(self.opt.mode))

        if opt.fov == '1/3':
            self.fov = 1/3.
        elif opt.fov == '1/2':
            self.fov = 1/2.
        elif opt.fov == '5/6':
            self.fov = 5/6.
        elif opt.fov == '1':
            self.fov = 1. 
        else:
            self.fov = 2/3.

        #self.categories = ['acoustic_guitar', 'banjo', 'bass', 'cello', 'drum', 'harp', 'piano', 'trumpet', 'ukelele']

    def construct_stereo_direct(self, pst_sources):
        stereo = np.zeros((2, self.exp_audio_len))
        for src in pst_sources:
            left_hrir, right_hrir = self.hrir_db.get_closest(src.position)[1:]
            left_signal = np.convolve(src.signal, np.flip(left_hrir, axis=0), 'valid')
            right_signal = np.convolve(src.signal, np.flip(right_hrir, axis=0), 'valid')

            n_valid, i_start = left_signal.shape[0], left_hrir.shape[0] - 1
            stereo[0, i_start:(i_start + n_valid)] += left_signal
            stereo[1, i_start:(i_start + n_valid)] += right_signal

        return stereo

    def construct_stereo_ambi(self, pst_sources):
        # encode to ambisonics
        Y = spherical_harmonics_matrix([src.position for src in pst_sources], max_order=1) # Y shape: [n_signals, 4] 
        signals = np.stack([src.signal for src in pst_sources], axis=1) # signals shape: [Len, n_signals]
        ambisonic = np.dot(signals, Y) # shape: [Len, 4]

        array_speakers_sound = np.dot(ambisonic, self.sph_mat.T)
        #array_speakers_sound = np.dot(ambisonic, np.linalg.pinv(self.sph_mat))
        array_sources = [PositionalSource(array_speakers_sound[:, i], speaker_pos, \
            self.opt.audio_sampling_rate) for i, speaker_pos in enumerate(self.speakers_pos)]

        return self.construct_stereo_direct(array_sources)

    def construct_stereo_ambi_direct(self, pst_sources):
        # encode to ambisonics
        Y = spherical_harmonics_matrix([src.position for src in pst_sources], max_order=1) 
        signals = np.stack([src.signal for src in pst_sources], axis=1)
        ambisonic = np.dot(signals, Y) # shape: [Len, 4]

        stereo = np.stack((
            ambisonic[:, 0] / 2 + ambisonic[:, 1] / 2,
            ambisonic[:, 0] / 2 - ambisonic[:, 1] / 2
        ))

        return stereo

    def _get_pseudo_item(self, index):        
        # ensure the number of audios in a scene
        N = np.random.choice([1,2,3], p=[0.4, 0.5, 0.1])
        chosen_samples = [self.total_samples[index]]
        # avoid repeat sample
        for _ in range(1, N):
            while True:
                new_sample = random.choice(self.total_samples)
                if new_sample not in chosen_samples:
                    chosen_samples.append(new_sample)
                    break

        audio_margin = 0
        init_H = 360
        init_W = 640
        pst_sources = []

        if self.opt.not_use_background:
            cur_bkg_img = np.zeros((init_H, init_W, 3)).astype(np.uint8) 
        else:
            # crop background img, exp_shape: [init_H, init_W, 3]
            bkg_start_x = np.random.randint(low=0, high=self.bkg_img.shape[1] - init_W)
            bkg_start_y = np.random.randint(low=0, high=self.bkg_img.shape[0] - init_H)
            cur_bkg_img = mmcv.imcrop(self.bkg_img.copy(), 
                np.array([bkg_start_x, bkg_start_y, bkg_start_x+init_W-1, bkg_start_y+init_H-1]))
            #H_bkg, W_bkg, _ = cur_bkg_img.shape

        corner_record = []
        patch_size_record = []
        center_x_record = []
        audio_list = []
        patch_list = []
        actual_N = 0
        #load audio
        for idx, chosen_sample in enumerate(chosen_samples):
            audio_file, img_folder = chosen_sample.split(',')
            # audio part
            audio, audio_rate = librosa.load(audio_file, sr=self.opt.audio_sampling_rate, mono=True)
            #randomly get a start time for the audio segment from the original clip
            audio_len = len(audio) / audio_rate
            assert audio_len - self.opt.audio_length - audio_margin > audio_margin
            audio_start_time = random.uniform(audio_margin, audio_len - self.opt.audio_length - audio_margin)
            audio_end_time = audio_start_time + self.opt.audio_length
            audio_start = int(audio_start_time * self.opt.audio_sampling_rate)
            audio_end = audio_start + self.exp_audio_len
            audio = audio[audio_start:audio_end]
            if self.opt.audio_normal:
                normalizer0, audio = audio_normalize(audio)

            # video part
            # load img **patches**, copy bkg_img and construct a new image
            # load accurate frame
            cur_img_list = natsort.natsorted(glob(osp.join(img_folder, '*.jpg')))
            # get the closest frame to the audio segment
            frame_idx = (audio_start_time + audio_end_time) / 2 * 10
            frame_idx = int(np.clip(frame_idx, 0, len(cur_img_list) - 1))
            img_file = cur_img_list[frame_idx]
            img_patch = mmcv.imread(img_file)

            if self.opt.patch_resize:
                h_patch, w_patch, _= img_patch.shape
                resize_ratio = min(1/normalizer0, init_H / h_patch, init_W / 2 / w_patch) 
                img_patch = mmcv.imrescale(img_patch, resize_ratio * random.uniform(0.8, 1))
            H_new, W_new, _ = img_patch.shape

            # just consider the overlap in the horizontal axis
            occupy_matrix = np.ones((init_W))
            # avoid cross border in x dim
            occupy_matrix[:(-W_new + 1)] = 0
            # avoid overlap
            for last_corner_x, W_last in zip(corner_record, patch_size_record):
                occupy_x = max(0, last_corner_x - W_new)
                occupy_matrix[occupy_x : last_corner_x + W_last] = 1

            # random sample position for this mono audio
            free_x_positions = np.where(occupy_matrix == 0)[0]
            if len(free_x_positions) < 2:
                break
            actual_N += 1
            corner_x = random.choice(free_x_positions)
            corner_record.append(corner_x)
            patch_size_record.append(W_new)
            corner_y = random.randint(0, init_H - H_new)
            
            center_y = corner_y + H_new // 2
            center_x = corner_x + W_new // 2
            center_x_record.append(center_x)
            azimuth = (init_W // 2 - center_x) / init_W * pi * self.fov
            elevation = (init_H // 2 - center_y) / init_H * pi / 2

            if self.opt.visualize_data:
                output_dir = 'others/dataset_visual/{:d}_{:d}'.format(N, index)
                if not osp.exists(output_dir):
                    os.mkdir(output_dir)
                if librosa.__version__ >= '0.8.0':
                    import soundfile as sf
                    sf.write(osp.join(output_dir, '{:d}.wav'.format(idx)), audio.transpose(), audio_rate)
                else:
                    librosa.output.write_wav(osp.join(output_dir, '{:d}.wav'.format(idx)), audio, sr=audio_rate)
            audio_list.append(audio)
            patch_list.append(img_file)
            pst_sources.append(PositionalSource(audio, Position(azimuth, elevation, 3, 'polar'), audio_rate))

            if self.opt.blending:
                center = (center_x, center_y)
                mask = 255 * np.ones(img_patch.shape, img_patch.dtype)
                cur_bkg_img = cv2.seamlessClone(img_patch, cur_bkg_img, mask, center, cv2.NORMAL_CLONE)
            else:
                patch_in_start_x = corner_x 
                patch_in_start_y = corner_y
                assert patch_in_start_x >= 0
                assert patch_in_start_y >= 0
                cur_bkg_img[patch_in_start_y : patch_in_start_y + H_new, patch_in_start_x : patch_in_start_x + W_new] = img_patch

        if self.opt.stereo_mode == 'direct':
            #print("use direct")
            stereo = self.construct_stereo_direct(pst_sources)
        elif self.opt.stereo_mode == 'ambisonic':
            #print("use ambisonic")
            stereo = self.construct_stereo_ambi(pst_sources)
        elif self.opt.stereo_mode == 'ambidirect':
            #print("use ambidirect")
            stereo = self.construct_stereo_ambi_direct(pst_sources)
        else:
            raise ValueError("please choose right stereo mode")

        normalizer, _ = audio_normalize(stereo[0] + stereo[1])
        stereo = stereo / normalizer
        audio_channel1, audio_channel2 = stereo
        frame = cur_bkg_img
        if self.opt.visualize_data:
            output_dir = 'others/dataset_visual/{:d}_{:d}'.format(N, index)
            if librosa.__version__ >= '0.8.0':
                import soundfile as sf
                sf.write(osp.join(output_dir, 'input_binaural.wav'), stereo.transpose(), audio_rate)
            else:
                librosa.output.write_wav(osp.join(output_dir, 'input_binaural.wav'), stereo, sr=audio_rate)
            mmcv.imwrite(frame, osp.join(output_dir, 'reference.jpg'))
        frame = process_image(frame, self.opt.enable_data_augmentation)
        frame = self.vision_transform(frame)

        #passing the spectrogram of the difference
        audio_diff_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 - audio_channel2))
        audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 + audio_channel2))
        
        data_ret = {'frame': frame, 'audio_diff_spec':audio_diff_spec, 'audio_mix_spec':audio_mix_spec}

        # incorporate separation part
        assert len(patch_list) == len(audio_list)
        left_channel = np.zeros(self.exp_audio_len).astype(np.float32)
        right_channel = np.zeros(self.exp_audio_len).astype(np.float32)
        if len(audio_list) >= 2:
            for cur_audio, center_x in zip(audio_list, center_x_record):
                if center_x < init_W // 3:
                    left_channel += cur_audio
                elif center_x > init_W // 3:
                    right_channel += cur_audio
                else:
                    left_channel += cur_audio
                    right_channel += cur_audio
                
            _, left_channel = audio_normalize(left_channel)
            _, right_channel = audio_normalize(right_channel)
        else:
            left_channel = audio_channel1
            right_channel = audio_channel2

        if self.opt.visualize_data:
            output_dir = 'others/dataset_visual/{:d}_{:d}'.format(N, index)
            if librosa.__version__ >= '0.8.0':
                import soundfile as sf
                sf.write(osp.join(output_dir, 'gt_left.wav'), left_channel, audio_rate)
                sf.write(osp.join(output_dir, 'gt_right.wav'), right_channel, audio_rate)
            else:
                librosa.output.write_wav(osp.join(output_dir, 'gt_left.wav'), left_channel, audio_rate)
                librosa.output.write_wav(osp.join(output_dir, 'gt_right.wav'), right_channel, audio_rate)

        sep_mix_spec = audio_mix_spec
        sep_diff_spec = torch.FloatTensor(generate_spectrogram(left_channel - right_channel))
        frame_sep_list = frame

        if self.opt.mode == 'train':
            data_ret_sep = {'frame_sep': frame_sep_list, 'sep_diff_spec': sep_diff_spec, 'sep_mix_spec': sep_mix_spec}
        else:
            data_ret_sep = {'frame_sep': frame_sep_list, 'sep_diff_spec': sep_diff_spec, 'sep_mix_spec': sep_mix_spec, 'left_audio': left_channel, 'right_audio': right_channel}

        return data_ret, data_ret_sep

    def __getitem__(self, index):
        data_ret, data_ret_sep = self._get_pseudo_item(index)
        if self.opt.dataset_mode == 'Pseudo_stereo':
            return data_ret
        elif self.opt.dataset_mode == 'Pseudo_sep':
            return data_ret_sep
        else:
            data_ret.update(data_ret_sep)
            return data_ret

    def __len__(self):
        return len(self.total_samples)

    def name(self):
        return 'PseudoDataset'

    def initialize(self, opt):
        pass
