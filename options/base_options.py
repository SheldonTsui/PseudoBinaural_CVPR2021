import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--splitPath', help='path to the folder that contains train.txt, val.txt and test.txt')
        self.parser.add_argument('--MUSICPath', help='path to the folder that contains the splits of MUSIC dataset')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='spatialAudioVisual', help='name of the experiment. It decides where to store models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/', help='models are saved here')
        self.parser.add_argument('--dataset_mode', type=str, choices=['sep', 'stereo', 'sepstereo', 'ASMR_stereo', 'Pseudo_stereo', 'Pseudo_sepstereo', 'Pseudo_sep', 'Augment_sepstereo', 'Augment_stereo', 'Augment_ASMR_sepstereo', 'Augment_ASMR_stereo', 'ASMR_stereo_crop'], default='stereo', help='chooses how datasets are loaded.')
        self.parser.add_argument('--audio_model', type=str, default='AudioNet', help='audio model type')
        self.parser.add_argument('--visual_model', type=str, choices=['resnet18', 'resnet34'], default='resnet18', help='visual model type')
        self.parser.add_argument('--fusion_model', type=str, choices=['none', 'AssoConv', 'APNet'], default='none', help='fusion model type')
        self.parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
        self.parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')
        self.parser.add_argument('--audio_sampling_rate', default=16000, type=int, help='audio sampling rate')
        self.parser.add_argument('--audio_length', default=0.63, type=float, help='audio length, default 0.63s')
        self.parser.add_argument('--norm_mode', type=str, choices=['syncbn', 'bn', 'in'], default='bn', help='norm mode')
        self.parser.add_argument('--blending', action='store_true', help='whether use Possion blending for Pseudo data')
        self.parser.add_argument('--not_use_background', action='store_true', help='whether use background to create the pseudo data')
        self.parser.add_argument('--visualize_data', action='store_true', help='whether visualize the pseudo data')
        self.parser.add_argument('--datalist', type=str, choices=['FAIR_data2', 'FAIR_data', 'split_data', 'split_data2', 'MUSIC', 'MUSIC2', 'face_data'], default='FAIR_data', help='the filename of data list')
        self.parser.add_argument('--audio_normal', action='store_true', help='whether normalizer audio')
        self.parser.add_argument('--fusion_loss_only', action='store_true', help='whether only use fusion loss for APNet')
        self.parser.add_argument('--patch_resize', action='store_true', help='whether resize the patch when creating the pseudo pair')
        self.parser.add_argument('--pseudo_ratio', default=0.5, type=float, help='the ratio of pseudo stereo for Augment mode')
        self.parser.add_argument('--stereo_mode', type=str, choices=['direct', 'ambisonic', 'ambidirect'], default='ambisonic', help='which mode to generate pseudo stereo label')
        self.parser.add_argument('--fov', type=str, choices=['1/2', '1/3', '2/3', '5/6', '1'], default='2/3', help='the choices for fov')
        self.parser.add_argument('--seed', type=int, default=1234)
        self.enable_data_augmentation = True
        self.initialized = True

    def parse(self):
        if not self.initialized:
                self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.mode = self.mode
        self.opt.isTrain = self.isTrain
        self.opt.enable_data_augmentation = self.enable_data_augmentation

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
                id = int(str_id)
                if id >= 0:
                        self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
                torch.cuda.set_device(self.opt.gpu_ids[0])


        #I should process the opt here, like gpu ids, etc.
        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')


        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
