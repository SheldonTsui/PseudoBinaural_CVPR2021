import torch

def CreateDataLoader(opt):
    dataset = None
    if opt.dataset_mode == 'stereo':
        from data.stereo_dataset import StereoDataset
        dataset = StereoDataset(opt)
    elif opt.dataset_mode == 'sep':
        from data.sep_dataset import SepDataset
        dataset = SepDataset(opt)
    elif opt.dataset_mode == 'sepstereo':
        from data.sepstereo_dataset import SepStereoDataset
        dataset = SepStereoDataset(opt)
    elif opt.dataset_mode == 'ASMR_stereo':
        from data.ASMR_dataset import ASMRDataset
        ASMR_file = 'data/asmr_mono_sources/{}.txt'.format(opt.mode)
        dataset = ASMRDataset(opt, ASMR_file)
    elif opt.dataset_mode == 'ASMR_stereo_crop':
        from data.ASMR_dataset import ASMRDataset
        ASMR_file = 'data/asmr_mono_sources/{}_crop.txt'.format(opt.mode)
        dataset = ASMRDataset(opt, ASMR_file)
    elif 'Pseudo' in opt.dataset_mode:
        from data.Pseudo_dataset import PseudoDataset
        list_sample_file = 'data/FAIR-Play_mono_sources/{}_{}.txt'.format(opt.mode, opt.datalist)
        dataset = PseudoDataset(opt, list_sample_file)
    elif 'Augment' in opt.dataset_mode:
        list_sample_file = 'data/FAIR-Play_mono_sources/{}_{}.txt'.format(opt.mode, opt.datalist)
        if 'ASMR' in opt.dataset_mode:
            from data.Augment_ASMR_dataset import AugmentDataset
            ASMR_file = 'data/asmr_mono_sources/{}.txt'.format(opt.mode)
            dataset = AugmentDataset(opt, ASMR_file)
        else:
            from data.Augment_dataset import AugmentDataset
            dataset = AugmentDataset(opt, list_sample_file)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    print("#%s clips = %d" %(opt.mode, len(dataset)))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=opt.mode=='train',
        num_workers=int(opt.nThreads),
        drop_last=opt.mode=='train'
    )

    return dataloader
