from easydict import EasyDict as edict
from albumentations import SmallestMaxSize, RandomSizedCrop, RandomCrop, Resize, Rotate, PadIfNeeded
from albumentations import RandomBrightnessContrast, RGBShift, OneOf, GaussNoise, IAAAdditiveGaussianNoise
from albumentations import Blur, HueSaturationValue, ToGray, ChannelShuffle, OpticalDistortion, ElasticTransform

from lib.core.captcha_trainer import CaptchaTrainer, CaptchaTester

from lib.data.captcha_dataset import CaptchaDataset
from lib.utils.image_normalization import ImageNormalization
import cv2
from torch import optim
from lib.utils.log import logger
import shutil
import torch.nn as nn
from lib.utils.weight_init import XavierGluon, Normal
from lib.utils.lr_scheduler import ComposeScheduler, ConstantSegment, CosineSegment
from lib.nn.modeling.simple_classifier import Classifier


def init_model(num_classes, num_labels):

    net = Classifier(num_classes=num_classes, num_labels=num_labels)
    net.apply(XavierGluon(rnd_type='gaussian', magnitude=2.34, factor_type='in'))

    return net


def train(cfg):

    img_width = 200
    img_height = 70

    normalization = ImageNormalization(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    interpolation = cv2.INTER_LINEAR

    train_dataset = CaptchaDataset(
        input_dir=cfg.dataset_path,
        input_file='train.txt',
        normalization=normalization,
        aug=[
            Resize(height=img_height, width=img_width, always_apply=True, interpolation=interpolation),
            # OneOf([
            #     OpticalDistortion(distort_limit=0.3, shift_limit=0.2, p=1., interpolation=interpolation),
            #     ElasticTransform(p=1., interpolation=interpolation),
            # ], p=0.25),
            RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.3, 0.5), p=0.2),
            RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
            HueSaturationValue(hue_shift_limit=50, sat_shift_limit=70, val_shift_limit=50, p=0.2),
            OneOf([
                ChannelShuffle(p=1.),
                ToGray(p=1.),
            ], p=0.25),
            Blur(blur_limit=5, p=0.5),
            OneOf([
                GaussNoise(var_limit=(20, 50), p=1),
                IAAAdditiveGaussianNoise(scale=(7.5, 7.5), p=1)
            ], p=0.5),
        ],
        interpolation=interpolation
    )

    val_dataset = CaptchaDataset(
        input_dir=cfg.dataset_path,
        input_file='val.txt',
        normalization=normalization,
        aug=[
            Resize(height=img_height, width=img_width, always_apply=True, interpolation=interpolation),
        ],
        interpolation=interpolation
    )

    num_classes = train_dataset.num_classes()
    num_labels = train_dataset.num_labels()
    net = init_model(num_classes, num_labels)

    batch_size_one_gpu = 32
    batch_size_cpu = 4
    batch_size = batch_size_one_gpu * len(cfg.gpu_ids) if len(cfg.gpu_ids) > 0 else batch_size_cpu

    total_kimgs = 10000

    # optim_params = edict()
    # optim_params.opt = optim.SGD
    # optim_params.params = edict()
    #
    # optim_params.params.lr = 0.01
    # optim_params.params.momentum = 0.9
    # optim_params.params.weight_decay = 1e-4

    # schedulers = [ComposeScheduler([(CosineSegment(period=total_kimgs, min_scale=0.0), total_kimgs)],
    #                                param_name='lr')]
    # optim_params.schedulers = schedulers

    optim_params = edict()
    optim_params.opt = optim.Adam
    optim_params.params = edict()
    optim_params.params.lr = 1e-3

    trainer = CaptchaTrainer(
        cfg, net=net,
        optim_params=optim_params, train_dataset=train_dataset,
        val_dataset=val_dataset,
        image_normalization=normalization,
        batch_size=batch_size,
        total_kimgs=total_kimgs,
        log_period_kimgs={0:1,20:1,50:5,500:10},
        log_images_period_kimgs={0:20,100:100,500:500},
        last_checkpoint_period_kimgs={0:100,500:500},
        checkpoint_period_kimgs=5000,
        eval_period={0:100,500:500},
        n_display=16,
    )
    trainer.train()


def test(cfg):

    base_size = 576

    normalization = ImageNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    interpolation = cv2.INTER_LINEAR

    test_dataset = CaptchaDataset(
        input_dir=cfg.val_dataset_path,
        normalization=normalization,
        input_file='test.txt',
        aug=[SmallestMaxSize(max_size=base_size, always_apply=True, interpolation=interpolation)],
        default_transform=False,
        original_masks=True,
    )

    num_classes = test_dataset.num_classes()
    num_labels = test_dataset.num_labels()
    net = init_model(num_classes, num_labels)

    batch_size = 1
    if len(cfg.gpu_ids) > 0:
        batch_size = batch_size * len(cfg.gpu_ids)

    tester = CaptchaTester(
        cfg, net=net, test_dataset=test_dataset,
        image_normalization=normalization,
        batch_size=batch_size,
        n_display=64
    )

    if cfg.test_cmd == 'metrics':
        results = tester.evaluate(use_flips=True)
        output_str = ', '.join([f'{k}: {results[k]:.4f}' for k in results])
        logger.info(output_str)
    elif cfg.test_cmd == 'visualize':
        raise NotImplementedError
    else:
        assert False, f'unknown test command {cfg.test_cmd}'