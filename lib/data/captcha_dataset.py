import cv2
from torch.utils.data import Dataset
from albumentations import Compose
import time
import numpy as np
import pickle
from lib.utils.log import logger
from albumentations import Resize, SmallestMaxSize, CenterCrop, PadIfNeeded
from albumentations.pytorch.transforms import ToTensor
from lib.utils.path import list_files_with_ext, list_subdirs, list_images
import torch
import random
from os.path import join, splitext, basename


class CaptchaDataset(Dataset):

    def __init__(self, input_dir, input_file=None, aug=None, normalization=None, interpolation=cv2.INTER_LINEAR):

        self._input_dir = input_dir
        self._input_file = input_file
        self._aug = aug
        self._normalization = normalization
        self._transform = None
        self._interpolation = interpolation
        self._load_samples()
        self._update_transforms()

    def _load_samples(self):

        tic = time.time()

        if self._input_file is not None:
            samples = []
            with open(join(self._input_dir, self._input_file), 'r') as fp:
                lines = fp.read().splitlines()
            for line in lines:
                imname, code_str = line.split(';')
                symbols = ''.join(code_str.split(','))
                labels = self._encode(symbols.lower())
                samples.append((imname, labels))
        else:
            logger.info(f'listing images {self._input_dir}..')
            imnames = list_files_with_ext(str(self._input_dir),
                                          valid_exts=['.png', '.jpg', '.jpeg'], recursive=True)
            samples = []
            for imname in imnames:
                imname_base = basename(imname)
                imname_base = splitext(imname_base)[0]
                labels = self._encode(imname_base.lower())
                samples.append((imname, labels))

        num_labels = 0
        for imname, labels in samples:
            num_labels = max(len(labels), num_labels)

        self._samples = samples
        self._num_labels = num_labels
        self._num_classes = ord('z') - ord('a')

        logger.info('finished in {:.3} sec'.format(time.time() - tic))
        logger.info('number of samples: {}'.format(len(self._samples)))
        logger.info('maximum number of labels: {}'.format(self._num_labels))
        logger.info('number of classes: {}'.format(self._num_classes))

    def _encode(self, symbols):
        encoding = []
        for s in symbols:
            raw_code = ord(s)
            assert raw_code <= ord('z') and raw_code >= ord('a')
            code = raw_code - ord('a')
            encoding.append(code)
        return encoding

    def _update_transforms(self):

        transform = [ToTensor()]
        if self._aug is not None:
            transform = self._aug + transform
        self._transform = Compose(transform)

    def __getitem__(self, idx):

        imname, labels = self._samples[idx]

        img = cv2.imread(join(self._input_dir, imname), 1)
        img = img[:,:,::-1] # BGR to RGB

        if self._transform is not None:
            img = self._transform(image=img)['image']

        if self._normalization is not None:
            img = self._normalization(img)

        sample = {}
        sample['img'] = img
        sample['label'] = torch.from_numpy(np.array(labels, dtype=np.long)).to(torch.long)

        return sample

    def num_classes(self):
        return self._num_classes

    def num_labels(self):
        return self._num_labels

    def __len__(self):
        return len(self._samples)