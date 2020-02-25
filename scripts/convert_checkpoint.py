import torch
import sys
sys.path.append('..')


def convert():
    checkpoint_path = '/media/denemmy/hdd/tensorboard/captcha/captcha_classification/baseline/007_300k_data_adam_50k_iter/checkpoints/latest_checkpoint.tar'
    output_path = '/media/denemmy/hdd/tensorboard/captcha/captcha_classification/baseline/007_300k_data_adam_50k_iter/checkpoints/model.pt'

    states = torch.load(checkpoint_path)
    net_states = states['net']
    torch.save(net_states, output_path)


if __name__ == '__main__':
    convert()