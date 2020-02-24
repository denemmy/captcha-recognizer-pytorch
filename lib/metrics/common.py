import torch
import numpy as np


class Accuracy:
    def __init__(self):
        self.reset()

    def update(self, labels, preds):
        correct, labeled = batch_pix_accuracy(preds, labels)
        self.total_correct += correct
        self.total_label += labeled

    def get(self):
        accuracy = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        return accuracy

    def reset(self):
        self.total_correct = 0
        self.total_label = 0


class CaptchaAccuracy:
    def __init__(self):
        self.reset()

    def update(self, labels, preds):

        correct, labeled = batch_pix_accuracy(preds, labels)

        self.total_correct_symbols += correct
        self.total_symbols += labeled

        labels = labels.view(labels.shape[0], -1)
        preds = preds.view(preds.shape[0], -1)

        captchas_correct, captchas_labeled = batch_accuracy(preds, labels)

        self.total_correct_captchas += captchas_correct
        self.total_captchas += captchas_labeled

    def get(self):
        symbol_accuracy = 1.0 * self.total_correct_symbols / (np.spacing(1) + self.total_symbols)
        captcha_accuracy = 1.0 * self.total_correct_captchas / (np.spacing(1) + self.total_captchas)
        return symbol_accuracy, captcha_accuracy

    def reset(self):
        self.total_symbols = 0
        self.total_correct_symbols = 0

        self.total_captchas = 0
        self.total_correct_captchas = 0

def batch_pix_accuracy(predicted, target):
    """PixAcc"""
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary

    predict = predicted.to(torch.int64) + 1
    target = target.to(torch.int64) + 1

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_accuracy(predicted, target):

    predict = predicted.to(torch.int64) + 1
    target = target.to(torch.int64) + 1

    per_row_target = torch.sum(target > 0, dim=1)
    per_row_pred = torch.sum((predict == target) * (target > 0), dim=1)

    labeled = torch.sum(per_row_target > 0).item()
    correct = torch.sum((per_row_pred == per_row_target) * (per_row_target > 0)).item()

    return labeled, correct