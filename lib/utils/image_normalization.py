import torch

class ImageNormalization():
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), dtype=torch.float32):
        self.mean = torch.tensor(mean, dtype=dtype)
        self.std = torch.tensor(std, dtype=dtype)

    def __call__(self, input, inverse=False):
        return self.denormalize(input) if inverse else self.normalize(input)

    def _get_coeffs(self, dims, device):
        if dims == 4:
            mean = self.mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            std = self.std.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        elif dims == 3:
            mean = self.mean.unsqueeze(1).unsqueeze(2)
            std = self.std.unsqueeze(1).unsqueeze(2)
        else:
            raise ValueError(f'expected input to ImageNormalization to be 4D or 3D tensor, but received {dims}D')

        return mean.to(device), std.to(device)

    def normalize(self, images):
        mean, std = self._get_coeffs(len(images.shape), images.device)
        images_norm = (images - mean) / std
        return images_norm

    def denormalize(self, images_norm):
        mean, std = self._get_coeffs(len(images_norm.shape), images_norm.device)
        images = images_norm * std + mean
        return images