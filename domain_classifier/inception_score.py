import numpy as np
from glob import glob
from PIL import Image
import argparse
import os 

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models.inception import inception_v3
from torchvision import datasets, transforms

# guidance values
w_lst = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


class SimpleDataset(Dataset):
    """An simple image dataset for calculating inception score and FID."""

    def __init__(self, root, exts=['png', 'jpg', 'JPEG'], transform=None, num_images=None):
        """Construct an image dataset.

        Args:
            root: Path to the image directory. This directory will be
                  recursively searched.
            exts: List of extensions to search for.
            transform: A torchvision transform to apply to the images. If
                       None, the images will be converted to tensors.
            num_images: The number of images to load. If None, all images
                        will be loaded.
        """
        self.paths = []
        self.transform = transform
        for ext in exts:
            self.paths.extend(
                list(glob(
                    os.path.join(root, '**/*.%s' % ext), recursive=True)))
        self.paths = self.paths[:num_images]

    def __len__(self):              # noqa
        return len(self.paths)

    def __getitem__(self, idx):     # noqa
        image = Image.open(self.paths[idx])
        image = image.convert('RGB')        # fix ImageNet grayscale images
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor(image)
        return image


def inception_score(folder, cuda=True, batch_size=64, resize=True, splits=5):
    dataset = SimpleDataset(
        root = folder,
        transform = transforms.ToTensor()
    )
    N = len(dataset)

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()

    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_prob(x):
        if resize:
            x = up(x)
        with torch.no_grad():
            x = inception_model(x)
        # return F.softmax(x).data.cpu().numpy()
        return F.softmax(x, dim=-1).detach()

    # Get predictions
    probs = torch.zeros((N, 1000))

    for i, batch in enumerate(dataloader):
        batch = batch.type(dtype)
        batch_size_i = batch.size()[0]

        probs[i*batch_size:i*batch_size + batch_size_i] = get_prob(batch)

    # Inception Score
    scores = []
    for i in range(splits):
        part = probs[
            (i * probs.shape[0] // splits):
            ((i + 1) * probs.shape[0] // splits), :]
        if cuda:
            kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (
                np.log(part) -
                np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if cuda:
        scores = torch.stack(scores)
        inception_score = torch.mean(scores).cpu().item()
        std = torch.std(scores).cpu().item()
    else:
        inception_score, std = (np.mean(scores), np.std(scores))
    del probs, scores
    return inception_score, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--master-folder", type=str, help="path to master folder, not including cfg_w")

    opt = parser.parse_args()

    for cfg_w in w_lst:
        master_folder = os.path.join(opt.master_folder, f"guide_w{cfg_w}")
        print(f"sampling from stable-diffusion-v2 w/ cfg_w = {cfg_w}")
        is_mean, is_std = inception_score(master_folder)
        print(f"inception score: mean {is_mean} +- std {is_std}")