import os
import time

import torch
import yaml
from torchvision.datasets import ImageFolder as torchIF
from torchvision import transforms as T
from torch.utils import data
import torchvision.utils as vutils


def get_loader(image_dir, crop_size=178, image_size=128,
               batch_size=16, dataset='CelebA', mode='train', num_workers=4):
    """Build and return a data loader."""
    # transform = []
    # if mode == 'train':
    #     transform.append(T.RandomHorizontalFlip())
    # transform.append(T.CenterCrop(crop_size))
    # transform.append(T.Resize(image_size))
    # transform.append(T.ToTensor())
    # transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    # transform = T.Compose(transform)
    #
    # dataset = ImageFolder(image_dir, transform)
    #
    # data_loader = data.DataLoader(dataset=dataset,
    #                               batch_size=batch_size,
    #                               shuffle=(mode == 'train'),
    #                               num_workers=num_workers)
    transform_list = [T.ToTensor(),
                      T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform_list = [T.RandomCrop((crop_size))] + transform_list
    transform_list = [T.Resize(image_size)] + transform_list
    transform_list = [T.RandomHorizontalFlip()] + transform_list
    transform = T.Compose(transform_list)
    dataset = torchIF(image_dir, transform=transform)
    loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode=='train'), drop_last=True, num_workers=num_workers)
    return loader

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory

class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)

def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)

def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs[0:n//2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def default_loader(path):
    return Image.open(path).convert('RGB')

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
