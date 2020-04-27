import argparse
import gc
import os
import sys

import tensorboardX
import torch
from torch.backends import cudnn

from trainer import MUNIT_Trainer

import yaml

from utils import get_config, get_loader, prepare_sub_folder, Timer, write_loss, write_2images

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
opts = parser.parse_args()

config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']

trainer = MUNIT_Trainer(config)

trainer.cuda()

data_loader_a = get_loader(os.path.join(config['data_root'], 'b'),
                        config['crop_image_height'], config['new_size'], config['batch_size'],
                        'afhq', 'train', 1)
data_loader_b = get_loader(os.path.join(config['data_root'], 'a'),
                        config['crop_image_height'], config['new_size'], config['batch_size'],
                        'afhq', 'train', 1)


train_display_images_a = torch.stack([data_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([data_loader_b.dataset[i][0] for i in range(display_size)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

gc.collect()
torch.cuda.empty_cache()
# For fast training.
cudnn.benchmark = True
iterations = 0
data_iter_a = iter(data_loader_a)
data_iter_b = iter(data_loader_b)
while True:

    for it, images in enumerate(data_loader_a):
        trainer.update_learning_rate()
        try:
            images_a, label_org = next(data_iter_a)
            images_b, label_org = next(data_iter_b)
        except:
            data_iter_a = iter(data_loader_a)
            data_iter_b = iter(data_loader_b)
            images_a, label_org = next(data_iter_a)
            images_b, label_org = next(data_iter_b)

        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                # test_image_outputs = trainer.sample(test_display_images_a)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            # write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')


