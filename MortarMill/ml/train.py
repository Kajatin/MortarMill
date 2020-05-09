import argparse
import os
import logging
from time import time

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt

#from eval import eval_net
from unet import UNet
from dataset import BrickDataset


logger = logging.getLogger(__name__)


def train_model(epoch):
    # set the model to training mode
    model.train()
    # accumulator for training loss
    training_loss = 0

    with tqdm(total=len(train_loader), desc=f'Train {epoch}/{args.epochs+c_epoch}', unit='img') as pbar:
        for batch_idx, data in enumerate(train_loader):
            # get the batch training data
            image = data['image'].to(device=device, dtype=torch.float32)
            mask = data['mask'].to(device=device, dtype=torch.float32)
            # reset gradients
            optimizer.zero_grad()
            # run data through model
            predicted_mask = model(image)
            # evaluate loss
            loss = criterion(predicted_mask, mask)
            training_loss += loss.item()
            # calculate gradients
            loss.backward()
            # clip gradients to avoid exploding problem
            nn.utils.clip_grad_value_(model.parameters(), 1)
            # update parameters
            optimizer.step()

            pbar.set_postfix(**{'average loss': training_loss/(batch_idx+1)})
            pbar.update(image.shape[0])

    try:
        os.mkdir('models/checkpoints/')
        logger.info('Created checkpoint directory')
    except OSError:
        pass
    
    save_path = f'models/checkpoints/CP_epoch{epoch}.tar'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'average_loss': training_loss/len(train_loader)
        }, save_path)
    # logger.info(f'Checkpoint {epoch} saved !')

    # return the average training loss
    return training_loss/len(train_loader)


def validate_model(epoch):
    # set the model to evaluation mode
    model.eval()
    # accumulator for training loss
    validation_loss = 0

    with tqdm(total=len(validation_loader), desc=f'Validation {epoch}/{args.epochs+c_epoch}', unit='img') as pbar:
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader):
                # get the batch training data
                image = data['image'].to(device=device, dtype=torch.float32)
                mask = data['mask'].to(device=device, dtype=torch.float32)
                # run data through model
                predicted_mask = model(image)
                # evaluate loss
                loss = criterion(predicted_mask, mask)
                validation_loss += loss.item()
                
                pbar.set_postfix(**{'average loss': validation_loss/(batch_idx+1)})
                pbar.update(image.shape[0])

    # return the average validation loss
    return validation_loss/len(validation_loader)


def test_model(epoch):
    # set the model to evaluation mode
    model.eval()
    # accumulator for training loss
    test_loss = 0

    with tqdm(total=len(test_loader), desc=f'Testing model after {epoch} epochs', unit='img') as pbar:
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                # get the batch training data
                image = data['image'].to(device=device, dtype=torch.float32)
                mask = data['mask'].to(device=device, dtype=torch.float32)
                # run data through model
                predicted_mask = model(image)
                # evaluate loss
                loss = criterion(predicted_mask, mask)
                test_loss += loss.item()
                
                pbar.set_postfix(**{'average loss': test_loss/(batch_idx+1)})
                pbar.update(image.shape[0])

    print('====> TEST: Average test loss: {:.4f}'.format(test_loss/len(test_loader)))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # create model, optimizer, and loss function on specific device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99)
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    logger.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\tTransposed conv upscaling\n'
                 f'\tUsing device {device}')

    # bootsrap model training if arg is provided
    c_epoch = 0
    if args.load:
        checkpoint = torch.load(args.load, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        c_epoch = checkpoint['epoch']
        average_loss = checkpoint['average_loss']
        print('Bootstrapping model from {}'.format(args.load))
        print('Continuing training from epoch: {}\n'.format(c_epoch+1))

    # train the model
    try:
        # create a dataset
        dataset = BrickDataset('samples/images/', 'samples/masks/', args.scale)
        # split dataset into train, validation, and test portions
        train_split = int(0.8 * len(dataset))
        valid_split = int(0.1 * len(dataset))
        test_split = int(len(dataset)-train_split-valid_split)
        train, validation, test = random_split(dataset, [train_split,valid_split,test_split])
        train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True)
        validation_loader = DataLoader(validation, batch_size=args.batchsize, shuffle=False, drop_last=True)
        test_loader = DataLoader(test, batch_size=args.batchsize, shuffle=False, drop_last=True)

        logger.info(f'''Starting training:
            Epochs:          {args.epochs}
            Batch size:      {args.batchsize}
            Learning rate:   {args.lr}
            Training size:   {len(train_loader)}
            Validation size: {len(validation_loader)}
            Test size:       {len(test_loader)}
            Device:          {device.type}
            Image scaling:   {args.scale}
        ''')

        train_losses = []
        validation_losses = []
        for epoch in range(c_epoch+1, (c_epoch + args.epochs + 1)):
            train_loss = train_model(epoch)
            validation_loss = validate_model(epoch)
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
        test_model(epoch)

        plt.plot(train_losses)
        plt.plot(validation_losses)
        plt.show()

    # if training is interrupted, save the currect model state
    except KeyboardInterrupt:
        save_path = f'models/checkpoints/INTERRUPTED_epoch{epoch}.tar'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'average_loss': 0
            }, save_path)
        logger.info('Saved interrupt')
