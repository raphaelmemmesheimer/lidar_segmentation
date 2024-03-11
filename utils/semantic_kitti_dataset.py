# This file contains the code to load the SemanticKITTI dataset
import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from tqdm.auto import tqdm
from matplotlib import pyplot as plt


def combine_images(folder, target_folder="."):
    # iterate over all images in the folder
    # combine the images into a single multi-channel image and save it
    
    # images to combine have the follwing names
    # <sequence>_<scan_no>.bin_range.png
    # <sequence>_<scan_no>.bin_remission.png
    # <sequence>_<scan_no>.bin_xyz.png
    
    # get files with the given ending
    range_files = sorted([f for f in os.listdir(folder) if f.endswith('bin_range.png')])
    remission_files = sorted([f for f in os.listdir(folder) if f.endswith('bin_remission.png')])
    xyz_files = sorted([f for f in os.listdir(folder) if f.endswith('bin_xyz.png')])

    inst_label_files = sorted([f for f in os.listdir(folder) if f.endswith('inst_label.png')])
    sem_label_files = sorted([f for f in os.listdir(folder) if f.endswith('sem_label.png')])

    # iterate over files and combine
    # 64x1024x5 (range, remission, xyz)
    combined = np.zeros((len(range_files), 64, 1024, 5), dtype=np.float32)
    combined_labels = np.zeros((len(range_files), 64, 1024, 2), dtype=np.float32)
    for i in tqdm(range(len(range_files))):
        range_img = Image.open(os.path.join(folder, range_files[i])) # 1 channel
        # get type of image
        #print(range_img.size, range_img.mode)
        remission_img = Image.open(os.path.join(folder, remission_files[i])) # 1 channel
        xyz_img = Image.open(os.path.join(folder, xyz_files[i])) # 3 channels
        #print(range_img.size, remission_img.size, xyz_img.size)
        combined[i, :, :, 0] = np.array(range_img)
        combined[i, :, :, 1] = np.array(remission_img)
        combined[i, :, :, 2:5] = np.array(xyz_img)

        inst_label_img = Image.open(os.path.join(folder, inst_label_files[i]))
        sem_label_img = Image.open(os.path.join(folder, sem_label_files[i]))
        combined_labels[i, :, :, 0] = np.array(inst_label_img)
        combined_labels[i, :, :, 1] = np.array(sem_label_img)

    # save numpy array
    print('Saving combined images of shape:', combined.shape, 'to', target_folder)
    np.save(os.path.join(target_folder, 'combined.npy'), combined)
    # saving labels
    print('Saving combined labels of shape:', combined_labels.shape, 'to', target_folder)
    np.save(os.path.join(target_folder, 'combined_labels.npy'), combined_labels)

class ToTensor(object):
    def __call__(self, img, mask):
        img = F.to_tensor(img)
        mask = F.to_tensor(mask)
        return img, mask

class SemanticKittiDataset(data.Dataset):
    def __init__(self, root, split="train", transform=None):
        # TODO: implement split
        self.root = root
        self.transform = transform

        data = np.load(os.path.join(root, 'combined.npy'))
        self.images = data
        label_data = np.load(os.path.join(root, 'combined_labels.npy'))
        self.labels = label_data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx]
        mask = self.labels[idx]

        # change shape to CxHxW
        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

    def show(self, idx):
        print('Image shape:', self.images[idx].shape)
        print('Label shape:', self.labels[idx].shape)
        img = self.images[idx]
        mask = self.labels[idx]
        plt.figure(figsize=(10, 10))

        fig, ax = plt.subplots(5, 2)
        ax[0, 0].imshow(img[:, :, 0], cmap='turbo')
        ax[0, 0].set_title('Range')

        ax[1, 0].imshow(img[:, :, 1], cmap='turbo')
        ax[1, 0].set_title('Remission')

        ax[2, 0].imshow(img[:, :, 2], cmap='turbo')
        ax[2, 0].set_title('X')

        ax[3, 0].imshow(img[:, :, 3], cmap='turbo')
        ax[3, 0].set_title('Y')

        ax[4, 0].imshow(img[:, :, 4], cmap='turbo')
        ax[4, 0].set_title('Z')


        ax[0, 1].imshow(mask[:, :, 0], cmap='tab20')
        ax[0, 1].set_title('Instance Label')

        ax[1, 1].imshow(mask[:, :, 1], cmap='tab20')
        ax[1, 1].set_title('Semantic Label')

        plt.show()

#combine_images('../coco/data/images')
