import torch
import torch.nn as nn

import torchattacks

from torchvision import models
from utils import get_imagenet_data, get_accuracy
from torchattacks import OnePixel

class ResNet152OnePixel():
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    ROOT = './imagenet-1k-mini/images'
    INDEX = './imagenet-1k-mini/imagenet_class_index.json'
    SAVE_PATH = './resnet152_one_pixel.pt'

    def __init__(self):
        self.images = None
        self.labels = None

        self.device = None
        self.model = None

        self.adv_images = None

        self.atk = None

        self.load_data()
        self.load_model()

    def load_data(self):
        print('[Loading data...]')
        self.images, self.labels = get_imagenet_data(mean=self.MEAN, std=self.STD, root=self.ROOT, index=self.INDEX)
        print('[Data loaded]')

    def load_model(self):
        print('[Loading model...]')
        self.device = "cuda"
        self.model = models.resnet152(pretrained=True).to(self.device).eval()
        print('[Model loaded]')

    def attack(self):
        self.atk = OnePixel(self.model, pixels=1)
        self.atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        print(self.atk)
        print('[Attacking...]')
        self.adv_images = self.atk(self.images, self.labels)
        print('[Done attacking]')

    def get_accuracy(self):
        # Percentage of correctly classified samples
        acc = get_accuracy(self.model, [(self.images.to(self.device), self.labels.to(self.device))])
        print('Acc: %2.2f %%'%(acc)) # Accuracy before attack should be 100%

    def save_adv_images(self):
        self.atk.save(data_loader=[(self.images.to(self.device), self.labels.to(self.device))], save_path=self.SAVE_PATH, verbose=True)
        print('[Adversarial images saved]')

if __name__ == '__main__':
    resnet152_one_pixel = ResNet152OnePixel()
