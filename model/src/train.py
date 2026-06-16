import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import configs
from ViT_model import IQARegression
from resnet50 import resnet50_backbone
from trainer import train_epoch, eval_epoch
from util import RandHorizontalFlip, Normalize, ToTensor, RandShuffle
from thop import profile
import torch.nn as nn
import numpy as np

config=configs.config()

# device
config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
print('Device:', config.device)

from tan_dataset import IQADataset

os.environ['CURRENT_DATASET'] = 'cviqd'

# prepare scene lists
train_scene, val_scene, test_scene = RandShuffle(config)


train_dataset = IQADataset(
    config=config,
    transform=transforms.Compose([Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), RandHorizontalFlip(), ToTensor()]),
    scene_list=train_scene
)
test_dataset = IQADataset(
    config=config,
    transform=transforms.Compose([Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), ToTensor()]),
    scene_list=val_scene
)

train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=True)


criterion = torch.nn.L1Loss()
params = list(model_backbone.parameters()) + list(model_transformer.parameters())

optimizer = torch.optim.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

start_epoch = 0
if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
    model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']

if not os.path.exists(config.snap_path):
    os.makedirs(config.snap_path, exist_ok=True)

for epoch in range(start_epoch, config.n_epoch):
    loss, srcc, plcc, rmse = train_epoch(config, epoch, model_transformer, model_backbone, criterion, optimizer, scheduler, train_loader)
    if (epoch+1) % config.val_freq == 0:
        loss, srcc, plcc, rmse = eval_epoch(config, epoch, model_transformer, model_backbone, criterion, test_loader)
