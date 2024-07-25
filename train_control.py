import os
import torch
import yaml
from utils import set_seed,train_control_model,control_dataloader_set,model_resume,set_gpus
import torch.nn as nn
import torch.optim as optim
import utils
from warmup_scheduler import GradualWarmupScheduler
from model.base_control_net import  BDCNet

## Set Seeds
torch.backends.cudnn.benchmark = True
rand_seed = 1024
set_seed(rand_seed)

## Load yaml configuration file
with open('finetune.yaml', 'r') as config:
    opt = yaml.safe_load(config)

## Build Model
print('==> Build the model')
model_restored = BDCNet
model_restored.cuda()

## Training model path direction
mode = opt['MODEL']['MODE']
model_dir = os.path.join(opt['TRAINING']['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = opt['TRAINING']['TRAIN_DIR']
val_dir = opt['TRAINING']['VAL_DIR']

## GPU
model_restored, device_ids = set_gpus(model_restored,opt)

## Optimizer
start_epoch = 1
new_lr = float(opt['OPTIM']['LR_INITIAL'])
optimizer = optim.AdamW(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt['OPTIM']['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(opt['OPTIM']['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)
if opt['TRAINING']['RESUME']:
    model_restored, scheduler, optimizer = model_resume(model_restored,scheduler,optimizer,model_dir,opt)

## Loss
L1_loss = nn.L1Loss()

## DataLoaders
train_loader,val_loader = control_dataloader_set(train_dir,val_dir,opt)

if __name__ == '__main__':
    train_control_model(model_restored,train_loader,val_loader,scheduler,optimizer
                ,start_epoch,opt,L1_loss,device_ids,model_dir)