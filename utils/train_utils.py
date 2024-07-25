import os
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from utils import model_save,network_parameters
from datasets.dataset import get_training_data, get_validation_data,get_training_data_control, get_validation_data_control
from torch.utils.data import DataLoader
import utils
from datetime import datetime

def set_gpus(model,opt):
    gpus = ','.join([str(i) for i in opt['GPU']])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    if len(device_ids) > 1:
        model_restored = nn.DataParallel(model_restored, device_ids=device_ids)
    return model, device_ids


def model_resume(model_restored,scheduler,optimizer,model_dir):
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')
    return model_restored, scheduler, optimizer


def dataloader_set(train_dir,val_dir,opt):
    print('==> Loading datasets')
    train_dataset = get_training_data(train_dir)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt['OPTIM']['BATCH'],
                            shuffle=True, num_workers=0, drop_last=False)
    val_dataset = get_validation_data(val_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, 
                            shuffle=False, num_workers=0,drop_last=False)
    return train_loader,val_loader


def control_dataloader_set(train_dir,val_dir,opt):
    print('==> Loading datasets')
    train_dataset = get_training_data_control(train_dir)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt['OPTIM']['BATCH'],
                            shuffle=True, num_workers=0, drop_last=False)
    val_dataset = get_validation_data_control(val_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, 
                            shuffle=False, num_workers=0,drop_last=False)
    return train_loader,val_loader



def print_configuration(opt,p_number,start_epoch,device_ids):
    print(f'''
    ------------------------------------------------------------------
        Datetime:           {datetime.now()}
        Restoration mode:   {opt['MODEL']['MODE']}
        Train patches size: {str(opt['TRAINING']['TRAIN_PS']) + 'x' + str(opt['TRAINING']['TRAIN_PS'])}
        Val patches size:   {str(opt['TRAINING']['VAL_PS']) + 'x' + str(opt['TRAINING']['VAL_PS'])}
        Model parameters:   {p_number}
        Start/End epochs:   {str(start_epoch) + '~' + str(opt['OPTIM']['EPOCHS'])}
        Batch sizes:        {opt['OPTIM']['BATCH']}
        Learning rate:      {opt['OPTIM']['LR_INITIAL']}
        GPU:                {'GPU' + str(device_ids)}''')
    print('------------------------------------------------------------------')


def train_model(model_restored,train_loader,val_loader,scheduler,optimizer
                ,start_epoch,opt,Loss,device_ids,model_dir):
    p_number = network_parameters(model_restored)
    print_configuration(opt,p_number,start_epoch,device_ids)
    # Start training!
    print('==> Training start: ')
    best_psnr = 0
    best_ssim = 0
    best_epoch_psnr = 0
    best_epoch_ssim = 0
    total_start_time = time.time()

    for epoch in range(start_epoch, opt['OPTIM']['EPOCHS'] + 1):
        epoch_start_time = time.time()
        epoch_loss = 0

        model_restored.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            # Forward propagation
            for param in model_restored.parameters():
                param.grad = None
            target, input = data[0].cuda(), data[1].cuda()

            restored = model_restored(input)

            # Compute loss
            loss = Loss(restored, target)

            # Back propagation
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        ## Evaluation (Validation)
        if epoch % opt['TRAINING']['VAL_AFTER_EVERY'] == 0:
            model_restored.eval()
            psnr_val = []
            ssim_val = []
            for i, data_val in enumerate(val_loader, 0):
                target = data_val[0].cuda()
                input = data_val[1].cuda()
                with torch.no_grad():
                    restored = model_restored(input)

                for res, tar in zip(restored, target):
                    psnr_val.append(utils.torchPSNR(res, tar))
                    ssim_val.append(utils.torchSSIM(restored, target))

            psnr_val = torch.stack(psnr_val).mean().item()
            ssim_val = torch.stack(ssim_val).mean().item()

        best_psnr,best_ssim = model_save(epoch,model_restored,optimizer,psnr_val,best_psnr,ssim_val,best_ssim,model_dir)
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val, best_epoch_psnr, best_psnr))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val, best_epoch_ssim, best_ssim))
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.2f}\tLoss: {:.2f}\tLearningRate {:.6f}"
            .format(epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

    total_finish_time = (time.time() - total_start_time) 
    print('Total training time: {:.3f} hours'.format((total_finish_time / 60 / 60)))


def train_control_model(model_restored,train_loader,val_loader,scheduler,optimizer
                ,start_epoch,opt,Loss,device_ids,model_dir):
    p_number = network_parameters(model_restored)
    print_configuration(opt,p_number,start_epoch,device_ids)

    # Start training!
    print('==> Training start: ')
    best_psnr = 0
    best_ssim = 0
    best_epoch_psnr = 0
    best_epoch_ssim = 0
    total_start_time = time.time()

    for epoch in range(start_epoch, opt['OPTIM']['EPOCHS'] + 1):
        epoch_start_time = time.time()
        epoch_loss = 0

        model_restored.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            # Forward propagation
            for param in model_restored.parameters():
                param.grad = None

            target, input_, hint_ = data[0].cuda(), data[1].cuda(), data[2].cuda()
            restored = model_restored(input_,hint_)

            # Compute loss
            loss = Loss(restored, target)

            # Back propagation
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        ## Evaluation (Validation)
        if epoch % opt['TRAINING']['VAL_AFTER_EVERY'] == 0:
            model_restored.eval()
            psnr_val = []
            ssim_val = []
            for ii, data_val in enumerate(val_loader, 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                hint_ = data_val[2].cuda()

                with torch.no_grad():
                    restored = model_restored(input_,hint_)

                for res, tar in zip(restored, target):
                    psnr_val.append(utils.torchPSNR(res, tar))
                    ssim_val.append(utils.torchSSIM(restored, target))

            psnr_val = torch.stack(psnr_val).mean().item()
            ssim_val = torch.stack(ssim_val).mean().item()


        best_psnr,best_ssim = model_save(epoch,model_restored,optimizer,psnr_val,best_psnr,ssim_val,best_ssim,model_dir)
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val, best_epoch_psnr, best_psnr))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val, best_epoch_ssim, best_ssim))

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.2f}\tLoss: {:.2f}\tLearningRate {:.6f}".
            format(epoch, time.time() - epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

    total_finish_time = (time.time() - total_start_time)
    print('Total training time: {:.3f} hours'.format((total_finish_time / 60 / 60)))



