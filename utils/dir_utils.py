import os
from natsort import natsorted
from glob import glob
import torch


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_last_path(path, session):
    x = natsorted(glob(os.path.join(path, '*%s' % session)))[-1]
    return x

def model_save(epoch,model_restored,optimizer,psnr_val,best_psnr,ssim_val,best_ssim,model_dir):
    # Save the best PSNR model of validation
    if psnr_val > best_psnr:
        best_psnr = psnr_val
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'psnr':psnr_val,
                    'ssim':ssim_val
                    }, os.path.join(model_dir, "model_bestPSNR.pth"))

    # Save the best SSIM model of validation
    if ssim_val > best_ssim:
        best_ssim = ssim_val
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'psnr':psnr_val,
                    'ssim':ssim_val
                    }, os.path.join(model_dir, "model_bestSSIM.pth"))
    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr':psnr_val,
                'ssim':ssim_val
                }, os.path.join(model_dir, "model_latest.pth"))
    
    return best_psnr,best_ssim