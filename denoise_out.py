import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
from model.base_network import BDNet

import time

parser = argparse.ArgumentParser(description='Image Restoration')
parser.add_argument('--input_dir', default='./your/input/path', type=str, help='Input images')
parser.add_argument('--result_dir', default='./your/output/path', type=str, help='Directory for results')
parser.add_argument('--weights',default='./checkpoints/your/model/path', type=str, help='Path to weights')

args = parser.parse_args()

# Load corresponding model architecture and weights
model = BDNet

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
                  + glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.PNG')))

if len(files) == 0:
    raise Exception("No files found at {}".format(inp_dir))

model.cuda()
load_checkpoint(model, args.weights)
model.eval()

print('Restoring images......')

time1 = time.time()
nums = 0 
for file_ in files:
    img = Image.open(file_).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    with torch.no_grad():
        restored = model(input_)
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

    restored = img_as_ubyte(restored[0])

    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img((os.path.join(out_dir, f + '.png')), restored)

    nums += 1
print('Each images use {.2f} seconds '.format((time.time()-time1)/nums))
print("Files saved at {}".format(out_dir))