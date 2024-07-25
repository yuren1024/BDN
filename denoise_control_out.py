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
from model.base_control_net import BDCNet
import time
from utils.fft_util import get_low_high_f

parser = argparse.ArgumentParser(description='Image Restoration')
parser.add_argument('--input_dir', default='./your/input/path', type=str, help='Input images')
parser.add_argument('--result_dir', default='./your/output/path', type=str, help='Directory for results')
parser.add_argument('--weights',default='./checkpoints/your/model/path', type=str, help='Path to weights')

args = parser.parse_args()

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
hint_dir = os.path.join(inp_dir[:-6],'ffthigh')
img_dir = os.listdir(inp_dir)
if not os.path.exists(hint_dir):
    os.mkdir(hint_dir)
for p in img_dir:
    sp = os.path.join(img_dir, p)   # train/input/1.png ……
    img = cv2.imread(sp,cv2.IMREAD_GRAYSCALE)
    _ , high_freq = get_low_high_f(img)
    tp = os.path.join(hint_dir,p)
    cv2.imwrite(tp,high_freq)
print(f"{hint_dir} HighFrequenc information Done!")

os.makedirs(out_dir, exist_ok=True)

files1 = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
                  + glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.PNG')))

files2 = natsorted(glob(os.path.join(hint_dir, '*.jpg'))
                  + glob(os.path.join(hint_dir, '*.JPG'))
                  + glob(os.path.join(hint_dir, '*.png'))
                  + glob(os.path.join(hint_dir, '*.PNG')))

if len(files1) == 0:
    raise Exception("No files found at {}".format(inp_dir))
if len(files2) == 0:
    raise Exception("No files found at {}".format(hint_dir))

model = BDCNet()
model.cuda()

load_checkpoint(model, args.weights)
model.eval()

print('Restoring images......')

time1 = time.time()
nums = 0
for input0,hint0 in zip(files1,files2):
    img1 = Image.open(input0).convert('L')
    input = TF.to_tensor(img1).unsqueeze(0).cuda()
    img2 = Image.open(hint0).convert('L')
    hint = TF.to_tensor(img2).unsqueeze(0).cuda()

    with torch.no_grad():
        restored = model(input,hint)
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

    restored = img_as_ubyte(restored[0])

    f = os.path.splitext(os.path.split(input0)[-1])[0]
    save_img((os.path.join(out_dir, f + '.png')), restored)

    nums += 1
print('Each images use {.2f} seconds '.format((time.time()-time1)/nums))
print("Files saved at {}".format(out_dir))