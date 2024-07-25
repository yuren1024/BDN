import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random
import cv2
from utils.fft_util import get_low_high_f
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, img_dir):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(img_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(img_dir, 'target')))

        self.inp_filenames = [os.path.join(img_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(img_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.sizex = len(self.tar_filenames)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('L')
        tar_img = Image.open(tar_path).convert('L')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        aug = random.randint(0, 8)

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, img_dir):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(img_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(img_dir, 'target')))

        self.inp_filenames = [os.path.join(img_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(img_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.sizex = len(self.tar_filenames) 


    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('L')
        tar_img = Image.open(tar_path).convert('L')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename

class DataLoaderTrainControl(Dataset):
    def __init__(self, img_dir):
        super(DataLoaderTrainControl, self).__init__()
        self.img_dir = img_dir
        self.get_highfft()
        inp_files = sorted(os.listdir(os.path.join(img_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(img_dir, 'target')))
        aux_files = sorted(os.listdir(os.path.join(img_dir, 'ffthigh')))

        self.inp_filenames = [os.path.join(img_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(img_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.aux_filenames = [os.path.join(img_dir, 'ffthigh', x) for x in aux_files if is_image_file(x)]

        self.sizex = len(self.tar_filenames)
        

    def __len__(self):
        return self.sizex
    
    def get_highfft(self):
        highfft_dir = os.path.join(self.img_dir, 'ffthigh')
        img_dir = os.listdir(os.path.join(self.img_dir, 'input'))
        if not os.path.exists(highfft_dir):
            os.mkdir(highfft_dir)

        for p in img_dir:
            sp = os.path.join(img_dir, p)   # train/input/1.png ……
            img = cv2.imread(sp,cv2.IMREAD_GRAYSCALE)
            _ , high_freq = get_low_high_f(img)
            tp = os.path.join(highfft_dir,p)
            cv2.imwrite(tp,high_freq)
        print(f"{highfft_dir} HighFrequenc information Done!")



    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        aux_path = self.aux_filenames[index_]

        inp_img = Image.open(inp_path).convert('L')
        tar_img = Image.open(tar_path).convert('L')
        aux_img = Image.open(aux_path).convert('L')

    
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        aux_img = TF.to_tensor(aux_img)

        aug = random.randint(0, 8)

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
            aux_img = aux_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
            aux_img = aux_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
            aux_img = torch.rot90(aux_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
            aux_img = torch.rot90(aux_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
            aux_img = torch.rot90(aux_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
            aux_img = torch.rot90(aux_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))
            aux_img = torch.rot90(aux_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, aux_img, filename


class DataLoaderValControl(Dataset):
    def __init__(self, img_dir):
        super(DataLoaderValControl, self).__init__()
        self.img_dir = img_dir
        self.get_highfft()
        inp_files = sorted(os.listdir(os.path.join(img_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(img_dir, 'target')))
        aux_files = sorted(os.listdir(os.path.join(img_dir, 'ffthigh')))

        self.inp_filenames = [os.path.join(img_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(img_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.aux_filenames = [os.path.join(img_dir, 'ffthigh', x) for x in aux_files if is_image_file(x)]
        self.sizex = len(self.tar_filenames)

    def __len__(self):
        return self.sizex

    def get_highfft(self):
        highfft_dir = os.path.join(self.img_dir, 'ffthigh')
        img_dir = os.listdir(os.path.join(self.img_dir, 'input'))
        if not os.path.exists(highfft_dir):
            os.mkdir(highfft_dir)

        for p in img_dir:
            sp = os.path.join(img_dir, p)   # train/input/1.png ……
            img = cv2.imread(sp,cv2.IMREAD_GRAYSCALE)
            _ , high_freq = get_low_high_f(img)
            tp = os.path.join(highfft_dir,p)
            cv2.imwrite(tp,high_freq)
        print(f"{highfft_dir} HighFrequenc information Done!")

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        aux_path = self.aux_filenames[index_]

        inp_img = Image.open(inp_path).convert('L')
        tar_img = Image.open(tar_path).convert('L')
        aux_img = Image.open(aux_path).convert('L')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        aux_img = TF.to_tensor(aux_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, aux_img, filename


def get_training_data(img_dir):
    assert os.path.exists(img_dir)
    return DataLoaderTrain(img_dir)


def get_validation_data(img_dir):
    assert os.path.exists(img_dir)
    return DataLoaderVal(img_dir)


def get_training_data_control(img_dir):
    assert os.path.exists(img_dir)
    return DataLoaderTrainControl(img_dir)


def get_validation_data_control(img_dir):
    assert os.path.exists(img_dir)
    return DataLoaderValControl(img_dir)
