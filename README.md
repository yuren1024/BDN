#[BFCNet]

***
> Abstract : Image restoration is a challenging ill-posed problem
Advanced spectral imaging techniques have found broad-ranging applications by offering new dimensions of information, enabling the discovery of novel features. However, the inherent limitation of noise, often obscuring these features, poses a significant barrier to the further advancement of spectral imaging technology. Traditional deep learning approaches, despite their extensive use of large datasets, falter in processing images from routine imaging scenarios characterized by varied samples and diverse imaging conditions. To address these challenges, we propose BiFormer Control Network (BFCNet) to extract sample features from image noises at two scales: local and global level. The sparsity design of BFCNet enables effective training by few-shot datasets commonly encountered in routine imaging scenarios. As demonstrated in experiments, BFCNet significantly enhances image quality from single-frame images taken at low laser power, especially at high spatial frequency. It demonstrates a signal-to-noise ratio (SNR) equivalent to averaging up to 20 frames of low-power images, equivalent to the same fold improvement in imaging speed. Furthermore, the architecture of BFCNet generally applies to various imaging modalities, as experimentally demonstrated in stimulated Raman scattering (SRS) and fluorescence imaging, offering new opportunities for biomedical imaging applications.

## Train  
- Dataset:  
  The preparation of dataset in more detail, see [datasets/README.md](datasets/README.md).  
  
- Train:  
  If the above path and data are all correctly setting, just simply run:  
  ```
  python train.py
  ```
  ```
  python denoise_out.py
  ```

  ```
  python BFCNet_model_create.py
  ```

  ```
  python train_control.py
  ```
  ```
  python denoise_control_out.py
  ```


## Citation  
If you use BFCNet, please consider citing:  
```
pass
```
