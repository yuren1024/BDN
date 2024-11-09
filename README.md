#[BDN] Enhanced stimulated Raman and fluorescence imaging by single-frame trained BDN

***
> Abstract: pass 

## Train  
- Dataset:  
  To prepare the dataset in more detail, please take a look at [datasets/README.md](datasets/README.md).  
  
- Train:  
  If the above path and data are all correctly set, just simply run the following:  
  ```
  python train.py
  ```
  ```
  python denoise_out.py
  ```

  ```
  python BDCNet_model_create.py
  ```

  ```
  python train_control.py
  ```
  ```
  python denoise_control_out.py
  ```


## Citation  
If you use BDNet, please consider citing:  
```
X. Tang, Y. Zhang, X. Huang, H. Lee, and D. Zhang, "Enhanced stimulated Raman and fluorescence imaging by single-frame trained BDN," Opt. Express  32, 40593-40604 (2024).
```
