# Channel Split Convolutional Neural Network (ChaSNet) for Thermal ImageSuper-Resolution

The repository contains the official code for the work **"Channel Split Convolutional Neural Network (ChaSNet) for Thermal ImageSuper-Resolution"** accepted for PBVS-2021 workshop in-conjuction with CVPR-2021 conference.

**- Description**

<img src="Images/Network.png" width="800">
<img src="Images/Track2Net.png" width="800">
<img src="Images/CB.png" width="800">

**- Result**

(* x2 results are taken on LR images obtained by bicubic downscaling on MR data)

|Method |x4 PSNR |x4 SSIM |x2 PSNR |x2 SSIM |
|----- |----- |----- |----- |----- |
|Bicubic |32.66 |0.8625 |34.74 |0.9200 |
|SRResNet |33.12 |0.9018 |33.66 |0.9229 |
|MSRN |34.47 |0.9076 |36.96 |0.9471 |
|SRFeat |34.12 |0.9007 |- |- |
|EDSR |34.48 |0.9068 |36.91 |0.9466 |
|RCAN |34.42 |0.9072 |36.67 |0.9438 |
|TEN |33.62 |0.8910 |36.10 |0.9392 |
|CNN-IR |33.77 |0.8938 |36.66 |0.9438 |
|PBVS-2020 winner |34.49 |0.9073 |- |- |
|TherISuRNet |34.49 |0.9101 |36.76 |0.9450 |
|Proposed |34.86 |0.9133 |37.38 |0.9509 |
|Proposed+ |**34.90** |**0.9134** |**37.49** |**0.9518** |

**- Pre-Trained models**

The pre-trained model for track-2 (i.e. scaling of x2) is shared with the repository while the pre-trained model for the track-1 (i.e. scaling of x4) can be downloaded from the [link](https://drive.google.com/file/d/1jpCZn1bDX2qSUKYc_2Q5bpBGsF_p0sCr/view?usp=sharing).

**- Training the model**

To train from scratch, you need to set root directory and dataset directory into `options/train/train_vkgenPSNR.json` file. Then run the following command to start the training.
```javascript
python train.py -opt PATH-to-json_file

```

**- Testng the model**

To test your pre-trained model, you need to set root directory and dataset directory into `options/test/test_VKGen.json` file. Then run the following command to start the training.
```javascript
python test.py -opt PATH-to-json_file

```
To get the SR images using self-assemble technique, you need to run the following line of code.
```javascript
python self_assemble_test.py -opt PATH-to-json_file

```

**- Requirement of packages**

The list of packages required to run the code is given in `chasnet.yml` file.

## Contributors

- Kalpesh Prajapati <kalpesh.jp89@gmail.com>
