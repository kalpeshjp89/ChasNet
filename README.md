The repository contains the official code for the work **"Channel Split Convolutional Neural Network (ChaSNet) for Thermal ImageSuper-Resolution"** accepted for PBVS-2021 workshop in-conjuction with CVPR-2021 conference.

<img src="Images/Network.png" width="800">
<img src="Images/Track2Net.png" width="800">
<img src="Images/CB.png" width="800">


**-Pre-Trained models**
The pre-trained model for track-2 (i.e. scaling of x2) is shared in the repository while the pre-trained model for the track-1 (i.e. scaling of x4) can be download from the [link](https://drive.google.com/file/d/1jpCZn1bDX2qSUKYc_2Q5bpBGsF_p0sCr/view?usp=sharing).

**-Training the model**
To train from scratch, you need to set root directory and dataset directory into 'options/train/train_vkgenPSNR.json' file. Then run the following command to start the training.
'''javascript
python train.py -opt PATH-to-json_file
'''

## Contributors
- Kalpesh Prajapati <kalpesh.jp89@gmail.com>
