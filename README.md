# super_resolution

[![Tests](https://github.com/sing-lab/super-resolution/workflows/Tests/badge.svg)](https://github.com/sing-lab/super-resolution/actions?workflow=Tests)

[![codecov](https://codecov.io/gh/sing-lab/super-resolution/branch/master/graph/badge.svg?token=1DDPH1JK1Q)](https://codecov.io/gh/sing-lab/super-resolution)

[![License](https://img.shields.io/pypi/l/cookiecutter-hypermodern-python-instance)][license]

[![Read the documentation at https://super-resolution.readthedocs.io](https://readthedocs.org/projects/super-resolution/badge/?version=latest)][read the docs]

[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]

[read the docs]: https://super-resolution.readthedocs.io
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[license]: https://github.com/sing-lab/super-resolution/blob/master/LICENSE.rst

# 1. Preamble
This repo is the python implementation of [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

SRGAN assumes an ideal **bicubic down-sampling kernel**, which is different from real degradations. Other models such as
[ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219) or [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833) use
different degradations to improve super resolution performances.

## Differences with original SRGAN paper

In addition to the SRGAN bicubic interpolation, we added to generate low resolution images from high resolution images:
- Random JPEG compression
- Random horizontal or vertical flip

However, as stated in [A Hybrid Approach Between Adversarial Generative Networks and Actor-Critic
Policy Gradient for Low Rate High-Resolution Image Compression](https://arxiv.org/abs/1906.04681) article, when using jpeg compression on low resolution images, the
drastic down-sampling of JPEG image causes loss of information, difficult to recover from the super resolution network,
which leads to lower results in PSNR. It was then only used for experimentation.

High resolution images are size 128x128 as it gives better result than 96x96.

Pixel shuffle layer may produce **checkerboard artifacts**. A better initialisation of convolutional layer kernel help reduce these
artifacts. To remove unpleasant checkerboard artifacts from reconstructed images, we used ICNR initialisation for the
subpixel convolutional block, see [this article](https://arxiv.org/abs/1707.02937).

Finally, SRGAN loss contains both MSE and VGG loss where in the paper only the VGG loss is used. Using the MSE
allows better reconstructed images: contrast is better and **batch normalization artifacts** are removed. Note that batch
normalization artifacts occurred during experiments for `0.9*MSE + 0.1*0.006*VGG_loss`, but were not present for
`0.9*MSE + 0.1*0.001*VGG_loss`. Therefore, VGG loss coefficient was set to `0.1 * 0.001`.

## Illustrations

![checkerboard_artifacts.png](illustrations%2Fcheckerboard_artifacts.png)

*Example of checkerboard artifacts when no MSE loss is used on the perceptual loss*

![batch_norm_artifact.png](illustrations%2Fbatch_norm_artifact.png)

*Example of batch normalization artifacts with VGG loss coefficient of `0.1*0.006`, explanations on [ESRGAN paper](https://arxiv.org/pdf/1809.00219.pdf).*

## Loss
SRRESNET is trained using only the **MSE loss**.

MSE loss is not able to deal with high frequency content in the image, that resulted in producing overly smooth images.
Therefore, the authors of the paper decided to  use loss of different VGG layers. This VGG loss is based on the ReLU activation layers of the pre-trained 19 layer VGG network

SRGAN generator loss is made of:
- **Perceptual loss**: `0.1*0.001*VGG_loss + 0.9*MSE`, the sum of the VGG loss (MSE on VGG space) and pixel loss (MSE loss on image space). VGG loss only
leads to unpleasant artifacts (checkerboard artifacts) as well as color shifting. Weights for
each are inspired from [this article](https://iopscience.iop.org/article/10.1088/1742-6596/1903/1/012050/pdf).
- **Generator adversarial loss**

SRGAN discriminator loss is only the **adversarial loss**.

# 2. Installation

1. Clone the projet

2. Download trained models from [here](https://drive.google.com/drive/folders/160z6A1eE5ye-JjZcOljNUUA-1o95xTg3?usp=sharing)
and add them into:
   - **[models/SRGAN](models/SRGAN)** folder for the SRGAN generator (`generator_epoch_28.torch`)
   - **[models/SRRESNET](models/SRRESNET)** folder for the SRResnet pretrained generator (`generator_epoch_204.torch`)

3. Project requirements can be installed with `poetry install`

# 3. Demo app
You can run the project demo locally or using docker.

## Run in a docker container

- Install docker desktop.

- In a terminal, in the project root, run:

  `docker-compose run --service-ports --rm app`


Then access the demo app [here](http://localhost:8000)

Note: url shown in the terminal won't work as it is relative to the Docker container it’s running in.

## Run locally

To run the app locally:

    poetry run streamlit run api/app/main.py

To use GPU, NVIDIA cuda driver must be installed.

Predict the whole image in one run may lead to out of memory for big images. Prediction is then made **tile by tile**.

# 4. Train, test or predict with the full project
You can run the project locally or using docker.
Project can be used to train, predict or test a model, using the correct **configuration file**.


## Download dataset
You need to download the dataset and put in under folders defined in the config files.
Dataset is from [COCO2017](https://cocodataset.org/#download):
  - 40.7K test images
  - 118K train images
  - 123K unlabeled images
  - 5000 val images

We use both train, unlabeled, and test split to train the model, and use another test set to get performances.
Therefore, the train set is made of 282K images and the validation set of 5K images.

**Linux / WSL**:
- Run `sudo apt-get install unzip`
- Run `sh get_dataset.sh` in the terminal to automatically download and extract dataset in the folders defined in
default configs files.

**Windows**:
- Download each split:
  - [eval](http://images.cocodataset.org/zips/val2017.zip)
  - [train](http://images.cocodataset.org/zips/train2017.zip)
  - [test](http://images.cocodataset.org/zips/test2017.zip)
  - [unlabeled](http://images.cocodataset.org/zips/unlabeled2017.zip)


- Extract `train2017.zip`, `unlabeled2017.zip` `test2017.zip` into `data/raw/train`
- Extract `val2017.zip` into `data/raw/val`
- Download [BSD100](https://figshare.com/ndownloader/files/38256840)
- Download [Set5](https://figshare.com/ndownloader/files/38256852)
- Download [Set14](https://figshare.com/ndownloader/files/38256855)
- Extract the 3 zip folders into `data/raw/test`
- Remove all images containing 'LR' from
  - `data/raw/test/BSD100/image_SRF_4`
  - `data/raw/test/Set5/image_SRF_4`
  - `data/raw/test/Set14/image_SRF_4`
- Move all images from `image_SRF_4` to parent folder for each `BSD100`, `Set5`, `Set14`
- Remove `image_SRF_2`, `image_SRF_3`, and `image_SRF_4` folder for each `BSD100`, `Set5`, `Set14`

## 1. Run locally
- Create your configuration file in the [configs](configs) folder, or use one of the existing config file.

- Run the project with `poetry run super_resolution $config_path`

Example:
- to train a model: `poetry run super_resolution configs/SRGAN/srgan_train_config.yml`

  Training can be monitored using tensorboard: `poetry run tensorboard --logdir logs/$EXPERIMENT_NAME`


- to test a model:`poetry run super_resolution configs/SRGAN/srgan_test_config.yml`


- to predict using a model: `poetry run super_resolution configs/SRGAN/srgan_predict_config.yml`


## 2. Run in a docker container
To run the project via a docker container, set up a config file then:

`docker-compose run main "$config_path"`

Examples:
1. To train a model:

`docker-compose run --rm main "configs/SRGAN/srgan_train_config.yml"`

After training is done, you can find:
- logs at `logs/SRGAN/$EXPERIMENT_NAME`
- models at `models/SRGAN/$EXPERIMENT_NAME`
- images at `data/processed/val/SRGAN/$EXPERIMENT_NAME`

Or in the directory specified in the `config_file` you used.

Training can be then monitored with:
`tensorboard --logdir=logs/$EXPERIMENT_NAME`

Note: it may require to install tensorboard first with `pip install tensorboard`

2. To test a model and get performances:

`docker-compose run --rm main "configs/SRGAN/srgan_test_config.yml"`

- Test data must exist in the location specified in the config file.
- Predicted data will be saved in the location specified in the config file.
- Average PSNR and SSIM for each folder are then displayed in the terminal.

3. To super-resolve images using a model:

`docker-compose run --rm main "configs/SRGAN/srgan_predict_config.yml"`

- Input data must exist in the location specified in the config file.
- Predicted data will be saved in the location specified in the config file.

Prediction can be done tile by tile to adapt to smaller GPU, cf [srgan_predict_config](configs/SRGAN/srgan_predict_config.yml)

# Performances

SRRESNET model was trained on 204 epochs on 158K images (COCO2017 unlabeled set was not used for SRRESNET training).

SRGAN was trained on 30 epochs on 282K images. After 15 epoch, the learning rate was divided by 10.
Epoch 29 (therefore checkpoint_28.torch, as indexes start at 0) gives the best results (around 250 000 iterations), based on empirical analysis of super resolved validation images at each epoch.

Hyperparameters used to trained models are the same as the ones in the default configuration files.
- `jpeg_compression` was not used for SRGAN training (cf Preamble section) as generator struggle to recover lost information. Empirically, it gave
worst results (for SRGAN: PSNR ~ 23, SSIM ~ 0.60).
- SRGAN perceptual loss `0.1*0.001*VGG_loss + 0.9*MSE` (cf Loss section).

SRGAN checkpoint is [available](https://drive.google.com/drive/folders/160z6A1eE5ye-JjZcOljNUUA-1o95xTg3?usp=sharing) to resume training for SRGAN. To use it, download it and add its path in the srgan training configuration file.
In addition, training logs for SRGAN are available (compatible with tensorboard).

## Results
Here are the results (with the paper's results in parantheses):

|              |     PSNR      |      SSIM      ||     PSNR      |      SSIM      ||     PSNR      |      SSIM      |
|:------------:|:-------------:|:--------------:|:-------------:|:--------------:|:-------------:|:--------------:|:---:|:---:|
| **SRResNet** | 31.40 (32.05) | 0.895 (0.9019) || 27.98 (28.49) | 0.782 (0.8184) || 27.01 (27.58) | 0.741 (0.7620) |
|  **SRGAN**   | 30.64 (29.40) | 0.870 (0.8472) || 26.91 (26.02) | 0.741 (0.7397) || 25.91 (25.16) | 0.692 (0.6688) |
|              |   **Set5**    |    **Set5**    ||   **Set14**   |   **Set14**    ||  **BSD100**   |   **BSD100**   |


## Illustrations

From left to right: low resolution, super resolved an original image.

SRGAN:
![epoch_28_image_0.png](illustrations%2FSRGAN%2Fepoch_28_image_0.png)
![epoch_28_image_500.png](illustrations%2FSRGAN%2Fepoch_28_image_500.png)
![epoch_28_image_1500.png](illustrations%2FSRGAN%2Fepoch_28_image_1500.png)
![epoch_28_image_2000.png](illustrations%2FSRGAN%2Fepoch_28_image_2000.png)
![epoch_28_image_2500.png](illustrations%2FSRGAN%2Fepoch_28_image_2500.png)

SRRESNET:
![epoch_1_image_0.png](illustrations%2FSRRESNET%2Fepoch_1_image_0.png)
![epoch_1_image_1.png](illustrations%2FSRRESNET%2Fepoch_1_image_1.png)
![epoch_1_image_2.png](illustrations%2FSRRESNET%2Fepoch_1_image_2.png)
![epoch_1_image_3.png](illustrations%2FSRRESNET%2Fepoch_1_image_3.png)
![epoch_1_image_4.png](illustrations%2FSRRESNET%2Fepoch_1_image_4.png)
![epoch_1_image_5.png](illustrations%2FSRRESNET%2Fepoch_1_image_5.png)


![SRGAN_training.PNG](illustrations%2FSRGAN_training.PNG)

*SRGAN training logs.*

# Contributing

Contributions are very welcome.
To learn more, see the `Contributor Guide`.

# License

Distributed under the terms of the `MIT license`,
_super_resolution_ is free and open source software.

# Issues

If you encounter any problems,
please `file an issue` along with a detailed description.

# Credits

This project was generated from `@cjolowicz's` `Hypermodern Python Cookiecutter` template.
This repo is inspired from the following GitHub repo [a-PyTorch-Tutorial-to-Super-Resolution](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution).
