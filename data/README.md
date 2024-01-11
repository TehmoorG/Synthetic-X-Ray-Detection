# Data Directory

## Overview
This directory is intended to hold the dataset used for the Synthetic X-Ray Detection project. 

## Data Availability
Due to constraints such as file size, the complete dataset is not included directly in this repository. Instead, the dataset can be accessed and downloaded from the following link:

[Access the Complete Dataset](https://drive.google.com/drive/folders/14avVgn52_uT-vuUKFml18YUilHVyNpVv?usp=drive_link)

## Dataset Structure
Upon downloading, the dataset should contain images named as follows:

- Real hand X-rays: `real_hand_XXXX.jpeg`
- VAE generated hand X-rays: `vae_hand_XXXX.jpeg`
- GAN generated hand X-rays: `gan_hand_XXXX.jpeg`

Here, `XXXX` represents the image numbering which you can use to reference specific images.

## Usage
After downloading, place the images in this directory to use them with the notebooks and scripts in the repository.