<p align="center">
  <img src="https://github.com/TehmoorG/Generative-Hand-X-ray/blob/main/data/real_hands/000000.jpeg" alt="VAE X-ray Image" width="300"/>
</p>

# Synthetic X-Ray Image classification

## Overview
This project focuses on using Deep Learning techniques, specifically Convolutional Neural Networks (CNNs), to differentiate between real and synthetic X-ray images of hands. The initiative is aimed at enhancing the understanding and application of AI in medical imaging, with potential implications for training and healthcare technology.

## Project Description
The core of this project is the development and optimization of a CNN model that classifies hand X-ray images into real or synthetic categories, where the synthetic images are generated using Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs). The repository contains detailed Jupyter notebooks covering data preprocessing, model training, hyperparameter tuning, and evaluation.

### Key Features
- Implementation of a custom CNN model for classification.
- Detailed process of model training and validation.
- Exploration of various hyperparameters and their impact on model performance.
- Analysis and classification of hand images into real or AI-generated categories.

## Repository Structure
- `notebooks/`: Contains the main Jupyter notebooks for model development and hyperparameter tuning.
- `src/`: Source code for model, dataset preparation, and training utilities.
- `model/`: Saved final CNN model after training.
- `acse-tg1523_classified_hands.csv`: Results of the model predictions on the test dataset.

## Data Availability
The dataset includes real hand X-ray images and synthetic images generated by VAEs and GANs. For detailed dataset information, refer to the provided documentation within the repository.

## Usage
To utilize this project:
1. Clone the repository.
2. Install necessary dependencies as listed in `requirements.txt`:
    ```
    pip install -r requirements.txt
    ```
3. Execute the Jupyter notebooks in the `notebooks` directory to train the model or make predictions.

## Contributions and Feedback
Contributions, suggestions, and feedback are highly encouraged. Feel free to open an issue or create a pull request for any improvements.

## License
This project is released under the MIT License. For more information, please refer to the LICENSE file.