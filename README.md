Diffusion-Based Ensemble Deep Learning for Lung Cancer Detection

This project demonstrates a hybrid deep learning approach for the classification of lung cancer from CT scan images. It leverages a UNet-style diffusion model for deep feature extraction and uses multiple classifiers whose predictions are combined through ensemble learning for maximum accuracy and robustness.



## Project Highlights

- Feature mining using a trained diffusion model.
- Multi-layered feature extraction (downsampling, bottleneck, upsampling).
- Training of multiple classifiers (MLP-256, MLP-512, MLP-1024, deep classifier, and simple classifier).
- Soft voting-based ensemble classifier.
- Performance visualization and comparison with existing methods.



## Steps to Execute

### 1. Prepare the Dataset
- Create a dataset directory named `dataset` inside your Google Drive.
- Structure it with three subfolders: `train`, `valid`, and `test`.
- Inside each, add two subfolders representing the classes:
  - `cancerous`
  - `normal`

> Your structure should look like:  
    `/MyDrive/dataset/train/cancerous/`, etc.


### 2. Set Up the Environment
- Open `Diffusion_feature.ipynb` in [Google Colab](https://colab.research.google.com/).
- Ensure you're connected to a **GPU runtime** (preferably **Tesla T4 or A100**).
- Required libraries:
  - `torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`, `Pillow`
All dependencies are pre-installed in Colab.


### 3. Train the Diffusion Model
- Run the training section to build a UNet-like diffusion model.
- The model learns to extract latent features using 1000-step forward and reverse noise processing.
- Best model weights are saved to `/MyDrive/lung_cancer_models/`.


### 4. Extract Features
- After training, extract features from the CT scans using the trained diffusion model.
- Features are extracted from three levels: `down3`, `bottleneck`, and `up1`.
- These are concatenated and flattened into 1D vectors.


### 5. Train Individual Classifiers
- Use the extracted features to train five different classifiers:
  - MLP with 256, 512, and 1024 units
  - Deep multi-layer MLP
  - Simple linear classifier
- Each model is trained for 50 epochs and the best validation weights are saved.


### 6. Train and Evaluate the Ensemble
- All five classifiers are used in an ensemble using soft voting on probabilities.
- Evaluation metrics include:
  - Accuracy
  - Precision and Recall
  - F1-score
  - Confusion Matrix
  - ROC Curve


### 7. Visualize the Results
- Run the plotting cells to generate:
  - Training loss and accuracy curves
  - Confusion matrix
  - ROC curve with AUC
  - Bar chart comparing classifiers
  - Line chart comparing with existing literature


### 8. Output
- All trained models are saved in:
  - `/MyDrive/lung_cancer_models/`
- All plots are shown inline in the notebook and can be exported manually.
- Final results and metrics can be copied into your report or presentation.





