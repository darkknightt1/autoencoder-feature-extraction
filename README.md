# 🧠 Autoencoder-Based Feature Extraction with KNN Classification

## 📌 Project Overview

This project explores **vanilla**, **variational**, and **convolutional autoencoders** as **feature extractors** for face recognition using the **CASIA Face Dataset**. The encoder output of each autoencoder serves as the **latent representation** for classification via a **K-Nearest Neighbors (KNN)** classifier.

The pipeline includes:
- Tuning autoencoder architectures via **cross-validation**
- Selecting the best model based on **reconstruction loss**
- Using the **encoder output as feature vectors** for KNN classification

---


## 🧪 Methodology

1. **Data Preparation**  
   - Resize & normalize CASIA face images  
   - Store in `.npz` format for efficient loading

2. **Autoencoder Architecture Tuning**  
   - Run `*_cross.py` scripts to test various encoder/decoder layer combinations  
   - Select model with best **reconstruction accuracy/loss**

3. **Feature Extraction & Classification**  
   - Retrain the best AE model using full data  
   - Use encoder output as input features to a **KNN classifier**  
   - Evaluate classification accuracy
