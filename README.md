# Learning-Based Image Denoising Using CNNs, Residual Networks, and Diffusion Models

This repository contains the full implementation, experiments, and results for my Deep Learning Final Project at Rivier University (April 2026).  
The project evaluates three deep learning approaches for image denoising:

- A **baseline CNN**
- A **Residual CNN** with skip connections
- A **Denoising Diffusion Probabilistic Model (DDPM)** based on Chapter 18 of *Understanding Deep Learning* by Simon Prince

The goal is to compare classical filters, CNN-based models, and diffusion-based denoising using PSNR and SSIM.

---

## 📌 Project Structure

image-denoising-residual-cnn/
│
├── models/
│   ├── baseline_cnn.py
│   ├── residual_cnn.py
│   ├── diffusion_ddpm.py
│
├── training/
│   ├── train_cnn.py
│   ├── train_residual.py
│   ├── train_ddpm.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│
├── results/
│   ├── quantitative_results.csv
│   ├── qualitative_examples/
│
├── report/
│   ├── Final_Report.pdf
│
├── requirements.txt
└── README.md

---

## 📚 Dataset

### **1. SIDD (Smartphone Image Denoising Dataset)**
Real noisy images with clean references.

### **2. BSD500 with Synthetic Noise**
Gaussian noise added at σ = 15, 25, 50 for controlled experiments.

### **Preprocessing**
- Resize to 128×128 or 256×256  
- Train/Val/Test = 70/15/15  
- Augmentations: random crop, flip, brightness jitter  

---

## 🧠 Models

### **1. Baseline CNN**
- 3–5 convolutional layers  
- ReLU activations  
- No skip connections  

### **2. Residual CNN**
- Residual blocks: Conv → ReLU → Conv  
- Identity skip connections  
- Optional batch normalization  

### **3. Diffusion Model (DDPM)**
Implemented to satisfy the professor’s requirement (Chapter 18).

Key components:
- Forward process: add Gaussian noise over T steps  
- Reverse process: UNet predicts noise  
- Loss: MSE noise prediction  
- Sampling: iterative denoising  

---

## ⚙️ Training Setup

- Optimizer: Adam  
- Learning rate: 1e‑3 with cosine decay  
- Batch size: 16–32  
- Epochs: 50–100  
- Losses: L1, optional SSIM, DDPM MSE  

---

## 📊 Results

### **Quantitative Results (PSNR / SSIM)**

| Model                   | PSNR | SSIM |
|------------------------|------|------|
| Median Filter          | 22.1 | 0.61 |
| Gaussian Blur          | 23.4 | 0.64 |
| Baseline CNN           | 27.8 | 0.79 |
| Residual CNN           | 30.5 | 0.87 |
| Diffusion Model (DDPM) | 31.2 | 0.89 |

### **Qualitative Observations**
- Sharper edges  
- Better texture preservation  
- Fewer color artifacts  
- Diffusion model gives strongest perceptual quality  

---

## 🧩 How to Run

pip install -r requirements.txt

python training/train_cnn.py

python training/train_residual.py

python training/train_ddpm.py

python evaluation/evaluate.py

---

## 🧩 Failure Analysis

Models struggle with:
- High‑frequency textures (hair, grass)  
- Extremely noisy images  
- Unseen noise types  
- Over‑smoothing in low‑contrast regions  

---

## ⚖️ Ethical Considerations

- Surveillance misuse  
- Dataset bias  
- Misinterpretation of enhanced images  
- Environmental cost of training  

---

## 📝 Final Report

The full PDF report is included in:

https://drive.google.com/file/d/1RWMHTdQ3yGtE6N1oI132sMkrMYU6IkJv/view?usp=sharing



## 🎥 Presentation Video



## 📎 References

- Zhang et al., *DnCNN: Beyond a Gaussian Denoiser*  
- Brooks et al., *Unprocessing Images for Learned Raw Denoising*  
- SIDD Dataset  
- Prince, S. (2023). *Understanding Deep Learning*. MIT Press  

---

## ✔ Status

This repository satisfies all professor requirements, including the mandatory **CNN vs Diffusion Model comparison**.
