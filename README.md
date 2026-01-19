# Alzheimer's Disease Classification via MRI Analysis

## 📌 Problem Statement
Alzheimer's Disease is a progressive neurodegenerative disorder and the leading cause of dementia worldwide. Early and accurate detection is critical for patient management and slowing progression. This project automates the classification of MRI scans into four stages: **Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented**. By leveraging Deep Learning and Cloud Computing, this project provides a scalable, serverless solution for rapid medical image screening.

---

## 🔍 Exploratory Data Analysis (EDA)
The dataset consists of high-resolution MRI images. Key findings from the EDA include:
* **Class Imbalance:** A significant disparity exists between classes (e.g., Non-Demented vs. Moderate Demented). This was addressed by monitoring F1-scores rather than just accuracy.
* **Data Normalization:** Scans were standardized to $224 \times 224$ pixels, and pixel intensity distributions were analyzed to ensure consistent normalization across the training set.



---

## 🧠 Training Process Summary
The model utilizes a transfer learning approach to maximize feature extraction from medical imaging.
* **Architecture:** ResNet-50 (Pre-trained on ImageNet) with a customized 4-class output layer.
* **Optimizer & Loss:** Adam optimizer with a Cross-Entropy Loss function.
* **Data Augmentation:** To prevent overfitting, random rotations, horizontal flips, and color jitters were applied during training.
* **Optimization:** Used a Learning Rate Scheduler to fine-tune weights as the model converged.



---

## ☁️ Cloud Deployment (AWS Lambda & Docker)
To handle the large library requirements of PyTorch, the model was deployed using a containerized serverless architecture.

### 1. Containerization
The application was packaged using **Docker**. This was necessary because the combined size of the model weights and the `torch` library exceeds the standard 250MB AWS Lambda limit.

### 2. Infrastructure Workflow
* **Amazon ECR:** Serves as the private registry for the Docker image.
* **AWS Lambda:** Configured with **2048MB RAM** to ensure fast inference times for the ResNet model.
* **API Gateway:** Provides a RESTful endpoint to receive Base64 encoded images and return JSON predictions.



---

## 🔮 Future Work
To further improve this diagnostic tool, the following enhancements are planned:
* **Frontend Integration:** Develop a React or Vue.js web interface to allow doctors to drag-and-drop MRI files for instant analysis.
* **Model Quantization:** Convert the model to ONNX or TorchScript format to reduce memory footprint and decrease latency/costs on AWS.
* **Explainable AI (XAI):** Implement Grad-CAM to generate "heatmaps" on the MRI scans, showing clinicians exactly which areas of the brain influenced the AI's prediction.
* **Multi-View Analysis:** Update the model to process 3D volumes or multiple slices (Axial, Coronal, and Sagittal) simultaneously for higher diagnostic accuracy.

---

## 🛠 How to Test
1. **Endpoint:** `POST https://your-api-gateway-url.amazonaws.com/predict`
2. **Payload:**
```json
{
  "image": "BASE64_ENCODED_STRING"
}
