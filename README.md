# Stone-Kidney-Detection

## Overview
This repository provides Quantum Machine Learning (QML) approaches for detecting kidney stones using ultrasound images. The dataset used for training and evaluation is available on Kaggle: [Kidney Ultrasound Images (Stone and No Stone)](https://www.kaggle.com/datasets/gurjeetkaurmangat/kidney-ultrasound-images-stone-and-no-stone).

## Files and Results

### 1. `classical_stone_detection.ipynb`
- **Classification Report:**
  ```
  {'0': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 199.0},
   '1': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 201.0},
   'accuracy': 1.0,
   'macro avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 400.0},
   'weighted avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 400.0}}
  ```
- **Confusion Matrix:**
  ```
  [[199   0]
   [  0 201]]
  ```
- **ROC AUC Score:** 1.0
- **Conclusion:** The classical model achieved perfect classification with 100% accuracy, precision, recall, and F1-scores, indicating excellent performance on the dataset.

### 2. `FidelityQuantumKernel_qiskit_stone_detection.ipynb`
- **Callable Kernel Classification Report:**
  ```
  Precision: 0.64, Recall: 0.64, F1-Score: 0.61, Accuracy: 64%
  ```
- **Precomputed Kernel Classification Report:**
  ```
  Precision: 0.66, Recall: 0.66, F1-Score: 0.66, Accuracy: 66%
  ```
- **QSVC Classification Report:**
  ```
  Precision: 0.63, Recall: 0.63, F1-Score: 0.60, Accuracy: 63%
  ```
- **Conclusion:** Fidelity quantum kernel approaches show moderate performance. Precomputed kernels outperform callable kernels slightly, indicating preprocessing benefits.

### 3. `ingenii_stone_detection.ipynb`
- **Classification Report:**
  ```
  Precision: 97.31%, Recall: 97.14%, F1-Score: 97.15%, Accuracy: 97.14%
  ```
- **Confusion Matrix:**
  ```
  [[32  0]
   [ 2 36]]
  ```
- **ROC AUC Score:** 0.97
- **Conclusion:** Ingenii achieves excellent performance with near-perfect accuracy and high ROC AUC, demonstrating reliable classification capabilities.

### 4. `Pegasos_qsvc_qiskit_stone_detection.ipynb`
- **Classification Report:**
  ```
  Precision: 64.5%, Recall: 64.5%, F1-Score: 64.5%, Accuracy: 64.5%
  ```
- **Confusion Matrix:**
  ```
  [[406 302]
   [301 692]]
  ```
- **ROC AUC Score:** 0.33
- **Conclusion:** Performance is suboptimal, with moderate accuracy and poor ROC AUC, indicating room for improvement in kernel optimization.

### 5. `Pegasos_updated_qsvc_qiskit_stone_detection.ipynb`
- **Classification Report:**
  ```
  Precision: 77.5%, Recall: 75.8%, F1-Score: 76.0%, Accuracy: 75.8%
  ```
- **Confusion Matrix:**
  ```
  [[582 126]
   [285 708]]
  ```
- **ROC AUC Score:** 0.18
- **Conclusion:** Updated Pegasos model shows improved performance with better accuracy and precision compared to its predecessor.

### 6. `pennylane_stone_detection.ipynb`
- **Test Accuracy:** 100%
- **Classification Report:**
  ```
  Precision: 100%, Recall: 100%, F1-Score: 100%, Accuracy: 100%
  ```
- **Conclusion:** Pennylane-based QML achieves perfect results, indicating robust detection capability.

### 7. `piqture_stone_detection.ipynb`
- **Classification Report:**
  ```
  Precision: 52.1%, Recall: 55.0%, F1-Score: 46.9%, Accuracy: 55.0%
  ```
- **Confusion Matrix:**
  ```
  [[ 26 187]
   [ 29 238]]
  ```
- **ROC AUC Score:** 0.50
- **Conclusion:** PIQURE-based approach struggles, yielding low precision and recall, and requires significant improvements.

### 8. `qiskit_hybrid_stone_detection.ipynb`
- **Classification Report:**
  ```
  Precision: 100%, Recall: 100%, F1-Score: 100%, Accuracy: 100%
  ```
- **Confusion Matrix:**
  ```
  [[35  0]
   [ 0 55]]
  ```
- **ROC AUC Score:** 1.0
- **Conclusion:** The hybrid Qiskit approach demonstrates flawless performance on the dataset.

### 9. `qiskit_ml_stone_detection.ipynb`
- **Classification Report:**
  ```
  Precision: 47.6%, Recall: 53.1%, F1-Score: 44.3%, Accuracy: 53.1%
  ```
- **Confusion Matrix:**
  ```
  [[ 20 193]
   [ 32 235]]
  ```
- **ROC AUC Score:** 0.49
- **Conclusion:** The model shows suboptimal performance, failing to achieve satisfactory precision and recall.

### 10. `qml_torch_stone_detection.ipynb`
- **Classification Report:**
  ```
  Precision: 100%, Recall: 100%, F1-Score: 100%, Accuracy: 100%
  ```
- **Confusion Matrix:**
  ```
  [[56  0]
   [ 0 74]]
  ```
- **ROC AUC Score:** 1.0
- **Conclusion:** PyTorch-based QML achieves perfect performance, indicating high reliability for this task.

### 11. `QuantumKernelTrainer_qiskit_stone_detection.ipynb`
- **Classification Report:**
  ```
  Precision: 64.5%, Recall: 64.3%, F1-Score: 63.2%, Accuracy: 64.3%
  ```
- **Confusion Matrix:**
  ```
  [[15 17]
   [ 8 30]]
  ```
- **ROC AUC Score:** 0.65
- **Conclusion:** This quantum kernel trainer shows moderate performance, with accuracy and ROC AUC comparable to other mid-performing models.

## Summary
- **Top Performing Models:**
  - `classical_stone_detection.ipynb`, `pennylane_stone_detection.ipynb`, `qml_torch_stone_detection.ipynb`, and `qiskit_hybrid_stone_detection.ipynb` all achieve 100% accuracy and demonstrate robust performance.

- **Moderate Performance:**
  - `FidelityQuantumKernel_qiskit_stone_detection.ipynb`, `Pegasos_updated_qsvc_qiskit_stone_detection.ipynb`, and `QuantumKernelTrainer_qiskit_stone_detection.ipynb` achieve acceptable but not outstanding results, indicating room for improvement.

- **Low Performance:**
  - `Pegasos_qsvc_qiskit_stone_detection.ipynb`, `piqture_stone_detection.ipynb`, and `qiskit_ml_stone_detection.ipynb` show subpar accuracy and require significant optimization.

