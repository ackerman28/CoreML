# CoreML: High-Performance Machine Learning from Scratch

## Overview
**CoreML** is a lightweight, vectorized machine learning library built entirely with NumPy. This project implements fundamental supervised and unsupervised algorithms from first principles, prioritizing mathematical transparency and computational efficiency.

Developed with a focus on "from-scratch" implementation, CoreML is an ideal framework for academic research, algorithmic prototyping, and understanding the linear algebra and optimization logic behind modern machine learning.

---

## Project Structure
```text
.
├── examples/
│   ├── Supervised Learning/
│   │   ├── Decision Trees/
│   │   ├── Ensembles/        
│   │   ├── KNN/
│   │   ├── Linear Regression/
│   │   ├── Logistic Regression/
│   │   ├── Neural Networks/  
│   │   └── Perceptron/
│   └── Unsupervised Learning/
│       ├── DBSCAN/
│       ├── KMeans/
│       └── PCA/
├── src/
│   └── mlpackage/
│       ├── __init__.py
│       ├── metrics.py          
│       ├── preprocess.py        
│       ├── supervised_learning/
│       └── unsupervised_learning/
├── tests/                   
├── requirements.txt
└── README.md