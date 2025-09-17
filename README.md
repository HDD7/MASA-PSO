# MASA-PSO
This repository contains the implementation and experimental code for "Multiscale-adaptive and Size-adaptive PSO-based feature selection for high dimensional classification", information-theoretic measures, and optimization methods applied to various datasets. The structure and purpose of each directory/file are described below.

---

## 📂 Datasets
- **datasets/large-sample/**  
  High-dimensional **large-sample datasets** used for experiments.

- **datasets/low-sample/**  
  High-dimensional **low-sample datasets** used for experiments.

---

## 📂 Core Components

- **CE/**  
  Implementation of **Copula Entropy (CE)** for feature selection and dependency analysis.

- **MIC/**  
  Implementation of the **Maximal Information Coefficient (MIC)** for measuring nonlinear relationships between variables.

- **SU/**  
  Implementation of **Symmetrical Uncertainty (SU)**, an information-theoretic metric for feature relevance.

- **PSO/**  
  Particle Swarm Optimization (PSO) experiments on **high-dimensional low-sample datasets**.

- **main_large/**  
  Experiments and evaluations on **high-dimensional large-sample datasets**.

- **tsne/**  
  **t-Distributed Stochastic Neighbor Embedding (t-SNE)** visualization of feature distributions and experimental results.

- **Wilcoxon_test/**  
  Statistical significance testing using the **Wilcoxon signed-rank test**.

- **word_op/**  
  Preprocessing operations for **NLP datasets**.

---

## 📂 Research-Specific Studies

- **Forgotten_in_9T_Tumors/**  
- **Forgotten_in_11Tumors/**  
- **Forgotten_in_Prostate/**  
  Research on the phenomenon of **particles being forgotten** in three different datasets.

---

## 🔧 Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
