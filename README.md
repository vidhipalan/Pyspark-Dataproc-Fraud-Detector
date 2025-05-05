# Spark Sherlock: Catching Fraudsters Faster than They Swipe!

A PySpark-based, end-to-end financial fraud detection pipeline processing ~6.3 million synthetic mobile-money transactions. Benchmarked on standalone Databricks clusters and Google Cloud Dataproc to study:

- **Scale-Up**: runtime vs. data sample sizes (20%, 50%, 75%, 100%)  
- **Scale-Out**: runtime vs. worker counts (1–5 nodes)  

Read the full project proposal in `docs/BIA-678_Financial_Fraud_Detection_Project_Proposal.pdf`

---

## Table of Contents
1. [Features](#features)  
2. [Dataset](#dataset)  
3. [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)  
4. [Modeling & Evaluation](#modeling--evaluation)  
5. [Scale-Up & Scale-Out Strategies](#scale-up--scale-out-strategies)  
6. [Streamlit Web App](#streamlit-web-app)  
7. [Repository Structure](#repository-structure)  
8. [Setup & Usage](#setup--usage)  
9. [Results & Figures](#results--figures)  
10. [Future Work](#future-work)  
11. [License](#license)  

---

## Features
- **Data Cleaning & Transformation**: drop irrelevant fields, handle missing values  
- **Feature Engineering**: compute balance deltas, amount-to-balance ratios, encode transaction types  
- **Multiple Algorithms**: Random Forest, Isolation Forest, XGBoost with Optuna tuning  
- **Performance Benchmarking**: measure execution time vs. data size and vs. cluster size  
- **Real-time Scoring UI**: Streamlit app for transaction input and fraud prediction  

## Dataset
- **Source**: Synthetic Mobile Money Transaction dataset (~6.3 M rows)  
- **Fields**: transaction type, amount, sender/receiver balances before & after, fraud label  

## Data Preprocessing & Feature Engineering
1. Drop columns: `step`, `nameOrig`, `nameDest`, `isFlaggedFraud`  
2. Compute ΔOrig (`oldbalanceOrg - newbalanceOrig`) & ΔDest (`newbalanceDest - oldbalanceDest`)  
3. Derive amount-to-balance ratios (`amount / (oldbalanceOrg+1)`, `amount / (oldbalanceDest+1)`)  
4. Index `type` → `type_idx`, one-hot encode → `type_vec`  
5. Assemble & scale feature vector for modeling  

## Modeling & Evaluation
- **Train/Test Split**: 80/20 stratified  
- **Algorithms**: Random Forest (primary), Isolation Forest, XGBoost  
- **Metrics**: F1-score (primary), precision, recall, ROC-AUC  
- **Hyperparameter Tuning**: Optuna for tree count & depth  

## Scale-Up & Scale-Out Strategies
- **Scale-Up**: runtime measured at 20%, 50%, 75%, and 100% of the full dataset  
- **Scale-Out**: identical workload on Databricks & Dataproc clusters with 1–5 workers, showing up to ~67% runtime reduction  

## Streamlit Web App
A simple UI for inputting transaction details and displaying fraud predictions:

<img width="1105" alt="image" src="https://github.com/user-attachments/assets/41358ae2-c290-4e08-9e78-76677e0e642c" />

<img width="1236" alt="image" src="https://github.com/user-attachments/assets/dd05783e-d040-4bb0-bed7-e403acbad5f2" />

<img width="1237" alt="image" src="https://github.com/user-attachments/assets/f966a269-0c8b-4403-b27c-184df050bb5e" />


