# Explainable Data-driven Autism Detection

## Overview
This project focuses on leveraging data-driven techniques to detect Autism Spectrum Disorder (ASD). The goal is to identify key features contributing to ASD detection while ensuring results are interpretable and explainable.

## Dataset
The dataset contains:
- Behavioral questionnaire responses (10 questions based on AQ-10 for adults)
- Demographic and family history information:
  - **Age**
  - **Gender**
  - **Ethnicity**
  - **Born with jaundice** (Yes/No)
  - **Family member with ASD** (Yes/No)
  - **Country of residence**
  - **Relation to the test taker** (Parent, self, caregiver, etc.)
  - **Use of screening app before** (Yes/No)
  - **ASD classification** (Binary label: `1` for ASD-positive, `0` for ASD-negative)

Thabtah, F. (2017). Autism Screening Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F019.

## Prerequisites
- Install required dependencies
```bash
pip install -r requirements.txt
````
- Preprocess the dataset
```bash
python preprocess.py
```

The script will clean and preprocess the data, saving it as `data/cleaned_data.csv`.

## Authors
 - Davit Shadunts
 - Dhruv Srikanth 
 - Samira Hajizadeh
