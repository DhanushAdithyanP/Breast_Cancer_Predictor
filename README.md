# Breast Cancer Predictor

A machine learning project to predict breast cancer using the Wisconsin Breast Cancer dataset.

Check out the [Breast Cancer Predictor Web App](https://bcp-breastcancerprediction.streamlit.app/) for a user-friendly interface.

## Table of Contents

1. [Introduction](##Introduction)
2. [Installation](#Installation)
3. [Usage](#usage)
4. [Dataset](#dataset)
5. [Model](#model)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

The Breast Cancer Predictor project aims to assist in the early detection of breast cancer by using machine learning algorithms to analyze various medical attributes. The project leverages the Wisconsin Breast Cancer dataset to train a predictive model that can help healthcare professionals in diagnosing breast cancer more accurately. This project also has a front end made using Streamlit that provides a user-friendly interface for the users.

## Installation

### Prerequisites

- Python 3.8 or higher
- Required libraries: NumPy, pandas, scikit-learn, matplotlib, seaborn, Streamlit

### Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/breast-cancer-predictor.git

# Navigate to the project directory
cd breast-cancer-predictor

# Install the required libraries
pip install -r requirements.txt
```
## Usage

### Running the Streamlit App

Before running the Streamlit app for predicting breast cancer, navigate to the project directory:

```bash
# Navigate to the project directory
cd breast-cancer-predictor

# Run the Streamlit app
streamlit run app/main.py
```
## Dataset

### Description

The Wisconsin Breast Cancer dataset (also known as the Wisconsin Diagnostic Breast Cancer (WDBC) dataset) is a well-known dataset for breast cancer classification tasks in machine learning. It consists of features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features are computed to describe characteristics of the cell nuclei present in the image. The dataset includes measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension, and diagnosis (Malignant or Benign).

### Access

You can download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

## Model

### Tested Models

The project evaluated several machine learning models for predicting breast cancer:

- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (K-NN)
- Naive Bayes
- Gradient Boosting Machine (GBM)

### Training Environment

All models were trained on a Mac M1 Silicon chip with 16GB RAM. This environment was chosen for its efficiency in handling machine learning tasks and dataset complexities.

## Results

### Evaluation Metrics

The table below shows the evaluation metrics (F1 Score, Precision, Accuracy, Recall) for various machine learning models tested in predicting breast cancer using the Wisconsin Breast Cancer dataset:

| Model              | F1 Score | Precision | Accuracy | Recall    |
|--------------------|----------|-----------|----------|-----------|
| Logistic Regression| 0.929825 | 0.913793  | 0.953216 | 0.946429  |
| Decision Trees     | 0.851852 | 0.884615  | 0.906433 | 0.821429  |
| Random Forest      | 0.897196 | 0.941176  | 0.935673 | 0.857143  |
| SVM                | 0.810811 | 0.818182  | 0.877193 | 0.803571  |
| K-NN               | 0.814159 | 0.807018  | 0.877193 | 0.821429  |
| Naive Bayes        | 0.915888 | 0.960784  | 0.947368 | 0.875000  |
| GBM                | 0.900901 | 0.909091  | 0.935673 | 0.892857  |

### Importance of Recall

In medical diagnosis, such as breast cancer detection, minimizing false negatives (missed cases of cancer) is crucial. Therefore, the recall score (true positive rate) is particularly important as it measures the proportion of actual positives that are correctly identified by the model. A higher recall score indicates fewer missed cases of cancer, which is essential for patient outcomes and treatment decisions.

### Visualization 

![3](https://github.com/DhanushAdithyanP/Breast_Cancer_Predictor/assets/91380094/c5a50cad-f331-4b3a-b1af-e0e3fd6c359a)

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Submit a pull request

## Contributors: 
1. Kavya M M - https://github.com/Kavya291
2. P Dhanush Adithyan - https://github.com/DhanushAdithyanP

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
