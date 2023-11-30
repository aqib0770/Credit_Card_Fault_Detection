# Credit Card Fraud Detection

This project is a machine learning model for detecting credit card fraud. It uses a RandomForestClassifier, which was found to be the best model for this problem statement.

## Dataset

The dataset used in this project is imbalanced, with the majority of credit card transactions being normal and a very small percentage being fraudulent. To handle this imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) is used to oversample the minority class (fraudulent transactions).

## Bulk Prediction

The project is designed to make predictions in bulk. It accepts a file of inputs and outputs a file of predictions.

## Running the Project

To run the project, follow these steps:

1. Clone the repository.
2. Install the required dependencies.
3. Run the Flask application.

```bash
git clone <repository_url>
pip install -r requirements.txt
python app.py