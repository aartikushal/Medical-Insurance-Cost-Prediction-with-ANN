🏥*Medical Insurance Charges Prediction Using ANN*
📖 Project Overview

This project predicts individual medical insurance charges based on personal attributes using an Artificial Neural Network (ANN).

Predicting medical costs helps insurance companies, healthcare providers, and researchers analyze cost patterns.

Dataset: Medical Insurance Dataset (Kaggle)

📊 Dataset Features
Feature	Type	Description
age	Numerical	Age of the insured
sex	Categorical	Gender of the insured (male/female)
bmi	Numerical	Body Mass Index (kg/m²)
children	Numerical	Number of children/dependents
smoker	Categorical	Smoking status (yes/no)
region	Categorical	Residential region (northeast, southeast…)
charges	Numerical	Insurance charges (target variable)
🔹 Data Understanding & Preprocessing

Explored the dataset using .head(), .info(), .describe()

Checked for missing values (data.isnull().sum()) → none found

Encoded categorical variables with pd.get_dummies(drop_first=True)

Scaled numerical features using StandardScaler

Split data: 80% training, 20% testing

🧠 ANN Model Architecture

Input Layer: Number of features after encoding

Hidden Layers:

1st layer: 128 neurons + ReLU + BatchNormalization + Dropout(0.3)

2nd layer: 64 neurons + ReLU + BatchNormalization + Dropout(0.2)

3rd layer: 32 neurons + ReLU

Output Layer: 1 neuron, Linear activation (regression)

Optimizer: Adam (learning rate = 0.001)

Loss Function: Mean Squared Error (MSE)

EarlyStopping: patience = 20, restore_best_weights = True

🚀 Model Training

Epochs: 300

Batch size: 16

Validation split: 20%

Training & validation loss plots:

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

📈 Model Evaluation
Metric	Value
Mean Squared Error (MSE)	21,286,053.43
R-squared (R²)	0.86

Actual vs Predicted Charges Plot:

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Insurance Charges')
plt.show()


✅ Interpretation:

R² = 0.86 → ANN explains 86% of variance in insurance charges.

Model generalizes well on unseen data.

⚡ Next Steps / Enhancements

Feature Engineering:

Add interaction terms like bmi * smoker

Polynomial features for non-linear patterns

Hyperparameter Tuning:

Explore different layer sizes, dropout rates, and learning rates

Alternative Models:

Try XGBoost, LightGBM, or Random Forest for tabular data

Deployment:

Convert the ANN into a Streamlit app or REST API for real-time predictions

📂 Project Structure
insurance-ann/
│
├── data/
│   └── insurance.csv          # Dataset
│
├── notebooks/
│   └── EDA.ipynb              # Exploratory Data Analysis
│
├── src/
│   └── insurance_ann.py       # ANN model code
│
├── outputs/
│   ├── model_weights.h5       # Trained model
│   └── predictions.csv        # Predicted charges
│
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies

📦 Dependencies
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
keras


Install with:

pip install -r requirements.txt

👤 Author

Aarti Potdar
Senior Consultant | Data Science Enthusiast
