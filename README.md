Obesity Prediction Using Machine Learning
📌 Project Description
Obesity is a serious health issue influenced by eating habits, physical activity, and lifestyle choices.
This project predicts an individual’s obesity level using machine learning classification models based on dietary habits and physical conditions.

Multiple ML models are trained and compared to identify the most accurate predictor.
An interactive dashboard is also developed for visualizing results and model performance.

🎯 Objectives
Analyze obesity-related lifestyle data

Perform data preprocessing and feature engineering

Predict obesity levels using ML models

Compare model performance using evaluation metrics

Visualize results through plots and a GUI dashboard

📂 Dataset
Name: ObesityDataSet_raw_and_data_sinthetic.csv

Source: UCI Machine Learning Repository

Type: Multiclass classification dataset

Target Variable: NObeyesdad

Features Include:
Age, Height, Weight

Eating habits (vegetable intake, water consumption, meals per day)

Physical activity frequency

Technology usage time

Gender and transportation mode

🧹 Data Preprocessing
Removal of duplicate records

Data type correction and standardization

Cleaning categorical values

Feature engineering using Body Mass Index (BMI)

One-hot encoding for categorical variables

Label encoding for the target variable

Min-Max normalization for numerical features

📊 Exploratory Data Analysis (EDA)
Target class distribution analysis

Univariate and bivariate analysis

Boxplots and histograms

Correlation heatmaps

BMI, age, and gender analysis across obesity categories

🤖 Machine Learning Models Used
K-Nearest Neighbors (KNN)

Naive Bayes

Decision Tree Classifier

Support Vector Machine (SVM)

📈 Evaluation Metrics
Accuracy

Precision (Macro Average)

Recall (Macro Average)

F1-Score (Macro Average)

Confusion Matrix

🏆 Best Model
Support Vector Machine (SVM) achieved the highest accuracy

Provided balanced performance across all obesity classes

🖥️ Dashboard
A Tkinter-based interactive dashboard includes:

Best model highlight

Model performance comparison table

Accuracy comparison graph

Dynamic confusion matrix visualization

🛠️ Technologies Used
Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Tkinter



Project Structure :-  

obesity-prediction-ml/
│
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
│
├── src/
│   └── obesity_prediction.py
│
├── README.md
└── requirements.txt



How to run :- 


git clone https://github.com/your-username/obesity-prediction-ml.git
cd obesity-prediction-ml
pip install -r requirements.txt
python obesity_prediction.py
