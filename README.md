# Heart-Disease-Prediction-App
This project builds a machine learning application to predict the presence of heart disease using clinical and demographic data. It uses multiple classification algorithms, evaluates their performance, and provides an interactive prediction interface.
## Dataset
The dataset used is the UCI Heart Disease dataset.  
Key features include:
- Age, sex, chest pain type
- Resting blood pressure, cholesterol level
- Fasting blood sugar, resting ECG results
- Maximum heart rate achieved, exercise-induced angina
- Oldpeak (ST depression), slope of peak exercise
- Number of major vessels colored by fluoroscopy, thalassemia type  
Target variable:
- **1** → Heart disease present  
- **0** → No heart disease  

## Data Preprocessing
- Removed irrelevant ID column.  
- Renamed target variable for clarity.  
- Handled missing values and converted data types.  
- Normalized features where necessary.  

## Models Trained
The following models were trained and evaluated:
1. Logistic Regression  
2. Decision Tree Classifier  
3. Random Forest Classifier  

## Evaluation Metrics
Each model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

## Model Insights and Results
- **Logistic Regression**: Balanced performance, interpretable coefficients.  
- **Decision Tree**: Captures non-linear relationships, but risk of overfitting.  
- **Random Forest**: Best trade-off between accuracy and robustness, strong ROC-AUC.  

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.83    | ~0.82     | ~0.81  | ~0.81    | ~0.87   |
| Decision Tree       | ~0.79    | ~0.77     | ~0.78  | ~0.77    | ~0.82   |
| Random Forest       | ~0.86    | ~0.85     | ~0.84  | ~0.84    | ~0.90   |

