#
# Cardiovascular Disease Prediction by Machine Learning: Predictive Models and Optimization

<span style="color: #008800">Navid Azimi, Masoud Nateghi</span>

[Link to Dataset](https://www.kaggle.com/datasets/colewelkins/cardiovascular-disease/data)


The primary objective of this project is to predict the presence or absence of cardiovascular disease based on various patient metrics.

### Variables

- ***ID:*** Unique identifier for each patient.
- ***age:*** Age of the patient in days.
- ***age_years:*** Age of the patient in years (derived from age).
- ***gender:*** Gender of the patient. Categorical variable (1: Female, 2: Male).
- ***height:*** Height of the patient in centimeters.
- ***weight:*** Weight of the patient in kilograms.
- ***ap_hi:*** Systolic blood pressure.
- ***ap_lo:*** Diastolic blood pressure.
- ***cholesterol:*** Cholesterol levels. Categorical variable (1: Normal, 2: Above Normal, 3: Well Above Normal).
- ***gluc:*** Glucose levels. Categorical variable (1: Normal, 2: Above Normal, 3: Well Above Normal).
- ***smoke:*** Smoking status. Binary variable (0: Non-smoker, 1: Smoker).
- ***alco:*** Alcohol intake. Binary variable (0: Does not consume alcohol, 1: Consumes alcohol).
- ***active:*** Physical activity. Binary variable (0: Not physically active, 1: Physically active).
- ***cardio:*** Presence or absence of cardiovascular disease. Target variable. Binary (0: Absence, 1: Presence).
- ***bmi:*** Body Mass Index, derived from weight and height. Calculated as BMI = weight (kg) / height (m)^2.
- ***bp_category:*** Blood pressure category based on ap_hi and ap_lo. Categories include "Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2".
- ***bp_category_encoded:*** Encoded form of bp_category for machine learning purposes.
