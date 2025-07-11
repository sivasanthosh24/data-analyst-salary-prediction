import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# ------------------------
# LOAD & TRAIN MODEL AGAIN (if no saved model)
# ------------------------
st.title("✅ Streamlit is working")
st.write("If you're seeing this, your app is running fine. Let's move forward!")


@st.cache_resource
def train_model():
    df = pd.read_csv("cleaned_data.csv")

    # Feature engineering
    df['has_python'] = df['job_description'].str.contains('python', case=False, na=False).astype(int)
    df['has_excel'] = df['job_description'].str.contains('excel', case=False, na=False).astype(int)
    df['tech_skill_score'] = df['has_python'] + df['has_excel']

    # Top categories only
    top_industries = df['industry'].value_counts().nlargest(10).index
    df['industry'] = df['industry'].apply(lambda x: x if x in top_industries else 'Other')

    # Encode
    features_to_encode = ['size', 'type_of_ownership', 'industry', 'sector']
    df_encoded = pd.get_dummies(df[features_to_encode], drop_first=True)

    df_model = pd.concat([df[['Rating', 'tech_skill_score', 'average_salary']], df_encoded], axis=1)
    df_model.dropna(inplace=True)

    X = df_model.drop('average_salary', axis=1)
    y = df_model['average_salary']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, X.columns

model, model_columns = train_model()

# ------------------------
# STREAMLIT UI
# ------------------------

st.title("💼 Data Analyst Salary Predictor")
st.write("Predict salary based on rating, skill, size, and sector")

# User inputs
rating = st.slider("Company Rating (1.0 to 5.0)", 1.0, 5.0, 3.5)
tech_skills = st.slider("Tech Skill Score (0 = none, 2 = Python + Excel)", 0, 2, 1)

# Minimal dummy encoding input
input_dict = {
    'Rating': rating,
    'tech_skill_score': tech_skills
}

# Dynamic dummy inputs
for col in model_columns:
    if col not in input_dict:
        input_dict[col] = 0

# Display dropdowns for top-encoded columns
if 'size_51 to 200 employees' in model_columns:
    size = st.selectbox("Company Size", ['51 to 200 employees', '201 to 500 employees', '1001 to 5000 employees', '10000+ employees'])
    input_dict[f"size_{size}"] = 1

if 'industry_Information Technology' in model_columns:
    industry = st.selectbox("Industry", ['Information Technology', 'Business Services', 'Finance', 'Health Care', 'Other'])
    input_dict[f"industry_{industry}"] = 1

if 'sector_Information Technology' in model_columns:
    sector = st.selectbox("Sector", ['Information Technology', 'Finance', 'Health Care', 'Education', 'Other'])
    input_dict[f"sector_{sector}"] = 1

if 'type_of_ownership_Private' in model_columns:
    ownership = st.selectbox("Ownership Type", ['Private', 'Non-profit', 'Public'])
    input_dict[f"type_of_ownership_{ownership}"] = 1

# Predict button
if st.button("Predict Salary"):
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Average Salary: ${prediction:,.2f}")
