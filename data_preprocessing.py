import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Data Preprocessing", page_icon="🔧")

st.header("Data Preprocessing Module")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Load and display raw data
    raw_data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(raw_data)

    # Preprocessing options
    st.subheader("Preprocessing Options")

    missing_values_strategy = st.selectbox(
        "Choose a strategy to handle missing values",
        ["Mean", "Median", "Most Frequent"],
    )

    encode_categorical = st.checkbox("Encode categorical variables")

    # Preprocessing pipeline
    numeric_features = raw_data.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = raw_data.select_dtypes(include=["object", "category"]).columns

    numeric_transformer = SimpleImputer(strategy=missing_values_strategy.lower())

    if encode_categorical:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
    else:
        categorical_transformer = SimpleImputer(strategy="most_frequent")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Apply preprocessing and display results
    st.subheader("Preprocessed Data")
    preprocessed_data = pd.DataFrame(preprocessor.fit_transform(raw_data))
    st.write(preprocessed_data)
