import joblib
import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor


col = [
    "LotArea",
    "OverallQual",
    "TotalBsmtSF",
    "GrLivArea",
    "TotRmsAbvGrd",
    "GarageArea",
]


@st.cache_data
def load_data(nrows, file):
    data = pd.read_csv(file, nrows=nrows)     # 'data/housing_dataset.csv'
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis='columns', inplace=True)
    return data


loaded_rf = joblib.load("random_forest_light.joblib")
LotArea = st.number_input('Insert the lot area')
OverallQual = st.number_input('Insert the overall quality (max 10)')
TotalBsmtSF = st.number_input('Insert the basement area')
GrLivArea = st.number_input('Insert the ground living area')
TotRmsAbvGrd = st.number_input('Insert the number of room above ground')
GarageArea = st.number_input('Insert the garage area')
df_data = np.array([LotArea, OverallQual, TotalBsmtSF, GrLivArea, TotRmsAbvGrd, GarageArea]).reshape(1, -1)
df_data = pd.DataFrame(df_data, columns=col)

if st.button('Predict price'):
    # Pipeline

    # for numerical_cols, all in category_cols1's situation
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, col),
        ])

    preprocessor.fit(df_data)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', loaded_rf)
                               ], memory='./')

    prediction = pipeline.predict(df_data)

    st.subheader('Predictions')
    st.write(prediction)

    if st.button('Show shap values'):
        # SHAP VALUE
        feature_names = pipeline['preprocessor'].transformers_[0][2]

        st.subheader('Shap values')

        explainer = shap.TreeExplainer(loaded_rf)
        choosen_instance = preprocessor.transform(df_data)
        shap_values = explainer.shap_values(choosen_instance)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values, choosen_instance, feature_names=feature_names)
        st.pyplot(bbox_inches='tight', pad_inches=0)
        plt.clf()
