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
    "MSSubClass",
    "MSZoning",
    "LotFrontage",
    "LotArea",
    "LotShape",
    "LotConfig",
    "Neighborhood",
    "HouseStyle",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "MasVnrType",
    "ExterQual",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtFinSF1",
    "TotalBsmtSF",
    "HeatingQC",
    "CentralAir",
    "1stFlrSF",
    "2ndFlrSF",
    "GrLivArea",
    "BsmtFullBath",
    "FullBath",
    "BedroomAbvGr",
    "KitchenQual",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageType",
    "GarageYrBlt",
    "GarageFinish",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
]


@st.cache_data
def load_data(nrows, file):
    data = pd.read_csv(file, nrows=nrows)     # 'data/housing_dataset.csv'
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis='columns', inplace=True)
    return data


loaded_rf = joblib.load("random_forest.joblib")
uploaded_file = st.file_uploader("Choose a csv file")
if uploaded_file is not None:
    data = load_data(1460, uploaded_file)

    cat_col = [c for c in data[col].columns if data[col][c].dtype == 'object']
    num_col = [c for c in data[col].columns if data[col][c].dtype == 'int64' or data[col][c].dtype == 'float64']

    # Pipeline

    # according to data_description.txt, missing values in columns below means something like 'No Pool'
    category_cols1 = ['BsmtQual', 'BsmtCond', 'GarageType', 'GarageFinish']
    category_transformer1 = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # other category_cols
    category_cols2=list(set(cat_col)-set(category_cols1))
    category_transformer2 = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # for numerical_cols, all in category_cols1's situation
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, num_col),
            ('category1', category_transformer1, category_cols1),
            ('category2', category_transformer2, category_cols2)
        ])

    preprocessor.fit(data[col])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', loaded_rf)
                               ], memory='./')

    predictions = pipeline.predict(data[col])
    df_predictions = pd.concat([data['Id'], data['SalePrice'], pd.Series(predictions, name='PricePrediction')], axis=1)

    st.subheader('Predictions')
    st.write(df_predictions.set_index('Id'))

    if st.button('Show shap values'):
        # SHAP VALUE
        feature_names = pipeline['preprocessor'].transformers_[0][2] + \
                        list(pipeline['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(category_cols1)) + \
                        list(pipeline['preprocessor'].transformers_[2][1]['onehot'].get_feature_names_out(category_cols2))

        st.subheader('Shap values')

        explainer = shap.TreeExplainer(loaded_rf)
        choosen_instance = preprocessor.transform(data)
        shap_values = explainer.shap_values(choosen_instance)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values, choosen_instance, feature_names=feature_names)
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
        plt.clf()
