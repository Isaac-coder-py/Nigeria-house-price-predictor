import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title='Nigeria House Price Predictor')

# typical error of the model, from the notebook (0.25 means plus or minus 25%)
MARGIN = 0.25

# scales 2023 listing prices up to 2026 (1.8 means prices are 80% higher now)
PRICE_ADJUSTMENT = 1.8


@st.cache_resource
def load_files():
    # load the saved model, its column names and the dropdown options
    model = joblib.load('house_model.pkl')
    columns = joblib.load('model_columns.pkl')
    options = joblib.load('options.pkl')
    return model, columns, options


model, columns, options = load_files()

st.title('Nigeria House Price Predictor')
st.write('Enter the house details to get an estimated price.')

# location inputs (towns depend on the chosen state)
state = st.selectbox('State', options['state'], index=options['state'].index('Lagos'))
town = st.selectbox('Town', options['towns_by_state'][state])
title = st.selectbox('House type', options['title'])

# size inputs
col1, col2 = st.columns(2)
bedrooms = col1.number_input('Bedrooms', min_value=1, max_value=9, value=4)
bathrooms = col2.number_input('Bathrooms', min_value=1, max_value=9, value=4)
toilets = col1.number_input('Toilets', min_value=1, max_value=9, value=5)
parking = col2.number_input('Parking spaces', min_value=1, max_value=9, value=4)

if st.button('Predict price'):
    # start with a row of zeros that has the same columns the model was trained on
    row = pd.DataFrame(0, index=[0], columns=columns)
    row['bedrooms'] = bedrooms
    row['bathrooms'] = bathrooms
    row['toilets'] = toilets
    row['parking_space'] = parking

    # switch on the matching state, town and house type columns
    for col in [f'state_{state}', f'town_{town}', f'title_{title}']:
        if col in row.columns:
            row[col] = 1

    # the model predicts the log of the price, so convert it back to naira
    # then scale it from 2023 prices up to 2026 prices
    price = np.exp(model.predict(row)[0]) * PRICE_ADJUSTMENT

    # build the range around the estimate
    low = price * (1 - MARGIN)
    high = price * (1 + MARGIN)

    st.success(f'Estimated price: ₦{low/1e6:,.1f}M to ₦{high/1e6:,.1f}M')
    