import streamlit as st
import pandas as pd
import numpy as np

st.title('Carbon market dashboard')

daily_prices = pd.read_csv( "../Data/daily_prices.csv", index_col=0,
                         parse_dates=True, dayfirst=True)
daily_prices.index.name = 'date'
daily_prices.head()


st.subheader('Raw data')
st.write(daily_prices.head())

st.subheader('Daily EU ETS carbon prices')
st.line_chart(daily_prices)
