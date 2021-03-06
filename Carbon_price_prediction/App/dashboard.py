import streamlit as st
import pandas as pd
import numpy as np

# Custom functions
@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

# Parameters
methodology = 'tf_idf' # 'tf_idf' or 'bag_of_words'
data_source = 'gdelt' # 'gdelt'
glossary_source = 'lemmatized_grouped_custom' # 'BBC' or 'IPCC' or 'custom' or 'lemmatized_custom'
version = '' # 'new' or '' for old (in case of BBC), otherwise use ''
rolling_mean_length = 30
dep_var = 'carbon_price'
rolling_corr_length = 30

# Constants
control_var_name_map = {'gas_price': 'gas prices',
'oil_price': 'oil prices',
'carbon_price': 'carbon prices',
'energy_price': 'electricity prices',
'coal_price': 'coal prices',
'stock_market_index_level': 'stock market index',}

st.title('EU ETS market dashboard')

daily_prices = pd.read_csv( "../Data/merged_dataset.csv", index_col=0,
                         parse_dates=True, dayfirst=True)
daily_prices.index.name = 'date'
daily_prices.head()

tf_idf = pd.read_csv(f'../Data/signals/{methodology}_{data_source}_{glossary_source}_{version}keywords.csv',
                     index_col=0, parse_dates=True)
tf_idf.index.name = 'date'


st.subheader('Raw data on carbon price and control variables')
tmp_daily_prices = daily_prices.copy()
tmp_daily_prices.index = tmp_daily_prices.index.to_series().apply(lambda x: x.strftime('%Y-%m-%d'))
st.dataframe(tmp_daily_prices)

st.download_button(
   "Download data",
   convert_df(tmp_daily_prices),
   "daily_prices.csv",
   "text/csv",
   key='download-csv'
)

selected_col = st.sidebar.selectbox('Select variable to compare to carbon price',
                                   list(daily_prices.columns))

st.subheader(f'Daily EU ETS carbon prices and {control_var_name_map[selected_col]}')
st.line_chart(daily_prices[[dep_var, selected_col]])

# Return correlation plot
ret_df = daily_prices.pct_change()
corr_ts = ret_df[dep_var].rolling(rolling_corr_length).corr(ret_df[selected_col])
st.subheader(f'Rolling {rolling_corr_length}-days correlation between returns of EU ETS carbon prices and selected other variables')
st.line_chart(corr_ts)


st.subheader('Raw data on TF-IDF scores')
tmp_tf_idf = tf_idf.copy()
tmp_tf_idf.index = tmp_tf_idf.index.to_series().apply(lambda x: x.strftime('%Y-%m-%d'))
st.dataframe(tmp_tf_idf)


selected_keyword_group = st.sidebar.selectbox('Select keyword group for TF-IDF score time series analysis',
                                   list(tf_idf.columns))

st.subheader(f'TF-IDF score history for keyword group: {selected_keyword_group} \
({rolling_mean_length}-days moving average)')
st.line_chart(tf_idf[[list(tf_idf.columns)[0], selected_keyword_group]].rolling(rolling_mean_length).mean())
