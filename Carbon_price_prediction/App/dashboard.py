import pandas as pd
import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt
import altair as alt
import streamlit as st

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
prediction_varname = 'prediction'
rolling_corr_length = 30

control_data_display = False
detailed_tf_idf_keyword_group_data_display = False
research_paper_period = True

# Constants
control_var_name_map = {'gas_price': 'gas prices',
'oil_price': 'oil prices',
'carbon_price': 'carbon prices',
'energy_price': 'electricity prices',
'coal_price': 'coal prices',
'stock_market_index_level': 'stock market index',}

keyword_var_name_map = {'0': 'Aggregated',
'emissions': 'Emissions',
'fossil_fuel': 'Fossil fuels',
'gas': 'Gas',
'policy': 'Policy',
'renewables': 'Renewables',
}

#### DASHBOARD ####
st.title('EU ETS news tracker')

st.markdown(
"""
# Authors
- Aron Hartvig
- Peter Palos
- Aron Pap
"""
)
st.markdown(
"""
# Paper abstract
The emissions trading system is a key driver of emissions reduction in the EU. Carbon prices have been rapidly increasing since 2020 and accurate forecasting of EU Emissions Trading System (ETS) prices has become essential. In this paper, we propose a novel method to generate alternative predictors for ETS prices using GDELT online news database. We compose the EU climate change news index (ECCNI) by calculating term frequency–inverse document frequency (TF-IDF) feature for climate change related keywords. As climate policies are widely discussed in the news, the index is capable of tracking the ongoing debate about climate change in the EU. Finally, we show that incorporating the ECCNI in a simple predictive model robustly improves forecasts of ETS prices compared to a control model where the traditional predictors of carbon prices are included.
"""
)

# Data import
daily_prices = pd.read_csv( "../Data/new_merged_dataset.csv", index_col=0,
                         parse_dates=True, dayfirst=True)
daily_prices.index.name = 'date'

start_date = st.sidebar.slider(
    "Analysis start date",
    min_value=min(daily_prices.index).to_pydatetime(),
    value=datetime(2018, 1, 1),
    max_value=max(daily_prices.index).to_pydatetime(),
    format="YYYY-MM-DD")

end_date = st.sidebar.slider(
    "Analysis end date",
    min_value=min(daily_prices.index).to_pydatetime(),
    value=datetime(2021, 11, 30),
    max_value=max(daily_prices.index).to_pydatetime(),
    format="YYYY-MM-DD")

# Switch between research paper analysis vs online/up-to-date data
if research_paper_period:
    tf_idf_file_path = '../Data/signals'
else:
    tf_idf_file_path = './data'

tf_idf = pd.read_csv(f'{tf_idf_file_path}/{methodology}_{data_source}_{glossary_source}_{version}keywords.csv',
                     index_col=0, parse_dates=True)
tf_idf.index.name = 'date'

tf_idf_aggr = pd.read_csv(f'{tf_idf_file_path}/{methodology}_{data_source}_lemmatized_aggregated_{version}keywords.csv',
                     index_col=0, parse_dates=True)
tf_idf_aggr.index.name = 'date'

# Join aggregated TF-IDF score column
tf_idf = tf_idf.join(tf_idf_aggr)
tf_idf = tf_idf.rename(columns = keyword_var_name_map)

# Filter control and TF-IDF dataframe for the relevant date range
daily_prices = daily_prices[(daily_prices.index >= start_date) &
                            (daily_prices.index <= end_date)]
tf_idf = tf_idf.loc[(tf_idf.index >= start_date) &
                    (tf_idf.index <= end_date)]


if control_data_display:
    st.subheader('Raw data on carbon price and control variables')
    tmp_daily_prices = daily_prices.copy()
    tmp_daily_prices.index = tmp_daily_prices.index.to_series().apply(lambda x: x.strftime('%Y-%m-%d'))
    st.dataframe(tmp_daily_prices)

    selected_col = st.sidebar.selectbox('Select variable to compare to carbon price',
                                       list(daily_prices.columns))

    st.subheader(f'Daily EU ETS carbon prices and {control_var_name_map[selected_col]}')
    st.line_chart(daily_prices[[dep_var, selected_col]])

    # Return correlation plot
    ret_df = daily_prices.pct_change()
    corr_ts = ret_df[dep_var].rolling(rolling_corr_length).corr(ret_df[selected_col])
    st.subheader(f'Rolling {rolling_corr_length}-days correlation between returns of EU ETS carbon prices and selected other variables')
    st.line_chart(corr_ts)

selected_keyword_group = st.sidebar.selectbox('Select keyword group for TF-IDF score time series analysis',
                                   list(tf_idf.columns))

if detailed_tf_idf_keyword_group_data_display:
    st.subheader(f'TF-IDF score history for keyword group: {selected_keyword_group} \
    ({rolling_mean_length}-days moving average)')
    st.line_chart(tf_idf[[list(tf_idf.columns)[0], selected_keyword_group]].rolling(rolling_mean_length).mean())

st.subheader(f'Carbon price time series and TF-IDF score history for keyword group: {selected_keyword_group} \
({rolling_mean_length}-days moving average)')

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(daily_prices[[dep_var]], label='EU ETS carbon price')
ax1.set_ylabel('EU ETS carbon price')
ax1.legend(loc=0)
ax1.tick_params(axis='x', rotation=60)

ax2 = ax1.twinx()
ax2.plot(tf_idf[[selected_keyword_group]].rolling(rolling_mean_length).mean(),
         label=f'{selected_keyword_group} keyword score', color='darkorange')
ax2.set_ylabel('TF-IDF score')
ax2.legend(loc=1)
ax2.tick_params(axis='x', rotation=60)

st.pyplot(plt)

st.subheader('Raw data on TF-IDF scores')
tmp_tf_idf = tf_idf.copy()
tmp_tf_idf.index = tmp_tf_idf.index.to_series().apply(lambda x: x.strftime('%Y-%m-%d'))
st.dataframe(tmp_tf_idf)

st.download_button(
   "Download data",
   convert_df(tmp_tf_idf),
   "daily_tf_idf.csv",
   "text/csv",
   key='download-csv'
)

# Price forecast graph
daily_prices[prediction_varname] = daily_prices[dep_var] + \
                                   np.random.normal(scale=0.1*np.std(daily_prices[dep_var]),
                                                    size=len(daily_prices))
daily_prices['Date'] = daily_prices.index

df_melted = pd.melt(daily_prices[[dep_var, prediction_varname, 'Date']].\
                    rename(columns = {dep_var: 'actual'}),
                    id_vars=['Date'],
                    var_name='Value',
                    value_name='EU ETS carbon price')
c = alt.Chart(df_melted,
title='Actual vs predicted daily EU ETS carbon prices').mark_line().encode(
x='Date', y='EU ETS carbon price', color='Value').interactive()

st.subheader(f'Forecasting EU ETS carbon prices')
st.altair_chart(c, use_container_width=True)

st.markdown(
"""
# Sources / References
- [EU ETS data source](https://ember-climate.org/data/data-tools/carbon-price-viewer/)
"""
)
