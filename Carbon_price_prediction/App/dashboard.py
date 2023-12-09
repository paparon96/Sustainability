import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import altair as alt
import streamlit as st

# Custom functions
@st.cache_data # remove decorator if you want to run the app locally with older `streamlit` version!
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
display_price_forecast_graph = False
matplotlib_dual_plot = False

base_path = './Carbon_price_prediction/App'

# Constants
control_var_name_map = {'gas_price': 'gas prices',
'oil_price': 'oil prices',
'carbon_price': 'carbon prices',
'energy_price': 'electricity prices',
'coal_price': 'coal prices',
'stock_market_index_level': 'stock market index',}

keyword_var_name_map = {'0': 'ECCNI - Total',
'emissions': 'ECCNI - Emissions',
'fossil_fuel': 'ECCNI - Fossil fuels',
'gas': 'ECCNI - Gas',
'policy': 'ECCNI - Policy',
'renewables': 'ECCNI - Renewables',
}

#### DASHBOARD ####
st.title('EU Climate Change News Index Dashboard')

# st.markdown(
# """
# # Authors
# - Aron Hartvig
# - Peter Palos
# - Aron Pap
# """
# )

st.markdown(
"""
# Paper abstract
The emissions trading system is a key driver of emissions reduction in the EU. Carbon prices have been rapidly increasing since 2020 and accurate forecasting of EU Emissions Trading System (ETS) prices has become essential. In this paper, we propose a novel method to generate alternative predictors for ETS prices using GDELT online news database. We compose the EU climate change news index (ECCNI) by calculating term frequency–inverse document frequency (TF-IDF) feature for climate change related keywords. As climate policies are widely discussed in the news, the index is capable of tracking the ongoing debate about climate change in the EU. Finally, we show that incorporating the ECCNI in a simple predictive model robustly improves forecasts of ETS prices compared to a control model where the traditional predictors of carbon prices are included.
\n
The complete research paper is available [online here](https://doi.org/10.1016/j.frl.2023.103720).
"""
)

research_paper_period = st.sidebar.checkbox('Use data from the research paper period (vs most recent data)',
                                value=False)


# Data import
daily_prices = pd.read_csv(f"{base_path}/../Data/new_merged_dataset.csv", index_col=0,
                         parse_dates=True, dayfirst=True)
daily_prices.index.name = 'date'
daily_prices.index = pd.to_datetime(daily_prices.index)

# Switch between research paper analysis vs online/up-to-date data
if research_paper_period:
    tf_idf_file_path = f'{base_path}/../Data/signals'
else:
    tf_idf_file_path = f'{base_path}/data'

tf_idf = pd.read_csv(f'{tf_idf_file_path}/{methodology}_{data_source}_{glossary_source}_{version}keywords.csv',
                     index_col=0, parse_dates=["date"])
tf_idf.index.name = 'date'

tf_idf_aggr = pd.read_csv(f'{tf_idf_file_path}/{methodology}_{data_source}_lemmatized_aggregated_{version}keywords.csv',
                     index_col=0, parse_dates=["date"])
tf_idf_aggr.index.name = 'date'

start_date = st.sidebar.slider(
    "Analysis start date",
    min_value=min(tf_idf.index).to_pydatetime(),
    value=min(tf_idf.index).to_pydatetime(),
    max_value=max(tf_idf.index).to_pydatetime(),
    format="YYYY-MM-DD")

end_date = st.sidebar.slider(
    "Analysis end date",
    min_value=min(tf_idf.index).to_pydatetime(),
    value=max(tf_idf.index).to_pydatetime(),
    max_value=max(tf_idf.index).to_pydatetime(),
    format="YYYY-MM-DD")

# Join aggregated TF-IDF score column
tf_idf = tf_idf.join(tf_idf_aggr)
tf_idf = tf_idf.rename(columns = keyword_var_name_map)

# Keyword group selector
tf_idf_colnames = list(tf_idf.columns)
selected_keyword_group = st.sidebar.selectbox('Select keyword group for TF-IDF score time series analysis',
                                   tf_idf_colnames,
                                   index=tf_idf_colnames.index('ECCNI - Total'))

# Add moving average for selected keyword group
tf_idf[f'{selected_keyword_group}_ma'] = tf_idf[[selected_keyword_group]].rolling(rolling_mean_length).mean()

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


if detailed_tf_idf_keyword_group_data_display:
    st.subheader(f'TF-IDF score history for keyword group: {selected_keyword_group} \
    ({rolling_mean_length}-days moving average)')
    st.line_chart(tf_idf[[list(tf_idf.columns)[0], selected_keyword_group]].rolling(rolling_mean_length).mean())

st.subheader(f'Carbon price time series and TF-IDF score history for {selected_keyword_group} \
({rolling_mean_length}-days moving average)')

ets_price_line_on = st.checkbox('Display ETS price along with TF-IDF score time series',
                                value=True)

if matplotlib_dual_plot:
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(daily_prices[[dep_var]], label='EU ETS carbon price')
    ax1.set_ylabel('EU ETS carbon price')
    ax1.legend(loc=0)
    ax1.tick_params(axis='x', rotation=60)

    ax2 = ax1.twinx()
    ax2.plot(tf_idf[[selected_keyword_group]].rolling(rolling_mean_length).mean(),
             label=selected_keyword_group, color='darkorange')
    ax2.set_ylabel('TF-IDF score')
    ax2.legend(loc=1)
    ax2.tick_params(axis='x', rotation=60)

    st.pyplot(plt)
else:
    tf_idf_melted = tf_idf.copy()
    tf_idf_melted['Date'] = tf_idf_melted.index
    tf_idf_melted = tf_idf_melted.merge(daily_prices[[dep_var]], left_index=True, right_index=True)

    ets_price_line = alt.Chart(tf_idf_melted).mark_line(stroke='#5276A7', interpolate='monotone').encode(
    alt.X('Date', axis=alt.Axis(title=None)), alt.Y(dep_var,
    axis=alt.Axis(title='EU ETS carbon price', titleColor='#5276A7'))).interactive()

    tf_idf_score = alt.Chart(tf_idf_melted).mark_line(stroke='#57A44C', interpolate='monotone').encode(
    alt.X('Date', axis=alt.Axis(title=None)), alt.Y(f'{selected_keyword_group}_ma',
    axis=alt.Axis(title=['TF-IDF score', f'({selected_keyword_group})'], titleColor='#57A44C'))).interactive()

    if ets_price_line_on:
        c = alt.layer(ets_price_line, tf_idf_score).resolve_scale(y='independent')
    else:
        c = alt.layer(tf_idf_score)

    st.altair_chart(c, use_container_width=True)


st.subheader('Raw data on TF-IDF scores')
tmp_tf_idf = tf_idf.copy().drop(columns=[f'{selected_keyword_group}_ma'])
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
if display_price_forecast_graph:
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
