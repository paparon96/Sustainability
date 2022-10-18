import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path

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
streamlit_cloud_deployment = True

# Constants
control_var_name_map = {'gas_price': 'gas prices',
'oil_price': 'oil prices',
'carbon_price': 'carbon prices',
'energy_price': 'electricity prices',
'coal_price': 'coal prices',
'stock_market_index_level': 'stock market index',}

keyword_var_name_map = {'0': 'Total',
'emissions': 'Emissions',
'fossil_fuel': 'Fossil fuels',
'gas': 'Gas',
'policy': 'Policy',
'renewables': 'Renewables',
}

#### DASHBOARD ####
st.title('EU Climate Change News Index Dashboard')

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

# Set path
if streamlit_cloud_deployment:
    base_path = '/app/sustainability/Carbon_price_prediction/App'
else:
    base_path = '.'


# Data import
daily_prices = pd.read_csv(f"{base_path}/../Data/new_merged_dataset.csv", index_col=0,
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
    tf_idf_file_path = f'{base_path}/../Data/signals'
else:
    tf_idf_file_path = f'{base_path}/data'

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

tf_idf_colnames = list(tf_idf.columns)
selected_keyword_group = st.sidebar.selectbox('Select keyword group for TF-IDF score time series analysis',
                                   tf_idf_colnames,
                                   index=tf_idf_colnames.index('Total'))

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
         label=f'ECCNI - {selected_keyword_group}', color='darkorange')
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
# Methodology
The aim of this study is to compose the ECCNI and to incorporate this index into the forecasts of ETS prices. The following subsection outlines the methodology of the index construction.

## Article collection
Our ECCNI relies on the GDELT database that gathers a wide range of online news with daily frequency. Thus, to focus our analysis, we restricted the dataset to the articles where the actor is \emph{European Union} or \emph{EU} and extracted their URL-s. We chose to filter on the actor to focus on issues and policies that are dealt by the EU. Moreover, the carbon prices are affected by global trends as well; consequently, filtering based on geography would not be adequate.

Moreover, we removed the articles from the database that are coming from unreliable sources. For this purpose we used one of the most cited media bias resource, Media Bias Fact Check (MBFC) \citep{mediabias}. We removed the articles from the data that appeared on ‘questionable’ websites according to the ‘Factual/Sourcing’ category of MBFC\footnote{We are grateful to Courtney Pitcher who fetched the data from MBFC and published an organized dataset on her blog \citep{my_pitcher}.}.


After the filtering, the overall number of news sites reduced from 9,497 to 719 from which our web scraper collected 27,777 articles.

## Feature generation workflow

We performed basic string pre-processing steps on the raw texts using the Natural Language Toolkit (NLTK) package \citep{nltk_package}. This package was also used to lemmatize words with WordNetLemmatizer, which is a more advanced solution than standard stemming because of the addition of morphological analysis.
Since our keyword collection contains several multi-word elements, bigrams and trigrams were also formed with the lemmatizer to create the Term Frequency — Inverse Document Frequency (TF-IDF) matrix, which is one of the most commonly used methods for NLP. The TF-IDF method is an adequate tool to incorporate alternative information to the forecast of financial time series \citep{coyne2017forecasting,lubis2021effect,mittermayer2004forecasting,nikfarjam2010text}.

The rows of our calculated matrix represent the individual articles, and its columns are the elements of the partially external, partially custom defined keyword list. We gathered our keywords around 5 main groups: fossil fuels, renewable energy carriers, energy policy, emissions and gas as an independent topic. We used keyword suggestions from Google Trends and our own intuition to expand the mentioned groups, the complete list of keywords is shown in Table \ref{tab:keywords}. We calculated the score for each keyword so it can also be used for further detailed analysis, but due to the high variance of the occurrences and the strong correlation between the keyword groups (shown on Figure \ref{fig:keyword_corr_heatmap}), we created the EU Climate Change News Index as the aggregated TF-IDF score of the groups\footnote{The TF-IDF scores of the EU Climate Change News Index and the keyword groups is available on the \href{https://ets-news-tracker.streamlitapp.com}{EU ETS news tracker dashboard}}.

## Forecasting models

The first (\emph{TF-IDF}) model includes the lags of the ETS price returns\footnote{By price return of variable $p$ we mean the log return: $\Delta log(p_t)=log(\frac{p_t}{p_{t-1}})$} ($r_{t}$) and the ECCNI ($z_{t}$) as predictors:

"""
)

st.latex(r'''
    \begin{equation}
    r_{t}  = c + \sum_{i=1}^{k} \Bigr( \phi_{i} \, r_{t-i} + \theta_{i} \, z_{t-i} \Bigr),
\end{equation}
    ''')

st.markdown(
"""
while the second model, called \emph{Control}, serves as a benchmark model which considers the lags of the ETS price returns and the fundamental driving factors of carbon prices based on the literature (gas, electricity, coal, oil and stock price returns represented by matrix $X$, and vector  $x_{t}$ for period \textit{t}):
"""
)

st.latex(r'''
\begin{equation}
    r_{t}  = c +
    \sum_{i=1}^{k} \Bigr( \phi_{i} \, r_{t-i} + \beta_{i}^T \, x_{t-i} \Bigr).
\end{equation}
    ''')

st.markdown(
"""
The final, \emph{Full} model includes all predictors: the lags of the ETS price returns, the control variables' price returns and the ECCNI:
"""
)

st.latex(r'''
\begin{equation}
\begin{split}
    r_{t}  = c +
    \sum_{i=1}^{k} \Bigr( \phi_{i} \, r_{t-i} + \beta_{i}^T \, x_{t-i} +
    \theta_{i} \, z_{t-i} \Bigr).
\end{split}
\end{equation}
    ''')

st.markdown(
"""
We use OLS regression and ElasticNet shrinkage method\footnote{We used the following hyperparameters for grid search: $L1 = [0, \, 0.01, \, 0.05, \, 0.1, \, 0.3, \, 0.5, \, 0.7, \, 0.9, \, 0.95,$ $0.99, \, 1]$ and $\alpha = [0, \, 0.001, \, 0.003, \, 0.005, \, 0.007, \, 0.009, \, 0.0095, \, 0.01, \, 0.1, \, 0.3, \, 0.5, \, 0.7, \, 0.9, \, 0.95, \, 0.99, \, 0.999, \, 1]$} to estimate the models.
"""
)

st.markdown(
"""
# Sources / References
- [EU ETS data source](https://ember-climate.org/data/data-tools/carbon-price-viewer/)
"""
)
