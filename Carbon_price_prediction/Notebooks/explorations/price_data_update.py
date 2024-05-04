import pandas as pd

# Import previous price data
daily_prices = pd.read_csv("./Carbon_price_prediction/Data/new_merged_dataset.csv", index_col=0,
                        parse_dates=True, dayfirst=True)
daily_prices.index.name = 'date'
daily_prices.index = pd.to_datetime(daily_prices.index)
# breakpoint()
# print(daily_prices.head())

# Import new price data
nov_dec_23_prices = pd.read_csv("./Carbon_price_prediction/Notebooks/explorations/prices_eu_ets_23nov21_23dec20.csv", index_col=0, parse_dates=True, dayfirst=True)
nov_dec_23_prices = nov_dec_23_prices.rename(columns={"price": "carbon_price"})[["carbon_price"]]

# Concatenate new daily prices to the old ones
extended_daily_prices = pd.concat([daily_prices, nov_dec_23_prices], axis=0)

# Export extended dataframe
extended_daily_prices.to_csv(f"./Carbon_price_prediction/Data/merged_dataset_{pd.to_datetime('today').strftime('%Y_%m_%d')}.csv")
