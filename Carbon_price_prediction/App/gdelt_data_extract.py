# Import packages
import gdelt
import pandas as pd

# Parameters
start_date = '2022-08-15'
end_date = '2022-08-15'
dates = [start_date, end_date]
prev_business_day = pd.to_datetime('today') - pd.tseries.offsets.BDay(1)
date = prev_business_day.strftime("%Y-%m-%d")
print(date)
actor_filters = ["THE EUROPEAN UNION", "EUROPEAN UNION"]

# Data import
gd = gdelt.gdelt()
events = gd.Search(date, table='events', output='gpd',
                   normcols=True, coverage=True)

# Data preprocessing
filtered_events = events[events.actor1name.isin(actor_filters)]
filtered_events = filtered_events.drop_duplicates(subset=['sourceurl', 'sqldate'])
print(filtered_events.head())
print(filtered_events.shape)

filtered_events = filtered_events[['globaleventid', 'sqldate', 'sourceurl']]

# Export dataset
filtered_events.to_csv('./data/gdelt_events.csv', index=False)
