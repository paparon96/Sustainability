# Import packages
import gdelt
import pandas as pd

# Parameters
start_date = '2021-08-29'
end_date = '2022-08-29'
dates = [start_date, end_date]
actor_filters = ["THE EUROPEAN UNION", "EUROPEAN UNION"]

# Data import
gd = gdelt.gdelt()
events = gd.Search(dates, table='events', output='gpd',
                   normcols=True, coverage=False)

# Data preprocessing
filtered_events = events[events.actor1name.isin(actor_filters)]
filtered_events = filtered_events.drop_duplicates(subset=['sourceurl', 'sqldate'])
print(filtered_events.head())

filtered_events = filtered_events[['globaleventid', 'sqldate', 'sourceurl']]

# Export dataset
filtered_events.to_csv('./data/gdelt_events.csv', index=False)
