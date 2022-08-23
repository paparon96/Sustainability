# Import packages
import gdelt
import pandas as pd

# Parameters
start_date = '2022-07-25'
end_date = '2022-08-01'
dates = [start_date, end_date]
actor_filters = ["THE EUROPEAN UNION", "EUROPEAN UNION"]

# Data import
gd = gdelt.gdelt()
events = gd.Search(dates, table='events', output='gpd',
                   normcols=True, coverage=False)

# Data preprocessing
filtered_events = events[events.actor1name.isin(actor_filters)]

urls = list(filtered_events.sourceurl)
print(urls)
