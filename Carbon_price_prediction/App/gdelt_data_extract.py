# Import packages
from datetime import datetime, timedelta

import gdelt
import pandas as pd

# Parameters
prev_date_offset = 1
business_calendar = False
extract_multiple_dates = True

def main():
    if extract_multiple_dates:
        tmp = pd.read_csv("./data/tf_idf_gdelt_lemmatized_aggregated_keywords.csv")
        start_date = pd.to_datetime(tmp.date.max()) + pd.tseries.offsets.Day(1) # or specify manually
        end_date = pd.to_datetime('today') - pd.tseries.offsets.Day(1) # or specify manually
        date_range = pd.date_range(start_date, end_date, freq="d")
        date_range = [date.strftime("%Y-%m-%d") for date in date_range]
        print(date_range)
    else:
        if business_calendar:
            prev_date = pd.to_datetime('today') - pd.tseries.offsets.BDay(prev_date_offset)
        else:
            prev_date = datetime.now() - timedelta(prev_date_offset)
        date = prev_date.strftime("%Y-%m-%d")
        print(date)
    actor_filters = ["THE EUROPEAN UNION", "EUROPEAN UNION"]

    # Data import
    gd = gdelt.gdelt()

    if extract_multiple_dates:
        events = gd.Search(date_range, table='events', output='gpd',
                        normcols=True, coverage=True)
    else:
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

if __name__ == "__main__":
    main()
