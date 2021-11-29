import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pandas_profiling import ProfileReport
import geopandas as gpd

# setup streamlit
import streamlit as st

# setup database
import sqlite3

DATABASE_PATH = './data/sf/database.sqlite'
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

# function to run sql queries on
def run_pd_query(query):
    return pd.read_sql(query, conn)

# SQL query to obtain all of the weather information
WEATHER_QUERY = 'SELECT * FROM weather;'
weather_df = run_pd_query(WEATHER_QUERY)

# SQL query to create a row for each trip and the start and end station
TRIP_STATION_QUERY = 'SELECT trip.id AS trip_id, \
                            trip.bike_id AS bike_id, \
                            trip.subscription_type AS subscription_type, \
                            trip.duration AS duration, \
                            trip.start_date AS start_date, \
                            trip.end_date AS end_date, \
                            start_station.name AS start_station_name, \
                            end_station.name AS end_station_name, \
                            start_station.lat AS start_lat, \
                            start_station.long AS start_long, \
                            end_station.lat AS end_lat, \
                            end_station.long AS end_long \
                        FROM trip \
                        JOIN station AS start_station \
                            ON trip.start_station_id = start_station.id \
                        JOIN station AS end_station \
                            ON trip.end_station_id = end_station.id;'

sf_df = run_pd_query(TRIP_STATION_QUERY)

# remove any rows that have empty columns
sf_df = sf_df.dropna(how="any")
weather_df = weather_df.dropna(how="any")

# remove the outliers based on duration
z_scores = stats.zscore(sf_df["duration"])

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
sf_df = sf_df[filtered_entries]

# convert the start and end dates to pandas datetime
sf_df["start_date"] = pd.to_datetime(sf_df["start_date"], format='%m/%d/%Y %H:%M')
sf_df["end_date"] = pd.to_datetime(sf_df["end_date"], format='%m/%d/%Y %H:%M')

weather_df["date"] = pd.to_datetime(weather_df["date"], format='%m/%d/%Y')

# add month column
sf_df["start_month"] = sf_df["start_date"].dt.month
sf_df["end_month"] = sf_df["end_date"].dt.month
weather_df["month"] = weather_df["date"].dt.month

# add year column
sf_df["start_year"] = sf_df["start_date"].dt.year
sf_df["end_year"] = sf_df["end_date"].dt.year

# add day column
sf_df["start_day"] = sf_df["start_date"].dt.day_name()
sf_df["end_day"] = sf_df["end_date"].dt.day_name()

# add hour column
sf_df["start_hour"] = sf_df["start_date"].dt.hour
sf_df["end_hour"] = sf_df["end_date"].dt.hour

# make duration into minutes
sf_df["duration_min"] = sf_df["duration"] / 60

# BEGIN APP

st.markdown('''
# Bay Area Bike Share (Aug 2013 - Aug 2015)
### Sachin Dabas | Samarth Gowda | Kevin Chian
### Carnegie Mellon University - Interactive Data Science (05839)
### An Urban Study on Bike Share Demand across the San Francisco Bay Area

This dataset is from the San Francisco Bay Area Bike Share database from August 2013 to August 2015. The bike share is meant to provide people in the Bay Area an easy way to travel around. The dataset is provided as a SQL database and a series of csv files. 

The database and files are available on Kaggle at the following [link](https://www.kaggle.com/benhamner/sf-bay-area-bike-share).

The tables/csv that are available for us to use are `station`, `status`, `trip`, and `weather`. We are interested in combining the trip, station, and weather to create a dataset for each trip. 
''')

st.markdown('''
ADD INFORMATION ABOUT THE STORY WE ARE TRYING TO TELL AND WHAT WE ARE INVESTIGATING
''')

st.write(sf_df.head(10))

# end location for rides
def joinplot_location(start=True):
    # fig = plt.figure(figsize=(10, 4))
    
    if start:
        lat, long = "start_lat", "start_long"
    else:
        lat, long = "end_lat", "end_long"

    fig = sns.jointplot(data=sf_df, x=long, y=lat)

    st.title("Joint plot of where the rides are ending")
    st.pyplot(fig)

# location on the map
# def map_rides():
#     # load in the shape file for SF
#     road_gdf_sf = gpd.read_file("./data/sf/shape/tl_2017_06075_roads.shp")
#     start_stations = pd.DataFrame({'count' : sf_df.groupby( [ "start_long", "start_lat"] ).size()}).reset_index()
#     start_gdf = gpd.GeoDataFrame(start_stations,
#                                     geometry=gpd.points_from_xy(start_stations["start_long"],
#                                                                 start_stations["start_lat"]),
#                                     )
#     # normalize count to values between 0 and 1
#     start_gdf["count"] = ((start_gdf["count"] - start_gdf["count"].min()) /
#                             (start_gdf["count"].max()- start_gdf["count"].min()))

#     ax = road_gdf_sf.plot(figsize=(20, 20))
#     ax.set(ylim=(37.6, 37.9), xlim=(-122.6, -122.3))
#     fig = start_gdf.plot(column="count", ax=ax, cmap="inferno", zorder=3, markersize=start_gdf["count"]*300)

#     st.title("Map of where the bike rides are starting")
#     st.pyplot(fig)


def main():
    joinplot_location()
    # map_rides()

if __name__ == "__main__":
    main()