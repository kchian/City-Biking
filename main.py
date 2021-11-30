import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pandas_profiling import ProfileReport
import geopandas as gpd
from datetime import datetime

# setup streamlit
import streamlit as st

# setup database
import sqlite3

# global variables
DATABASE_PATH = './data/sf/database.sqlite'

@st.cache(allow_output_mutation=True)
def get_connection():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    return conn

conn = get_connection()

# function to run sql queries on
def run_pd_query(query):
    return pd.read_sql(query, conn)

# SQL query to obtain all of the weather information
@st.cache(hash_funcs={sqlite3.Connection: lambda _: None})
def select_weather():
    WEATHER_QUERY = 'SELECT * FROM weather;'
    weather_df = run_pd_query(WEATHER_QUERY)
    return weather_df

weather_df = select_weather()

# SQL query to create a row for each trip and the start and end station
@st.cache(hash_funcs={sqlite3.Connection: lambda _: None})
def select_trips():
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
    return sf_df

sf_df = select_trips()

# preprocess the data
@st.cache
def preprocess(sf_df, weather_df):
    # remove any rows that have empty columns
    sf_df = sf_df.dropna(how="any")
    weather_df = weather_df.dropna(how="any")

    # remove the outliers based on duration
    z_scores = stats.zscore(sf_df["duration"])

    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3)
    sf_df = sf_df[filtered_entries]

    # convert the start and end dates to pandas datetime
    sf_df["start_date"] = pd.to_datetime(sf_df["start_date"], format='%m/%d/%Y %H:%M', errors="coerce")
    sf_df["end_date"] = pd.to_datetime(sf_df["end_date"], format='%m/%d/%Y %H:%M', errors="coerce")

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

    return sf_df, weather_df

sf_df, weather_df = preprocess(sf_df, weather_df)

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
def joinplot_location(start_stations=True):
    st.title("Joint plot of where the bike trips are ending")
    
    # parameter selector
    start_date = st.date_input('Start date', datetime(2013, 8, 8))
    end_date = st.date_input('End date', datetime(2015, 8, 8))

    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
    if start_stations:
        lat, long = "start_lat", "start_long"
    else:
        lat, long = "end_lat", "end_long"

    mask = (sf_df["start_date"] >= start_date) & (sf_df["end_date"] <= end_date)
    df = sf_df.loc[mask]

    fig = sns.jointplot(data=df, x=long, y=lat)

    st.caption(f"Displaying {df.shape[0]} trips for {start_date} through {end_date}")
    st.pyplot(fig)


def main():
    joinplot_location(start_stations=True)
    # map_rides()

if __name__ == "__main__":
    main()