import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pandas_profiling import ProfileReport
import geopandas as gpd
from datetime import datetime
import altair as alt
import pydeck as pdk

# setup streamlit
import streamlit as st

# setup database
import sqlite3

st.set_page_config(layout="wide")

# global variables
DATABASE_PATH = './data/sf/database.sqlite'
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MAX_ALLOWED = 100000
N_ROWS = 10

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
    sf_df["start_datetime"] = pd.to_datetime(sf_df["start_date"], format='%m/%d/%Y %H:%M', errors="coerce")
    sf_df["end_datetime"] = pd.to_datetime(sf_df["end_date"], format='%m/%d/%Y %H:%M', errors="coerce")
    
    sf_df["start_date"] = sf_df["start_datetime"].dt.date
    sf_df["end_date"] = sf_df["end_datetime"].dt.date

    weather_df["date"] = pd.to_datetime(weather_df["date"], format='%m/%d/%Y')

    # add month column
    sf_df["start_month"] = sf_df["start_datetime"].dt.month_name()
    sf_df["end_month"] = sf_df["end_datetime"].dt.month_name()
    weather_df["month"] = weather_df["date"].dt.month_name()

    # add year column
    sf_df["start_year"] = sf_df["start_datetime"].dt.year
    sf_df["end_year"] = sf_df["end_datetime"].dt.year

    # add day column
    sf_df["start_day"] = sf_df["start_datetime"].dt.day_name()
    sf_df["end_day"] = sf_df["end_datetime"].dt.day_name()

    # add hour column
    sf_df["start_hour"] = sf_df["start_datetime"].dt.hour
    sf_df["end_hour"] = sf_df["end_datetime"].dt.hour

    # make duration into minutes
    sf_df["duration_min"] = sf_df["duration"] / 60

    return sf_df, weather_df

sf_df, weather_df = preprocess(sf_df, weather_df)

# BEGIN APP

# end location for rides
def filters():
    filter_row_1, filter_row_2 = st.columns((1, 1))

    with filter_row_1:
        selected_hour_least, selected_hour_greatest = st.slider("Select a hour range", 0, 23, (6, 10))

        trip_type = st.radio("Would you like to look at starting stations or ending stations?", ("start", "end"))

        st.write("Select the subscription type to display")
        subscriber = st.checkbox("Subscriber", True)
        customer = st.checkbox("Customer", True)
    
        if trip_type == "start":
            lat_name, long_name = "start_lat", "start_long"
        else:
            lat_name, long_name = "end_lat", "end_long"

        subscription_type = []
        if subscriber:
            subscription_type.append("Subscriber")
        if customer:
            subscription_type.append("Customer")

    with filter_row_2:
        selected_months = st.multiselect("Select the months to view", MONTHS, ["May", "June", "July", "August"])

        selected_days_of_week = st.multiselect("Select the days of the week to view", DAYS_OF_WEEK, ["Monday", "Tuesday", "Wednesday"])

    with st.spinner("Loading ..."):
        mask = (
                (sf_df["subscription_type"].isin(subscription_type)) \
                & (sf_df[f"{trip_type}_day"].isin(selected_days_of_week)) \
                & (sf_df[f"{trip_type}_month"].isin(selected_months)) \
                & (sf_df[f"{trip_type}_hour"] >= selected_hour_least) \
                & (sf_df[f"{trip_type}_hour"] <= selected_hour_greatest)
            )

        mask_sf_df = sf_df.loc[mask]

        lat_long_data = mask_sf_df[[lat_name, long_name]]
        lat_long_data = lat_long_data.rename(columns={lat_name: "lat", long_name: "lon"})

        if lat_long_data.shape[0] > MAX_ALLOWED:
            lat_long_data = lat_long_data.sample(MAX_ALLOWED)

        return lat_long_data, mask_sf_df, trip_type

# Skeleton for this function was taken from streamlit demo (https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/streamlit_app.py)
def display_trips_map(data, lat, lon, zoom):
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": lat,
            "longitude": lon,
            "zoom": zoom,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position=["lon", "lat"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ]
    ))

def display_station_value_counts(data, trip_type, nunique_dates, largest=True, nrows=10):
    station_value_counts = data[f"{trip_type}_station_name"].value_counts()

    if largest:
        station_value_counts = station_value_counts.nlargest(nrows)
    else:
        station_value_counts = station_value_counts.nsmallest(nrows)

    station_value_counts = np.round(station_value_counts / nunique_dates, 3)

    station_value_counts = pd.DataFrame(station_value_counts)

    station_value_counts.columns = ["Avg daily trips"]
    station_value_counts.index.name = "Station name"
    
    st.table(station_value_counts)

def display_average_duration(data):
    avg_duration = np.average(data["duration_min"])
    avg_duration = np.round(avg_duration, 3)
    st.metric(label="Average trip duration", value=f"{avg_duration} mins")

def display_avg_daily_trip_count(data, trip_type, nunique_dates):
    avg_trips = np.average(data[f"{trip_type}_station_name"].value_counts())
    avg_trips = np.round(avg_trips / nunique_dates, 3)
    st.metric(label="Average daily trips per station", value=f"{avg_trips} trips")


def main():

    st.title("An Urban Study on Bike Share Demand across the San Francisco Bay Area")
    st.markdown('''
    Bay Area Bike Share (Aug 2013 - Aug 2015)

    Sachin Dabas | Samarth Gowda | Kevin Chian

    Carnegie Mellon University - Interactive Data Science (05839)

    This dataset is from the San Francisco Bay Area Bike Share database from August 2013 to August 2015. The bike share is meant to provide people in the Bay Area an easy way to travel around. The dataset is provided as a SQL database and a series of csv files. 

    The database and files are available on Kaggle at the following [link](https://www.kaggle.com/benhamner/sf-bay-area-bike-share).

    ''')

    st.markdown('''
    ADD INFORMATION ABOUT THE STORY WE ARE TRYING TO TELL AND WHAT WE ARE INVESTIGATING
    ''')

    # st.dataframe(sf_df.head(10))

    st.header("Understanding demand at bike stations")
    st.write("Change the different settings to explore how demand at bike stations changes depending on the time of the day, month in the year, day of the week, whether or not you are a subscriber and more. You are also able to select between looking at demand at stations where trips start in comparison to where trips are ending.")


    lat_long_data, sf_df, trip_type = filters()

    nunique_dates = sf_df[f"{trip_type}_date"].nunique()
    
    sf_coor = [37.7949, -122.4]
    palo_alto_coor = [37.4419, -122.1430]
    san_jose_coor = [37.3382, -121.8863]

    stat_row_1, stat_row_2, stat_row_3 = st.columns((1, 1, 1))
    with stat_row_1:
        st.metric(label="Displaying trips for ", value=f"{nunique_dates} days")

    with stat_row_2:
        display_average_duration(sf_df)
    
    with stat_row_3:
        display_avg_daily_trip_count(sf_df, trip_type, nunique_dates)


    map_row_1, map_row_2, map_row_3 = st.columns((2, 1, 1))

    with map_row_1:
        st.write("** San Francisco **")
        display_trips_map(lat_long_data, sf_coor[0], sf_coor[1], 12)
    
    with map_row_2:
        st.write("** Palo Alto **")
        display_trips_map(lat_long_data, palo_alto_coor[0], palo_alto_coor[1], 11)
    
    with map_row_3:
        st.write("** San Jose **")
        display_trips_map(lat_long_data, san_jose_coor[0], san_jose_coor[1], 12)

    station_count_row_1, station_count_row_2 = st.columns((1, 1))

    with station_count_row_1:
        st.write(f"** {N_ROWS} most popular stations where trips are {trip_type}ing based on the above configuration **")
        display_station_value_counts(sf_df, trip_type, nunique_dates, True, N_ROWS)

    with station_count_row_2:
        st.write(f"** {N_ROWS} least popular stations where trips are {trip_type}ing based on the above configuration **")
        display_station_value_counts(sf_df, trip_type, nunique_dates, False, N_ROWS)


    

if __name__ == "__main__":
    main()