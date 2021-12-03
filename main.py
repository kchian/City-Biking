import pandas as pd
import numpy as np
import pydeck as pdk
import pickle
import datetime
import plotly.express as px
import streamlit as st


st.set_page_config(
        page_title="An Urban Study on Bike Share Demand across the San Francisco Bay Area",
        page_icon="🚲",
        layout="wide"
    )

# GLOBAL VARS
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MAX_ALLOWED = 100000
N_ROWS = 10
MODEL = "stored_vars/best_model.pkl"
TRAIN_PATH = "stored_vars/training_data.pkl"
TEST_PATH = "stored_vars/test_data.pkl"
USAGE_PATH = "stored_vars/sf_usage.pkl"
PRED_PATH = "stored_vars/yhat.pkl"
TRUTH_PATH = "stored_vars/y_te.pkl"

@st.cache
def load_csv():
    jan = pd.read_csv("./data/sf/trips/jan.csv")
    feb = pd.read_csv("./data/sf/trips/feb.csv")
    mar = pd.read_csv("./data/sf/trips/mar.csv")
    apr = pd.read_csv("./data/sf/trips/apr.csv")
    may = pd.read_csv("./data/sf/trips/may.csv")
    june = pd.read_csv("./data/sf/trips/june.csv")
    july = pd.read_csv("./data/sf/trips/july.csv")
    aug = pd.read_csv("./data/sf/trips/aug.csv")
    sep = pd.read_csv("./data/sf/trips/sep.csv")
    oct = pd.read_csv("./data/sf/trips/oct.csv")
    nov = pd.read_csv("./data/sf/trips/nov.csv")
    dec = pd.read_csv("./data/sf/trips/dec.csv")

    sf_df = pd.concat([jan, feb, mar, apr, may, june, july, aug, sep, oct, nov, dec], ignore_index=True)


    weather_df = pd.read_csv("./data/sf/weather.csv")

    return sf_df, weather_df

sf_df, weather_df = load_csv()

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

# ML MODEL
def load_df(path, sort=True):
    with open(path, "rb") as f:
        df = pickle.load(f)
        df_agg = df.groupby(["month", "day"]).agg(date=("date", "min"), usage=("usage", "mean"))
        df_agg["date"] = df_agg["date"].apply(lambda i: i.replace(year=2014))
        df_agg = df_agg.sort_values("date")
        if sort:
            df = df.sort_values("date")
    return df, df_agg
    
train_df, train_agg = load_df(TRAIN_PATH)

test_df, test_agg = load_df(TEST_PATH, sort=False)
with open(PRED_PATH, "rb") as f:
    yhat = pickle.load(f)
usage_df, usage_agg = load_df(USAGE_PATH)

stations = pd.read_csv("data/sf/station.csv")
stations_map = {name: id for name, id in zip(stations["name"], stations["id"])}

# Skeleton for this function was taken from streamlit demo (https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/streamlit_app.py)
# @st.cache 
def display_usage():
    option = st.selectbox(
        'Start Station',
        tuple(["All stations"] + list(stations_map.keys())))
    
    highlight = st.radio(
        'Highlight Weekends',
        ("True", "False"))
    
    if option == "All stations":
        data = usage_agg
        x = "date"
        title=f'Average Usage in All Stations Across Years'

    else:
        data = usage_df[usage_df["start_station_id"] == stations_map[option]]
        x = "date"
        title=f'Average Usage in {option} by Date'
    fig = px.line(data, x="date", y="usage", title=title)
    
    # Highlight weekends
    if highlight == "True":
        start_dt = data["date"].min()
        end_dt = data["date"].max()
        cur = start_dt
        while cur < end_dt:
            if cur.weekday() >= 5:
                fig.add_vrect(
                    x0=cur-pd.Timedelta(days=0.5), x1=cur+pd.Timedelta(days=0.5),
                    fillcolor="LightSalmon", opacity=0.5,
                    layer="below", line_width=0,
                )
            cur += datetime.timedelta(days=1)
    # for d in weekend_rows["date"].iteritems():
    #     d = d[1]
    #     fig.add_vrect(
    #         x0=d, x1=d+pd.Timedelta(days=1),
    #         fillcolor="LightSalmon", opacity=0.5,
    #         layer="below", line_width=0,
    #     );
    if option == "All stations":
        fig.update_xaxes(
            dtick="M1",
            tickformat="%b")
    st.plotly_chart(fig, use_container_width=True)

def display_error_hist():
    error = test_df["usage"] - yhat
    fig = px.histogram(error, title="Regression Model Error (MAE: 3.50)")
    st.plotly_chart(fig, use_container_width=True)


def display_model_comparison():
    # hard coded bar grpah from the ipynb file
    
    # data = {
    #     "Lasso - MSE": 75.99630450481934,
    #     "Lasso - MAE": 5.244548440456221,
    #     "Decision Tree (max depth 10) - MSE": 42.90820231599476,
    #     "Decision Tree (max depth 10) - MAE": 4.076241427075703,
    #     "Decision Tree (max depth 20) - MSE": 56.83001558861096,
    #     "Decision Tree (max depth 20) - MAE": 4.674037766715455,
    #     "KNeighborsRegressor (neighbors 3) - MSE": 43.018884903647134,
    #     "KNeighborsRegressor (neighbors 3) - MAE": 4.226156041497936,
    #     "KNeighborsRegressor (neighbors 5) - MSE": 39.30108064329516,
    #     "KNeighborsRegressor (neighbors 5) - MAE": 4.023302094180261,
    #     "GradientBoostingRegressor - MSE": 29.460115545532794,
    #     "GradientBoostingRegressor - MAE": 3.504163064866572,
    # }
    data = {
        "Lasso": [75.99630450481934,5.244548440456221],
        "Decision Tree (max depth 10)": [42.90820231599476,4.076241427075703],
        "Decision Tree (max depth 20)": [56.83001558861096,4.674037766715455],
        "KNeighborsRegressor (neighbors 3)": [43.018884903647134,4.226156041497936],
        "KNeighborsRegressor (neighbors 5)": [39.30108064329516,4.023302094180261],
        "GradientBoostingRegressor": [29.460115545532794,3.504163064866572],
    }
    
    data = pd.DataFrame.from_dict(data, orient='index', columns=["MSE", "MAE"])
    fig = px.bar(data, title="Model Comparison")
    fig.update_layout(barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    

def display_test_graph():
    preds = pd.DataFrame(data={"predicted_usage":yhat})
    data = pd.concat((test_df, preds), axis=1)
    data = data.groupby(["month", "day"]).agg(
        date=("date", "min"), 
        usage=("usage", "mean"), 
        predicted_usage=("predicted_usage", "mean"), 
        is_weekend=("is_weekend", "max"))
    data = data.melt(id_vars='date', value_vars=['predicted_usage', 'usage'])
    data = data.sort_values("date")
    fig = px.line(data, x='date', y='value', color='variable', title="Average Test Point vs Average Prediction per Day")
    # fig = px.line(data, x="date", y="usage", color="is_weekend", title="Average Test Point vs Average prediction")
    # fig.add_trace(px.scatter(data, x="date", y="predicted_usage", color="yellow").data[0])
    # fig.update_layout(barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    

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

    display_usage()
    display_error_hist()
    display_model_comparison()
    display_test_graph()


if __name__ == "__main__":
    main()