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
import pickle
import datetime

import plotly.express as px

# setup streamlit
import streamlit as st

# setup database
import sqlite3

st.set_page_config(layout="wide")

# global variables
MODEL = "stored_vars/best_model.pkl"
TRAIN_PATH = "stored_vars/training_data.pkl"
TEST_PATH = "stored_vars/test_data.pkl"
USAGE_PATH = "stored_vars/sf_usage.pkl"
PRED_PATH = "stored_vars/yhat.pkl"
TRUTH_PATH = "stored_vars/y_te.pkl"


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
                );
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
    display_usage()
    display_error_hist()
    display_model_comparison()
    display_test_graph()

if __name__ == "__main__":
    main()