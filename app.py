# from datetime import datetime
import time
import numpy as np
import pandas as pd

# import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


mega_csv_url = "https://www.texaslottery.com/export/sites/lottery/Games/Mega_Millions/Winning_Numbers/megamillions.csv"
pb_csv_url = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/powerball.csv"

mega_table_num_order_url = "https://www.texaslottery.com/export/sites/lottery/Games/Mega_Millions/Winning_Numbers/index.html_2013354932.html"
mega_table_draw_order_url = "https://www.texaslottery.com/export/sites/lottery/Games/Mega_Millions/Winning_Numbers/index.html_1444349437.html"
mega_table_xpath = '//*[@id="content"]/div/div/table'

pb_table_num_order_url = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/index.html_2013354932.html"
pb_table_draw_order_url = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/index.html_1444349437.html"
pb_table_xpath = '//*[@id="content"]/div/div/table'

mega_df = pd.read_csv(
    mega_csv_url,
    names=[
        "game",
        "month",
        "day",
        "year",
        "ball_1",
        "ball_2",
        "ball_3",
        "ball_4",
        "ball_5",
        "megaball",
        "multiplier",
    ],
)
mega_df["date"] = pd.to_datetime(mega_df[["year", "month", "day"]])

# Using "with" notation
with st.sidebar:
    selected_one_or_more = st.selectbox(
        "Single or Starting Date mode",
        ("Single", "Starting"),
    )

    selected_date = st.selectbox(
        label="Starting with what date would you like to review?",
        key="date_select",
        options=mega_df["date"].sort_values(ascending=False),
    )

    selected_type = st.selectbox(
        label="How would you like to view results?",
        key="type_select",
        options=["Frequency", "Date Weighted"],
    )

    if selected_one_or_more == "Single":
        selected_df = mega_df[mega_df["date"] == selected_date]
    else:
        selected_df = mega_df[mega_df["date"] >= selected_date]

    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode("utf-8")

    csv = convert_df(selected_df.reset_index())

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="megamillions.csv",
        mime="text/csv",
    )

    # main ticket frequency prep
    main_df_select = selected_df.iloc[:, 4:9]
    main_balls = main_df_select.values.flatten() - 1
    main_array = np.zeros((10, 7), dtype=int)
    for value in main_balls:
        col, row = divmod(value, 7)
        main_array[col, row] += 1
    main_ticket = main_array

    # mega ticket frequency prep
    mega_df_select = selected_df.iloc[:, 9]
    mega_balls = mega_df_select.values.flatten() - 1
    mega_array = np.zeros((4, 7), dtype=int)
    for value in mega_balls:
        col, row = divmod(value, 7)
        mega_array[col, row] += 1
    mb_ticket = mega_array

    # begin weighted calculation
    # dates selection for date weighting
    mm_date_series = selected_df["date"].values

    # main ticket recency prep
    values = main_df_select.values.flatten() - 1  # Adjust values to be 0-indexed
    timestamps = pd.to_datetime(mm_date_series).astype(int) / 10**9
    array = np.zeros((10, 7), dtype=float)
    current_time = np.max(timestamps)
    weights = np.exp(-(current_time - timestamps) / (current_time - np.min(timestamps)))
    for value, weight in zip(values, weights):
        col, row = divmod(value, 7)
        array[col, row] += weight
    weight_main_ticket = array

    # mega ticket recency prep
    values = mega_df_select.values.flatten() - 1  # Adjust values to be 0-indexed
    timestamps = pd.to_datetime(mm_date_series).astype(int) / 10**9
    array = np.zeros((4, 7), dtype=float)
    current_time = np.max(timestamps)
    weights = np.exp(-(current_time - timestamps) / (current_time - np.min(timestamps)))
    for value, weight in zip(values, weights):
        col, row = divmod(value, 7)
        array[col, row] += weight
    weight_mb_ticket = array

    if selected_type == "Date Weighted":
        chart_df = weight_main_ticket
        subchart_df = weight_mb_ticket
    else:
        chart_df = main_ticket
        subchart_df = mb_ticket

with st.container():
    st.write("Ticket Layout")

    st.write(selected_df)

    main_x_axis_labels = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
    ]  # labels for x-axis
    main_y_axis_labels = [1, 10, 20, 30, 40, 50, 60]  # labels for y-axis

    mb_x_axis_labels = [1, 2, 3, 4, 5, 6, 7]  # labels for x-axis
    mb_y_axis_labels = [1, 10, 20]  # labels for y-axis

    fig, ax = plt.subplots(2, 1)

    sns.heatmap(
        chart_df,
        ax=ax[0],
        # cmap="crest",
        annot=True,
        xticklabels=main_x_axis_labels,
        # yticklabels=main_y_axis_labels,
    )
    sns.heatmap(
        subchart_df,
        ax=ax[1],
        # cmap="crest",
        annot=True,
        # xticklabels=mb_x_axis_labels,
        # yticklabels=mb_y_axis_labels,
    )

    st.pyplot(fig)
