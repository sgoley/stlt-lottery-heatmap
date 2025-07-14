import time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

# Configure Streamlit for wide layout
st.set_page_config(layout="wide", page_title="Lottery Heatmap Analysis")


mega_csv_url = "https://www.texaslottery.com/export/sites/lottery/Games/Mega_Millions/Winning_Numbers/megamillions.csv"
pb_csv_url = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/powerball.csv"

mega_table_num_order_url = "https://www.texaslottery.com/export/sites/lottery/Games/Mega_Millions/Winning_Numbers/index.html_2013354932.html"
mega_table_draw_order_url = "https://www.texaslottery.com/export/sites/lottery/Games/Mega_Millions/Winning_Numbers/index.html_1444349437.html"
mega_table_xpath = '//*[@id="content"]/div/div/table'

pb_table_num_order_url = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/index.html_2013354932.html"
pb_table_draw_order_url = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/index.html_1444349437.html"
pb_table_xpath = '//*[@id="content"]/div/div/table'

# Using "with" notation
with st.sidebar:
    selected_game = st.selectbox(
        "Which game's results are you interested in reviewing?",
        ("Powerball", "Megamillions"),
        index=0
    )

    if selected_game == "Powerball":
        game_csv_url = pb_csv_url
    elif selected_game == "Megamillions":
        game_csv_url = mega_csv_url
    else:
        # placeholder for additional future game's once clean csv sources can be found
        pass

    data_source = st.selectbox(
        label="Data Source",
        key="source_select",
        options=["CSV Direct", "Web Scraping"],
        index=0
    )

    @st.cache_data
    def load_csv_data(url):
        df = pd.read_csv(
            url,
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
                "special",
                "multiplier",
            ],
        )
        # Add null columns to match web scraping structure
        df['estimated_jackpot'] = 'N/A'
        df['jackpot_winners'] = 'N/A'
        df['jackpot_option'] = 'N/A'
        df['multiplier_details'] = df['multiplier'].astype(str) + 'X'
        return df
    
    @st.cache_data
    def scrape_web_data(game_type):
        if game_type == "Powerball":
            url = pb_table_draw_order_url
        else:
            url = mega_table_draw_order_url
            
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find("table")
        
        if table:
            headers = [header.text.strip() for header in table.findAll("th")]
            rows = table.findAll("tr")
            data = []
            for row in rows[1:]:
                columns = row.findAll("td")
                columns = [column.text.strip() for column in columns]
                data.append(columns)
            
            # Create DataFrame with original headers to understand structure
            df = pd.DataFrame(data, columns=headers)
            
            # Filter out rows with missing data
            df = df[df.iloc[:, 1].notna()]  # Filter by second column (winning numbers)
            
            # Extract all data including rich web information
            result_df = pd.DataFrame()
            result_df['date'] = pd.to_datetime(df.iloc[:, 0], format='%m/%d/%Y')
            
            # Split the winning numbers (format: "12 - 23 - 24 - 31 - 56")
            winning_numbers = df.iloc[:, 1].str.split(r'\s*-\s*', expand=True)
            winning_numbers.columns = [f'ball_{i+1}' for i in range(winning_numbers.shape[1])]
            
            # Add the split winning numbers to result
            for col in winning_numbers.columns:
                result_df[col] = winning_numbers[col]
            
            # Add special ball (mega ball/powerball)
            result_df['special'] = df.iloc[:, 2]
            
            # Add all the rich web data (columns 3-6 from table)
            result_df['multiplier_details'] = df.iloc[:, 3] if df.shape[1] > 3 else 'N/A'
            result_df['estimated_jackpot'] = df.iloc[:, 4] if df.shape[1] > 4 else 'N/A'
            result_df['jackpot_winners'] = df.iloc[:, 5] if df.shape[1] > 5 else 'N/A'
            result_df['jackpot_option'] = df.iloc[:, 6] if df.shape[1] > 6 else 'N/A'
            
            # Add standard columns for compatibility
            result_df['game'] = game_type
            result_df['month'] = result_df['date'].dt.month
            result_df['day'] = result_df['date'].dt.day
            result_df['year'] = result_df['date'].dt.year
            result_df['multiplier'] = 1  # Default multiplier
            
            return result_df
        return None
    
    if data_source == "CSV Direct":
        drawing_df = load_csv_data(game_csv_url)
    else:
        scraped_df = scrape_web_data(selected_game)
        if scraped_df is not None:
            drawing_df = scraped_df
            # Convert ball columns to numeric, handling any non-numeric values
            ball_columns = ['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'special']
            for col in ball_columns:
                drawing_df[col] = pd.to_numeric(drawing_df[col], errors='coerce')
            
            # Remove rows with any NaN values in ball columns
            drawing_df = drawing_df.dropna(subset=ball_columns)
            
            # Convert to int after cleaning
            drawing_df[ball_columns] = drawing_df[ball_columns].astype(int)
            
            drawing_df['month'] = drawing_df['date'].dt.month
            drawing_df['day'] = drawing_df['date'].dt.day
            drawing_df['year'] = drawing_df['date'].dt.year
        else:
            st.error("Failed to scrape web data, falling back to CSV")
            drawing_df = load_csv_data(game_csv_url)

    drawing_df["date"] = pd.to_datetime(drawing_df[["year", "month", "day"]])
    max_ticket_num = drawing_df["ball_5"].max()
    max_special_num = drawing_df["special"].max()

    selected_one_or_more = st.selectbox(
        "Single or Starting Date mode",
        ("Single", "Starting"),
        index=1
    )

    selected_date = st.selectbox(
        label="Starting with what date would you like to review?",
        key="date_select",
        options=drawing_df["date"].sort_values(ascending=False),
    )

    selected_type = st.selectbox(
        label="How would you like to view results?",
        key="type_select",
        options=["Frequency", "Date Weighted"],
    )
    
    visualization_type = st.selectbox(
        label="Visualization Type",
        key="viz_select",
        options=["Matplotlib/Seaborn", "Plotly Interactive"],
        index=1
    )

    if selected_one_or_more == "Single":
        selected_df = drawing_df[drawing_df["date"] == selected_date]
    else:
        selected_df = drawing_df[drawing_df["date"] >= selected_date]

    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode("utf-8")


# Enable row selection in dataframe
selected_rows = st.dataframe(
    selected_df,
    on_select="rerun",
    selection_mode="multi-row",
    use_container_width=True
)

# Use selected rows for visualization, or all rows if none selected
if selected_rows['selection']['rows']:
    filtered_df = selected_df.iloc[selected_rows['selection']['rows']]
    st.info(f"Using {len(filtered_df)} selected rows for visualization")
else:
    filtered_df = selected_df
    st.info("Select specific rows to filter the heatmap, or leave unselected to show all data")

# Download button for filtered data
csv = convert_df(filtered_df.reset_index())
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name="lottery_filtered.csv",
    mime="text/csv",
)

# main ticket frequency prep
main_df_select = filtered_df[['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5']]
# Ensure all values are numeric before subtraction
main_df_select = main_df_select.apply(pd.to_numeric, errors='coerce')
main_balls = main_df_select.values.flatten() - 1
# Filter out NaN values
main_balls = main_balls[~np.isnan(main_balls)]
main_array = np.zeros((10, 7), dtype=int)
for value in main_balls:
    # Ensure value is within valid range (0-69 after subtracting 1)
    if 0 <= value <= 69:
        col, row = divmod(int(value), 7)
        main_array[col, row] += 1
main_ticket = main_array

# special ticket frequency prep
drawing_df_select = filtered_df['special']
# Ensure all values are numeric before subtraction
drawing_df_select = pd.to_numeric(drawing_df_select, errors='coerce')
main_balls = drawing_df_select.values.flatten() - 1
# Filter out NaN values
main_balls = main_balls[~np.isnan(main_balls)]
main_array = np.zeros((4, 7), dtype=int)
for value in main_balls:
    # Ensure value is within valid range (0-27 after subtracting 1 for 4x7 grid)
    if 0 <= value <= 27:
        col, row = divmod(int(value), 7)
        main_array[col, row] += 1
special_ticket = main_array

# begin weighted calculation
# dates selection for date weighting
mm_date_series = filtered_df["date"].values

# main ticket recency prep
# Ensure main_df_select is numeric for date weighting
main_df_select_numeric = main_df_select.apply(pd.to_numeric, errors='coerce')
values = main_df_select_numeric.values.flatten() - 1  # Adjust values to be 0-indexed
# Filter out NaN values and corresponding timestamps
valid_mask = ~np.isnan(values)
values = values[valid_mask]
timestamps = pd.to_datetime(mm_date_series).astype(int) / 10**9
# Repeat timestamps for each ball column and filter
timestamps_repeated = np.tile(timestamps, main_df_select_numeric.shape[1])
timestamps = timestamps_repeated[valid_mask]

array = np.zeros((10, 7), dtype=float)
current_time = np.max(timestamps)
weights = np.exp(-(current_time - timestamps) / (current_time - np.min(timestamps)))
for value, weight in zip(values, weights):
    # Ensure value is within valid range (0-69 after subtracting 1)
    if 0 <= value <= 69:
        col, row = divmod(int(value), 7)
        array[col, row] += weight
weight_main_ticket = array

# special ticket recency prep
# Ensure drawing_df_select is numeric for date weighting
drawing_df_select_numeric = pd.to_numeric(drawing_df_select, errors='coerce')
values = drawing_df_select_numeric.values.flatten() - 1  # Adjust values to be 0-indexed
# Filter out NaN values and corresponding timestamps
valid_mask = ~np.isnan(values)
values = values[valid_mask]
timestamps = pd.to_datetime(mm_date_series).astype(int) / 10**9
timestamps = timestamps[valid_mask]

array = np.zeros((4, 7), dtype=float)
current_time = np.max(timestamps)
weights = np.exp(-(current_time - timestamps) / (current_time - np.min(timestamps)))
for value, weight in zip(values, weights):
    # Ensure value is within valid range (0-27 after subtracting 1 for 4x7 grid)
    if 0 <= value <= 27:
        col, row = divmod(int(value), 7)
        array[col, row] += weight
weight_special_ticket = array

if selected_type == "Date Weighted":
    chart_df = weight_main_ticket
    subchart_df = weight_special_ticket
else:
    chart_df = main_ticket
    subchart_df = special_ticket

with st.container():
    st.write("Ticket Layout")

    main_x_axis_labels = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
    ]
    main_y_axis_labels = ["1-7", "8-14", "15-21", "22-28", "29-35", "36-42", "43-49", "50-56", "57-63", "64-70"]

    special_x_axis_labels = [1, 2, 3, 4, 5, 6, 7]
    special_y_axis_labels = ["1-7", "8-14", "15-21", "22-28"]

    if visualization_type == "Matplotlib/Seaborn":
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        
        sns.heatmap(
            chart_df,
            ax=ax[0],
            cmap="YlGnBu",
            annot=True,
            fmt='.1f' if selected_type == "Date Weighted" else 'd',
            xticklabels=main_x_axis_labels,
            yticklabels=main_y_axis_labels,
            cbar_kws={'label': 'Frequency'}
        )
        ax[0].set_title(f'{selected_game} Main Numbers - {selected_type}')
        
        sns.heatmap(
            subchart_df,
            ax=ax[1],
            cmap="YlGnBu",
            annot=True,
            fmt='.1f' if selected_type == "Date Weighted" else 'd',
            xticklabels=special_x_axis_labels,
            yticklabels=special_y_axis_labels,
            cbar_kws={'label': 'Frequency'}
        )
        ax[1].set_title(f'{selected_game} Special Numbers - {selected_type}')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        # Create hover text matrices with exact numbers
        main_hover_text = []
        for row in range(10):
            hover_row = []
            for col in range(7):
                exact_number = row * 7 + col + 1
                if exact_number <= 70:  # Powerball/Mega Millions main numbers go up to 70
                    hover_row.append(f"Number: {exact_number}<br>Frequency: {chart_df[row, col]}")
                else:
                    hover_row.append(f"N/A<br>Frequency: {chart_df[row, col]}")
            main_hover_text.append(hover_row)
        
        special_hover_text = []
        for row in range(4):
            hover_row = []
            for col in range(7):
                exact_number = row * 7 + col + 1
                max_special = 26 if selected_game == "Powerball" else 25  # Powerball: 26, Mega Millions: 25
                if exact_number <= max_special:
                    hover_row.append(f"Number: {exact_number}<br>Frequency: {subchart_df[row, col]}")
                else:
                    hover_row.append(f"N/A<br>Frequency: {subchart_df[row, col]}")
            special_hover_text.append(hover_row)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_main = px.imshow(
                chart_df,
                labels=dict(x="Position", y="Range", color="Frequency"),
                x=main_x_axis_labels,
                y=main_y_axis_labels,
                title=f'{selected_game} Main Numbers - {selected_type}',
                color_continuous_scale='Viridis',
                text_auto=True
            )
            fig_main.update_traces(hovertemplate='%{customdata}<extra></extra>', customdata=main_hover_text)
            fig_main.update_layout(height=400)
            st.plotly_chart(fig_main, use_container_width=True)
        
        with col2:
            fig_special = px.imshow(
                subchart_df,
                labels=dict(x="Position", y="Range", color="Frequency"),
                x=special_x_axis_labels,
                y=special_y_axis_labels,
                title=f'{selected_game} Special Numbers - {selected_type}',
                color_continuous_scale='Viridis',
                text_auto=True
            )
            fig_special.update_traces(hovertemplate='%{customdata}<extra></extra>', customdata=special_hover_text)
            fig_special.update_layout(height=400)
            st.plotly_chart(fig_special, use_container_width=True)
    
    st.subheader("Data Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Drawings", len(filtered_df))
    with col2:
        st.metric("Date Range", f"{filtered_df['date'].min().strftime('%Y-%m-%d')} to {filtered_df['date'].max().strftime('%Y-%m-%d')}")
    with col3:
        max_pos = np.unravel_index(chart_df.argmax(), chart_df.shape)
        max_number = max_pos[0] * 7 + max_pos[1] + 1
        st.metric("Most Frequent Main Number", max_number)
