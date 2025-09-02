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

# === MAIN NAVIGATION ===
with st.sidebar:
    st.title("🎲 Lottery Analysis")
    st.divider()
    
    # Page selection
    page = st.selectbox(
        "📋 Select Analysis Page", 
        ["History", "Gap Analysis", "Neighbors"],
        format_func=lambda x: {
            "History": "📊 History Analysis", 
            "Gap Analysis": "🔍 Gap Analysis", 
            "Neighbors": "🎯 Neighbors Analysis"
        }[x]
    )
    
    st.divider()
    st.caption("Navigate between different analysis modes using the selector above.")

mega_csv_url = "https://www.texaslottery.com/export/sites/lottery/Games/Mega_Millions/Winning_Numbers/megamillions.csv"
pb_csv_url = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/powerball.csv"

mega_table_num_order_url = "https://www.texaslottery.com/export/sites/lottery/Games/Mega_Millions/Winning_Numbers/index.html_2013354932.html"
mega_table_draw_order_url = "https://www.texaslottery.com/export/sites/lottery/Games/Mega_Millions/Winning_Numbers/index.html_1444349437.html"
mega_table_xpath = '//*[@id="content"]/div/div/table'

pb_table_num_order_url = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/index.html_2013354932.html"
pb_table_draw_order_url = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/index.html_1444349437.html"
pb_table_xpath = '//*[@id="content"]/div/div/table'

if page == "History":
    st.title("📊 Lottery History Analysis")
    st.info("🎯 **Goal**: Analyze lottery number frequencies and patterns over time with historical data.")
    
    # Sidebar controls (navigation, game, and data source only)
    with st.sidebar:
        selected_game = st.selectbox(
            "🎲 Game Selection",
            ("Powerball", "Megamillions"),
            index=0
        )

        data_source = st.selectbox(
            "📡 Data Source",
            options=["CSV Direct", "Web Scraping"],
            index=0,
            key="source_select"
        )
    
    # Set game URL based on selection
    if selected_game == "Powerball":
        game_csv_url = pb_csv_url
    elif selected_game == "Megamillions":
        game_csv_url = mega_csv_url

    # Data loading functions
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

    # Load data based on source selection
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

    # === ANALYSIS CONFIGURATION ===
    st.divider()
    st.subheader("🔧 Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Jackpot Winner Filtering (only for web scraped data)
        if data_source == "Web Scraping" and 'jackpot_winners' in drawing_df.columns:
            jackpot_filter_option = st.selectbox(
                "🎰 Filter by Jackpot Status",
                ["All Drawings", "Winners Only (No Rolls)", "Rolls Only"],
                index=0,
                help="Filter drawings based on jackpot outcome"
            )
            
            if jackpot_filter_option == "Winners Only (No Rolls)":
                drawing_df = drawing_df[drawing_df['jackpot_winners'] != 'Roll']
                st.success(f"✅ Filtered to {len(drawing_df)} jackpot winner drawings")
            elif jackpot_filter_option == "Rolls Only":
                drawing_df = drawing_df[drawing_df['jackpot_winners'] == 'Roll']
                st.info(f"🔄 Filtered to {len(drawing_df)} roll-over drawings")

        selected_one_or_more = st.selectbox(
            "📅 Date Analysis Mode",
            ("Single", "Starting"),
            index=1,
            help="Single: Analyze one specific date. Starting: Analyze from a date onwards."
        )
        
    with col2:
        selected_date = st.selectbox(
            "📆 Select Date",
            options=drawing_df["date"].sort_values(ascending=False),
            format_func=lambda x: str(x.date()),
            help="Choose your analysis date"
        )

        selected_type = st.selectbox(
            "📈 Analysis Type",
            options=["Frequency", "Date Weighted"],
            help="Frequency: Raw counts. Date Weighted: Recent draws weighted more heavily."
        )

    # Visualization settings
    st.divider()
    st.subheader("🎨 Visualization Settings")
    
    visualization_type = st.selectbox(
        "📊 Chart Type",
        options=["Matplotlib/Seaborn", "Plotly Interactive"],
        index=1,
        help="Choose your preferred visualization library"
    )

    # Process data based on selection
    if selected_one_or_more == "Single":
        selected_df = drawing_df[drawing_df["date"] == selected_date]
    else:
        selected_df = drawing_df[drawing_df["date"] >= selected_date]

    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode("utf-8")
    
    # === DATA PREVIEW ===
    st.divider()
    st.subheader("📋 Data Preview & Selection")


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

elif page == "Gap Analysis":
    st.title("🔍 Gap Analysis - Historical vs Recent Patterns")
    st.info("🎯 **Goal**: Compare historical baselines against recent patterns to identify 'hot' and 'cold' numbers that may be due for selection.")
    
    # Sidebar controls (navigation, game, and data source only)
    with st.sidebar:
        selected_game = st.selectbox(
            "🎲 Game Selection",
            ("Powerball", "Megamillions"),
            key="gap_game"
        )
        
        data_source = st.selectbox(
            "📡 Data Source",
            options=["CSV Direct", "Web Scraping"],
            index=0,
            key="gap_source_select"
        )
    
    # Set game URL based on selection
    if selected_game == "Powerball":
        game_csv_url = pb_csv_url
    else:
        game_csv_url = mega_csv_url
    
    # Load data functions
    @st.cache_data
    def load_gap_csv_data(url):
        df = pd.read_csv(url, names=["game", "month", "day", "year", "ball_1", "ball_2", "ball_3", "ball_4", "ball_5", "special", "multiplier"])
        # Add null columns to match web scraping structure
        df['estimated_jackpot'] = 'N/A'
        df['jackpot_winners'] = 'N/A'
        df['jackpot_option'] = 'N/A'
        df['multiplier_details'] = df['multiplier'].astype(str) + 'X'
        return df
    
    @st.cache_data
    def scrape_gap_web_data(game_type):
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
    
    # Load data based on source selection
    if data_source == "CSV Direct":
        drawing_df = load_gap_csv_data(game_csv_url)
    else:
        scraped_df = scrape_gap_web_data(selected_game)
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
            drawing_df = load_gap_csv_data(game_csv_url)
    
    drawing_df["date"] = pd.to_datetime(drawing_df[["year", "month", "day"]])
    
    # === ANALYSIS CONFIGURATION ===
    st.divider()
    st.subheader("⚙️ Analysis Configuration")
    
    # Jackpot Winner Filtering (only for web scraped data)
    if data_source == "Web Scraping" and 'jackpot_winners' in drawing_df.columns:
        jackpot_filter_option = st.selectbox(
            "🎰 Filter by Jackpot Status",
            ["All Drawings", "Winners Only (No Rolls)", "Rolls Only"],
            index=0,
            key="gap_jackpot_filter",
            help="Filter drawings based on jackpot outcome"
        )
        
        if jackpot_filter_option == "Winners Only (No Rolls)":
            drawing_df = drawing_df[drawing_df['jackpot_winners'] != 'Roll']
            st.success(f"✅ Filtered to {len(drawing_df)} jackpot winner drawings")
        elif jackpot_filter_option == "Rolls Only":
            drawing_df = drawing_df[drawing_df['jackpot_winners'] == 'Roll']
            st.info(f"🔄 Filtered to {len(drawing_df)} roll-over drawings")
    
    # === DATE RANGE SELECTION ===
    st.divider()
    
    # Historical Period Selection
    st.subheader("📊 Historical Period (Baseline)")
    col1, col2 = st.columns(2)
    
    with col1:
        hist_start = st.selectbox(
            "Historical Start Date",
            options=drawing_df["date"].sort_values(ascending=False),
            format_func=lambda x: str(x.date()),
            help="Start date for historical baseline pattern (inclusive)"
        )
    
    with col2:
        hist_end = st.selectbox(
            "Historical End Date", 
            options=drawing_df["date"].sort_values(ascending=False),
            format_func=lambda x: str(x.date()),
            help="End date for historical baseline pattern (inclusive)"
        )
    
    # Recent Period Selection with Presets
    st.subheader("🔥 Recent Period (To Subtract)")
    
    preset = st.selectbox(
        "Quick Preset Options",
        ["Custom", "Last 30 days", "Last 60 days", "Year to date"],
        help="Choose a preset or select Custom for manual date selection"
    )
    
    if preset == "Last 30 days":
        recent_start = drawing_df["date"].max() - pd.Timedelta(days=30)
        recent_end = drawing_df["date"].max()
        st.info(f"Using last 30 days: {recent_start.date()} to {recent_end.date()}")
    elif preset == "Last 60 days":
        recent_start = drawing_df["date"].max() - pd.Timedelta(days=60)
        recent_end = drawing_df["date"].max()
        st.info(f"Using last 60 days: {recent_start.date()} to {recent_end.date()}")
    elif preset == "Year to date":
        recent_start = pd.Timestamp(f"{drawing_df['date'].max().year}-01-01")
        recent_end = drawing_df["date"].max()
        st.info(f"Using year to date: {recent_start.date()} to {recent_end.date()}")
    else:
        col3, col4 = st.columns(2)
        with col3:
            recent_start = st.selectbox(
                "Recent Start Date",
                options=drawing_df["date"].sort_values(ascending=False),
                format_func=lambda x: str(x.date()),
                help="Start date for recent period (inclusive)",
                key="recent_start"
            )
        with col4:
            recent_end = st.selectbox(
                "Recent End Date",
                options=drawing_df["date"].sort_values(ascending=False), 
                format_func=lambda x: str(x.date()),
                help="End date for recent period (inclusive)",
                key="recent_end"
            )
    
    # Filter data
    hist_df = drawing_df[(drawing_df["date"] >= hist_start) & (drawing_df["date"] <= hist_end)]
    recent_df = drawing_df[(drawing_df["date"] >= recent_start) & (drawing_df["date"] <= recent_end)]
    
    # Calculate frequency matrices
    def calc_frequency_matrix(df, ball_type='main'):
        if ball_type == 'main':
            df_select = df[['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5']]
            shape = (10, 7)
            max_val = 69
        else:
            df_select = df[['special']]
            shape = (4, 7) 
            max_val = 27
        
        df_select = df_select.apply(pd.to_numeric, errors='coerce')
        balls = df_select.values.flatten() - 1
        balls = balls[~np.isnan(balls)]
        
        matrix = np.zeros(shape, dtype=int)
        for val in balls:
            if 0 <= val <= max_val:
                col, row = divmod(int(val), 7)
                if col < shape[0]:
                    matrix[col, row] += 1
        return matrix
    
    # Calculate matrices for both periods
    hist_main = calc_frequency_matrix(hist_df, 'main')
    recent_main = calc_frequency_matrix(recent_df, 'main')
    hist_special = calc_frequency_matrix(hist_df, 'special')
    recent_special = calc_frequency_matrix(recent_df, 'special')
    
    # Calculate gaps (historical - recent)
    gap_main = hist_main - recent_main
    gap_special = hist_special - recent_special
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Historical Drawings", len(hist_df))
    with col2:
        st.metric("Recent Drawings", len(recent_df))
    with col3:
        st.metric("Analysis Span", f"{len(hist_df) - len(recent_df)} drawing difference")
    
    # Gap Analysis Heatmaps
    st.subheader("🔍 Gap Analysis Results")
    st.write("**Red areas**: Numbers that appeared more historically than recently (potentially 'overdue')")
    st.write("**Blue areas**: Numbers that appeared more recently than historically (potentially 'hot')")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Main numbers heatmap
        fig_main = px.imshow(
            gap_main,
            title=f"Main Numbers Gap Analysis ({selected_game})",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            labels=dict(color="Frequency Gap")
        )
        
        # Add number annotations with proper positioning
        for i in range(10):
            for j in range(7):
                number = i * 7 + j + 1
                if number <= 70:
                    fig_main.add_annotation(
                        x=j, y=i, text=str(number), showarrow=False,
                        font=dict(color="white" if abs(gap_main[i,j]) > gap_main.std() else "black", size=12)
                    )
        
        # Update axis labels manually
        fig_main.update_xaxes(
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=["1", "2", "3", "4", "5", "6", "7"],
            title="Position"
        )
        fig_main.update_yaxes(
            tickmode='array',
            tickvals=list(range(10)),
            ticktext=["1-7", "8-14", "15-21", "22-28", "29-35", "36-42", "43-49", "50-56", "57-63", "64-70"],
            title="Range"
        )
        
        st.plotly_chart(fig_main, use_container_width=True)
    
    with col2:
        # Special numbers heatmap
        max_special = 26 if selected_game == "Powerball" else 25
        
        fig_special = px.imshow(
            gap_special,
            title=f"Special Numbers Gap Analysis ({selected_game})", 
            color_continuous_scale="RdBu_r",
            aspect="auto",
            labels=dict(color="Frequency Gap")
        )
        
        # Add number annotations with proper positioning
        for i in range(4):
            for j in range(7):
                number = i * 7 + j + 1
                if number <= max_special:
                    fig_special.add_annotation(
                        x=j, y=i, text=str(number), showarrow=False,
                        font=dict(color="white" if abs(gap_special[i,j]) > gap_special.std() else "black", size=12)
                    )
        
        # Update axis labels manually
        fig_special.update_xaxes(
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=["1", "2", "3", "4", "5", "6", "7"],
            title="Position"
        )
        fig_special.update_yaxes(
            tickmode='array',
            tickvals=list(range(4)),
            ticktext=["1-7", "8-14", "15-21", "22-28"],
            title="Range"
        )
        
        st.plotly_chart(fig_special, use_container_width=True)
    
    # Most overdue numbers analysis
    st.subheader("📈 Most 'Overdue' Numbers")
    
    # Find most overdue main numbers
    main_gaps = [(i * 7 + j + 1, gap_main[i, j]) for i in range(10) for j in range(7) if i * 7 + j + 1 <= 70]
    main_gaps.sort(key=lambda x: x[1], reverse=True)
    
    special_gaps = [(i * 7 + j + 1, gap_special[i, j]) for i in range(4) for j in range(7) if i * 7 + j + 1 <= 28]
    special_gaps.sort(key=lambda x: x[1], reverse=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Main Numbers:**")
        for i, (num, gap) in enumerate(main_gaps[:5]):
            if gap > 0:
                st.write(f"🔴 **{num}**: +{gap} gap")
    
    with col2:
        st.write("**Special Numbers:**")
        for i, (num, gap) in enumerate(special_gaps[:5]):
            if gap > 0:
                st.write(f"🔴 **{num}**: +{gap} gap")
    
    st.info("💡 **Interpretation**: Positive gaps suggest numbers that appeared more frequently in the historical period compared to recent draws. This could indicate numbers that are 'due' based on historical patterns.")

elif page == "Neighbors":
    st.title("🎯 Neighbors Analysis - Regional Lottery Ticket Patterns")
    st.info("🎯 **Goal**: Analyze lottery numbers based on their proximity on the ticket layout to identify regional patterns and clustering behaviors.")
    
    # Sidebar controls (navigation, game, and data source only)
    with st.sidebar:
        selected_game = st.selectbox(
            "🎲 Game Selection",
            ("Powerball", "Megamillions"),
            key="neighbors_game"
        )
        
        data_source = st.selectbox(
            "📡 Data Source",
            options=["CSV Direct", "Web Scraping"],
            index=0,
            key="neighbors_source_select"
        )
    
    # Set game URL based on selection
    if selected_game == "Powerball":
        game_csv_url = pb_csv_url
    else:
        game_csv_url = mega_csv_url
    
    # Load data (reuse existing functions)
    @st.cache_data
    def load_neighbors_csv_data(url):
        df = pd.read_csv(url, names=["game", "month", "day", "year", "ball_1", "ball_2", "ball_3", "ball_4", "ball_5", "special", "multiplier"])
        df['estimated_jackpot'] = 'N/A'
        df['jackpot_winners'] = 'N/A'
        df['jackpot_option'] = 'N/A'
        df['multiplier_details'] = df['multiplier'].astype(str) + 'X'
        return df
    
    @st.cache_data
    def scrape_neighbors_web_data(game_type):
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
            
            df = pd.DataFrame(data, columns=headers)
            df = df[df.iloc[:, 1].notna()]
            
            result_df = pd.DataFrame()
            result_df['date'] = pd.to_datetime(df.iloc[:, 0], format='%m/%d/%Y')
            
            winning_numbers = df.iloc[:, 1].str.split(r'\s*-\s*', expand=True)
            winning_numbers.columns = [f'ball_{i+1}' for i in range(winning_numbers.shape[1])]
            
            for col in winning_numbers.columns:
                result_df[col] = winning_numbers[col]
            
            result_df['special'] = df.iloc[:, 2]
            result_df['multiplier_details'] = df.iloc[:, 3] if df.shape[1] > 3 else 'N/A'
            result_df['estimated_jackpot'] = df.iloc[:, 4] if df.shape[1] > 4 else 'N/A'
            result_df['jackpot_winners'] = df.iloc[:, 5] if df.shape[1] > 5 else 'N/A'
            result_df['jackpot_option'] = df.iloc[:, 6] if df.shape[1] > 6 else 'N/A'
            
            result_df['game'] = game_type
            result_df['month'] = result_df['date'].dt.month
            result_df['day'] = result_df['date'].dt.day
            result_df['year'] = result_df['date'].dt.year
            result_df['multiplier'] = 1
            
            return result_df
        return None
    
    # Load data based on source selection
    if data_source == "CSV Direct":
        drawing_df = load_neighbors_csv_data(game_csv_url)
    else:
        scraped_df = scrape_neighbors_web_data(selected_game)
        if scraped_df is not None:
            drawing_df = scraped_df
            ball_columns = ['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'special']
            for col in ball_columns:
                drawing_df[col] = pd.to_numeric(drawing_df[col], errors='coerce')
            drawing_df = drawing_df.dropna(subset=ball_columns)
            drawing_df[ball_columns] = drawing_df[ball_columns].astype(int)
            drawing_df['month'] = drawing_df['date'].dt.month
            drawing_df['day'] = drawing_df['date'].dt.day
            drawing_df['year'] = drawing_df['date'].dt.year
        else:
            st.error("Failed to scrape web data, falling back to CSV")
            drawing_df = load_neighbors_csv_data(game_csv_url)
    
    drawing_df["date"] = pd.to_datetime(drawing_df[["year", "month", "day"]])
    
    # === ANALYSIS CONFIGURATION ===
    st.divider()
    st.subheader("🔧 Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        neighbor_radius = st.slider(
            "🎯 Neighbor Radius",
            min_value=1, max_value=3, value=1,
            help="How many adjacent cells to include in neighborhood analysis (1 = immediate neighbors, 2 = extended area)"
        )
        
        analysis_mode = st.selectbox(
            "📊 Analysis Mode",
            ["Regional Heat", "Cluster Density", "Cross-Pattern"],
            help="Regional Heat: Shows accumulated frequency in neighborhoods. Cluster Density: Shows how often neighbors appear together. Cross-Pattern: Shows correlation between adjacent positions."
        )
    
    with col2:
        # Jackpot Winner Filtering (only for web scraped data)
        if data_source == "Web Scraping" and 'jackpot_winners' in drawing_df.columns:
            jackpot_filter_option = st.selectbox(
                "🎰 Filter by Jackpot Status",
                ["All Drawings", "Winners Only (No Rolls)", "Rolls Only"],
                index=0,
                key="neighbors_jackpot_filter",
                help="Filter drawings based on jackpot outcome"
            )
            
            if jackpot_filter_option == "Winners Only (No Rolls)":
                drawing_df = drawing_df[drawing_df['jackpot_winners'] != 'Roll']
                st.success(f"✅ Filtered to {len(drawing_df)} jackpot winner drawings")
            elif jackpot_filter_option == "Rolls Only":
                drawing_df = drawing_df[drawing_df['jackpot_winners'] == 'Roll']
                st.info(f"🔄 Filtered to {len(drawing_df)} roll-over drawings")
    
    # === DATE RANGE SELECTION ===
    st.divider()
    
    # Date range selection (simplified for neighbors analysis)
    st.subheader("📅 Analysis Period")
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.selectbox(
            "Start Date",
            options=drawing_df["date"].sort_values(ascending=False),
            format_func=lambda x: str(x.date()),
            help="Start date for neighbor analysis (inclusive)"
        )
    
    with col2:
        end_date = st.selectbox(
            "End Date",
            options=[None] + list(drawing_df["date"].sort_values(ascending=False)),
            format_func=lambda x: "Most Recent" if x is None else str(x.date()),
            help="End date for neighbor analysis (inclusive)"
        )
    
    # Filter data
    if end_date is None:
        filtered_df = drawing_df[drawing_df["date"] >= start_date]
    else:
        filtered_df = drawing_df[(drawing_df["date"] >= start_date) & (drawing_df["date"] <= end_date)]
    
    st.metric("Total Drawings in Analysis", len(filtered_df))
    
    # Neighbor Analysis Functions
    def get_neighbors(row, col, radius=1):
        """Get all neighbors within radius of a grid position"""
        neighbors = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:  # Skip center
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 10 and 0 <= new_col < 7:  # Within bounds
                    neighbors.append((new_row, new_col))
        return neighbors
    
    def calculate_neighbor_heat(df, analysis_mode, radius):
        """Calculate neighbor-based heat matrix"""
        main_df_select = df[['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5']]
        main_df_select = main_df_select.apply(pd.to_numeric, errors='coerce')
        
        # Create base frequency matrix
        base_matrix = np.zeros((10, 7), dtype=float)
        neighbor_matrix = np.zeros((10, 7), dtype=float)
        
        # Calculate base frequencies
        for _, draw in main_df_select.iterrows():
            draw_numbers = [int(x) - 1 for x in draw.dropna() if 0 <= int(x) - 1 <= 69]
            draw_positions = [(num // 7, num % 7) for num in draw_numbers]
            
            for row, col in draw_positions:
                base_matrix[row, col] += 1
                
                if analysis_mode == "Regional Heat":
                    # Add weighted heat to neighbors
                    neighbors = get_neighbors(row, col, radius)
                    for nr, nc in neighbors:
                        # Weight decreases with distance
                        distance = max(abs(nr - row), abs(nc - col))
                        weight = 1.0 / (distance + 1)
                        neighbor_matrix[nr, nc] += weight
                        
                elif analysis_mode == "Cluster Density":
                    # Check how many neighbors are also in this draw
                    neighbors = get_neighbors(row, col, radius)
                    neighbor_count = sum(1 for nr, nc in neighbors if (nr, nc) in draw_positions)
                    neighbor_matrix[row, col] += neighbor_count
                    
                elif analysis_mode == "Cross-Pattern":
                    # Check for cross patterns (adjacent in same draw)
                    neighbors = get_neighbors(row, col, 1)  # Only immediate neighbors
                    for nr, nc in neighbors:
                        if (nr, nc) in draw_positions:
                            neighbor_matrix[row, col] += 1
                            neighbor_matrix[nr, nc] += 1
        
        if analysis_mode == "Regional Heat":
            return neighbor_matrix + base_matrix  # Combine base frequency with neighbor heat
        else:
            return neighbor_matrix
    
    # Calculate neighbor heat matrices
    main_heat_matrix = calculate_neighbor_heat(filtered_df, analysis_mode, neighbor_radius)
    
    # Special numbers neighbor analysis (simplified for 4x7 grid)
    def calculate_special_neighbor_heat(df, analysis_mode, radius):
        special_df_select = df[['special']].apply(pd.to_numeric, errors='coerce')
        
        base_matrix = np.zeros((4, 7), dtype=float)
        neighbor_matrix = np.zeros((4, 7), dtype=float)
        
        for _, draw in special_df_select.iterrows():
            special_num = int(draw['special']) - 1
            if 0 <= special_num <= 27:
                row, col = special_num // 7, special_num % 7
                if row < 4:  # Within special number grid bounds
                    base_matrix[row, col] += 1
                    
                    if analysis_mode == "Regional Heat":
                        # Add heat to neighbors within 4x7 grid
                        for dr in range(-radius, radius + 1):
                            for dc in range(-radius, radius + 1):
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = row + dr, col + dc
                                if 0 <= nr < 4 and 0 <= nc < 7:
                                    distance = max(abs(dr), abs(dc))
                                    weight = 1.0 / (distance + 1)
                                    neighbor_matrix[nr, nc] += weight
        
        return neighbor_matrix + base_matrix if analysis_mode == "Regional Heat" else base_matrix
    
    special_heat_matrix = calculate_special_neighbor_heat(filtered_df, analysis_mode, neighbor_radius)
    
    # Visualization
    st.subheader("🌡️ Neighbor Heat Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Main numbers heatmap
        fig_main = px.imshow(
            main_heat_matrix,
            title=f"Main Numbers - {analysis_mode} (Radius: {neighbor_radius})",
            color_continuous_scale="YlOrRd",
            aspect="auto",
            labels=dict(color="Heat Score")
        )
        
        # Add number annotations
        for i in range(10):
            for j in range(7):
                number = i * 7 + j + 1
                if number <= 70:
                    fig_main.add_annotation(
                        x=j, y=i, text=str(number), showarrow=False,
                        font=dict(color="white" if main_heat_matrix[i,j] > main_heat_matrix.mean() else "black", size=11)
                    )
        
        # Update axis labels
        fig_main.update_xaxes(
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=["1", "2", "3", "4", "5", "6", "7"],
            title="Position"
        )
        fig_main.update_yaxes(
            tickmode='array',
            tickvals=list(range(10)),
            ticktext=["1-7", "8-14", "15-21", "22-28", "29-35", "36-42", "43-49", "50-56", "57-63", "64-70"],
            title="Range"
        )
        
        st.plotly_chart(fig_main, use_container_width=True)
    
    with col2:
        # Special numbers heatmap
        max_special = 26 if selected_game == "Powerball" else 25
        
        fig_special = px.imshow(
            special_heat_matrix,
            title=f"Special Numbers - {analysis_mode} (Radius: {neighbor_radius})",
            color_continuous_scale="YlOrRd",
            aspect="auto",
            labels=dict(color="Heat Score")
        )
        
        # Add number annotations
        for i in range(4):
            for j in range(7):
                number = i * 7 + j + 1
                if number <= max_special:
                    fig_special.add_annotation(
                        x=j, y=i, text=str(number), showarrow=False,
                        font=dict(color="white" if special_heat_matrix[i,j] > special_heat_matrix.mean() else "black", size=11)
                    )
        
        # Update axis labels
        fig_special.update_xaxes(
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=["1", "2", "3", "4", "5", "6", "7"],
            title="Position"
        )
        fig_special.update_yaxes(
            tickmode='array',
            tickvals=list(range(4)),
            ticktext=["1-7", "8-14", "15-21", "22-28"],
            title="Range"
        )
        
        st.plotly_chart(fig_special, use_container_width=True)
    
    # Analysis insights
    st.subheader("🔥 Hottest Neighbor Regions")
    
    # Find hottest regions (3x3 areas with highest combined heat)
    def find_hot_regions(matrix, region_size=3):
        hot_regions = []
        rows, cols = matrix.shape
        for i in range(rows - region_size + 1):
            for j in range(cols - region_size + 1):
                region_heat = matrix[i:i+region_size, j:j+region_size].sum()
                center_number = ((i + region_size//2) * 7) + (j + region_size//2) + 1
                hot_regions.append((center_number, region_heat, (i + region_size//2, j + region_size//2)))
        return sorted(hot_regions, key=lambda x: x[1], reverse=True)
    
    main_hot_regions = find_hot_regions(main_heat_matrix)
    special_hot_regions = find_hot_regions(special_heat_matrix, region_size=2)  # Smaller regions for special numbers
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Hottest Main Number Regions:**")
        for i, (center_num, heat, pos) in enumerate(main_hot_regions[:5]):
            if center_num <= 70:
                st.write(f"🔥 **Region {i+1}**: Center #{center_num} - Heat Score: {heat:.1f}")
    
    with col2:
        st.write("**Hottest Special Number Regions:**")
        max_special = 26 if selected_game == "Powerball" else 25
        for i, (center_num, heat, pos) in enumerate(special_hot_regions[:5]):
            if center_num <= max_special:
                st.write(f"🔥 **Region {i+1}**: Center #{center_num} - Heat Score: {heat:.1f}")
    
    # Explanation based on analysis mode
    st.info({
        "Regional Heat": "💡 **Regional Heat** shows accumulated frequency in neighborhoods. Higher scores indicate areas where numbers frequently appear with their neighbors.",
        "Cluster Density": "💡 **Cluster Density** shows how often numbers appear alongside their neighbors in the same drawing. Higher scores indicate clustering patterns.",
        "Cross-Pattern": "💡 **Cross-Pattern** shows correlation between adjacent positions. Higher scores indicate numbers that frequently have neighbors drawn in the same drawing."
    }[analysis_mode])
