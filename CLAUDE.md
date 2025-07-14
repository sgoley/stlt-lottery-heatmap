# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a lottery data analysis and visualization project that creates heatmaps for Powerball and Mega Millions lottery numbers. The main application is a Streamlit web app that displays frequency and date-weighted analysis of lottery drawings.

## Dependencies
Core dependencies (defined in setup.py):
- `requests` - for web scraping
- `bs4` (BeautifulSoup) - for HTML parsing
- `numpy` - numerical operations
- `pandas` - data manipulation
- `seaborn` - statistical plotting
- `matplotlib` - visualization
- `streamlit` - web app framework
- `plotly` - interactive visualizations

## Running the Application
- Main Streamlit app: `streamlit run app.py`
- Data collection scripts can be run directly with Python

## Architecture
The project consists of three main components:

1. **app.py** - Main Streamlit application
   - Interactive web interface with sidebar controls for game selection (Powerball/Megamillions)
   - Date range selection (single date or starting date mode)
   - Visualization type selection (Frequency vs Date Weighted)
   - Generates dual heatmaps: main numbers (10x7 grid) and special numbers (4x7 grid)
   - Date weighting uses exponential decay based on recency

2. **pb_mm_csv_link.py** - CSV-based data processing
   - Downloads lottery data from Texas Lottery CSV URLs
   - Focuses on Powerball data analysis
   - Creates frequency heatmaps using matplotlib/seaborn

3. **pb_mm_tsv_page.py** - Web scraping approach
   - Scrapes lottery data from HTML tables using BeautifulSoup
   - Creates interactive heatmaps using Plotly
   - Processes data into structured DataFrame format

## Data Sources
- Texas Lottery CSV endpoints for current data
- Texas Lottery HTML tables for historical data
- Local CSV files in `/data` directory for offline processing

## Key Features
- **Frequency Analysis**: Raw count of number occurrences
- **Date Weighted Analysis**: Recent drawings weighted more heavily using exponential decay
- **Dual Heatmap Display**: Separate visualizations for main numbers and special/power balls
- **Interactive Controls**: Game selection, date filtering, analysis type switching
- **Data Export**: CSV download functionality for filtered datasets