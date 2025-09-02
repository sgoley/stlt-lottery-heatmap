# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
A lottery data analysis and visualization project that creates heatmaps for Powerball and Mega Millions lottery numbers. The main application is a Streamlit web app displaying frequency and date-weighted analysis of lottery drawings arranged to match actual lottery ticket layouts.

## Running the Application
- Main Streamlit app: `streamlit run app.py`
- Docker: Build with `docker build -t lottery-heatmap .` and run with `docker run -p 8501:8501 lottery-heatmap`
- Individual data processing scripts: `python pb_mm_csv_link.py` or `python pb_mm_tsv_page.py`

## Dependencies  
Key dependencies (defined in requirements.txt):
- `streamlit>=1.37.1` - web app framework
- `pandas>=2.2.2`, `numpy>=2.3.1` - data manipulation
- `plotly>=5.23.0` - interactive visualizations  
- `matplotlib>=3.10.3`, `seaborn>=0.13.2` - statistical plotting
- `requests>=2.32.4`, `beautifulsoup4>=4.12.3` - web scraping

## Architecture
Multi-page Streamlit application with comprehensive lottery analysis:

**app.py** - Primary Streamlit application featuring:
- **Multi-page structure**: History Analysis and Gap Analysis pages
- **Dual data source modes**: CSV Direct vs Web Scraping (both pages)
- **History Page**: Traditional frequency and date-weighted analysis
  - Interactive sidebar controls for game/date/visualization selection
  - Two analysis modes: Frequency (raw counts) vs Date Weighted (exponential decay by recency)
  - Date range filtering with inclusive start/end dates and helpful tooltips
  - Row selection for targeted heatmap analysis
  - Dual heatmap visualization: main numbers (10x7 grid) + special/power balls (4x7 grid)
  - CSV export functionality for filtered datasets
- **Gap Analysis Page**: Historical vs recent pattern comparison
  - Dual date range selection (historical baseline vs recent period)
  - Preset options: Last 30/60 days, Year to date, Since last jackpot win, Custom
  - Difference calculation: Historical Frequency - Recent Frequency
  - Red-blue divergent heatmaps showing "overdue" vs "hot" numbers
  - Top 5 most overdue numbers analysis with frequency gaps

**pb_mm_csv_link.py** - CSV-based data processor:
- Direct CSV download from Texas Lottery endpoints
- Matplotlib/seaborn-based static heatmap generation
- Focused primarily on Powerball analysis

**pb_mm_tsv_page.py** - Web scraping processor:
- BeautifulSoup-based HTML table extraction
- Plotly interactive heatmap generation
- Handles richer web data (jackpot amounts, winners, etc.)

## Data Sources & Structure
- **Live CSV URLs**: Texas Lottery endpoints for Powerball/Mega Millions current data
- **Web Scraping URLs**: HTML table data with additional jackpot/winner information  
- **Local Data**: `/data/texas/` contains CSV samples, `/data/` has historical datasets
- **Ticket Layouts**: `/ticketlayouts/` contains reference images for various state lottery card layouts

## Key Technical Patterns
- **Multi-page Navigation**: Sidebar page selector for History vs Gap Analysis modes
- **Date Filtering**: Supports single-date overlay, date-range frequency analysis, and dual date range comparison
- **Number Grid Layout**: Mimics actual lottery ticket arrangements (10x7 main + 4x7 special)
- **Matrix Creation Logic**: Uses `divmod(value, 7)` for proper lottery grid positioning (col=quotient, row=remainder)
- **Exponential Decay Weighting**: Recent drawings weighted more heavily using configurable decay factor
- **Gap Analysis Algorithm**: Calculates frequency differences between historical and recent periods
- **Dual Visualization Strategy**: Plotly for interactive heatmaps with proper axis labeling and number annotations
- **Data Source Flexibility**: Both pages support CSV and web scraping with unified data processing pipelines

## Gap Analysis Use Cases
- **Year-to-date vs Last 30 days**: Identify historically common numbers that haven't appeared recently
- **Long historical vs Recent periods**: Find patterns that might be "due" based on historical frequency
- **Since last jackpot**: Compare pre/post jackpot number patterns using jackpot winner data
- **Custom date ranges**: Flexible comparison of any two time periods for strategic analysis