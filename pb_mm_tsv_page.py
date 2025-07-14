import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objs as go
from plotly.offline import plot

pb_link = 'https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/index.html_1444349437.html'
mm_link = 'https://www.texaslottery.com/export/sites/lottery/Games/Mega_Millions/Winning_Numbers/index.html_1444349437.html'

URL = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/index.html_1444349437.html"
response = requests.get(URL)
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table
table = soup.find("table")

# Extract table headers
headers = [header.text.strip() for header in table.findAll("th")]

# Extract rows from table
rows = table.findAll("tr")
data = []
for row in rows[1:]:  # Skip the header
    columns = row.findAll("td")
    columns = [column.text.strip() for column in columns]
    data.append(columns)

# Convert to pandas DataFrame
df = pd.DataFrame(data, columns=headers)
df.columns = ['date', 'winning_numbers', 'powerball', 'powerplay', 'estimated_jackpot', 'jackpot_winners', 'jackpot_option']
filtered_df = df[df['winning_numbers'].notna()]
filtered_df['date'] = pd.to_datetime(filtered_df['date'], format='%m/%d/%Y')

# Split the 'winning_numbers' column into separate columns
split_df = filtered_df['winning_numbers'].str.split(r'\s*-\s*', expand=True)
# Rename the columns in the split_df if needed
split_df.columns = [f'ball_{i+1}' for i in range(split_df.shape[1])]
# Concatenate the split_df with the original DataFrame, keeping all columns
result_df = pd.concat([filtered_df, split_df], axis=1)


# Create a list of lists containing the values for the heatmap
z = [list(map(int, row)) for _, row in result_df.iterrows()]

# Create a heatmap trace
heatmap_trace = go.Heatmap(z=z, x=result_df.columns, y=result_df.index, colorscale='Viridis')

# Create a layout
layout = go.Layout(
    title='Heatmap from DataFrame',
    xaxis=dict(title='X-axis Label'),
    yaxis=dict(title='Y-axis Label')
)

# Create a figure
fig = go.Figure(data=[heatmap_trace], layout=layout)

# Display the heatmap (you can also use plotly.offline.plot to save it as an HTML file)
plot(fig)
