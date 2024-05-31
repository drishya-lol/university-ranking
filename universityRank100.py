import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
df = pd.read_csv('cwurData.csv')

# Filter top 100 universities for each year
top_100_per_year = df[df['world_rank'] <= 100]

# Prepare data for plotting
years = sorted(top_100_per_year['year'].unique())

# Create a subplot for each year
fig = make_subplots(rows=len(years), cols=1, subplot_titles=[f'Top 100 Universities in {year}' for year in years])

for i, year in enumerate(years, start=1):
    data_year = top_100_per_year[top_100_per_year['year'] == year].sort_values('world_rank', ascending=True)
    
    # Generate colors for gradient effect from red to pink to purple to blue to yellow to green
    colors = []
    num_colors = 100
    for j in range(num_colors):
        if j < num_colors / 5:
            # Red to Pink
            r = 255
            g = int(192 * j / (num_colors / 5))
            b = int(203 * j / (num_colors / 5))
        elif j < 2 * num_colors / 5:
            # Pink to Purple
            r = 255 - int(106 * (j - num_colors / 5) / (num_colors / 5))
            g = 192 - int(192 * (j - num_colors / 5) / (num_colors / 5))
            b = 203 + int(52 * (j - num_colors / 5) / (num_colors / 5))
        elif j < 3 * num_colors / 5:
            # Purple to Blue
            r = 149 - int(149 * (j - 2 * num_colors / 5) / (num_colors / 5))
            g = int(0 * (j - 2 * num_colors / 5) / (num_colors / 5))
            b = 255
        elif j < 4 * num_colors / 5:
            # Blue to Yellow
            r = int(255 * (j - 3 * num_colors / 5) / (num_colors / 5))
            g = int(255 * (j - 3 * num_colors / 5) / (num_colors / 5))
            b = 255 - int(255 * (j - 3 * num_colors / 5) / (num_colors / 5))
        else:
            # Yellow to Green
            r = 255 - int(255 * (j - 4 * num_colors / 5) / (num_colors / 5))
            g = 255
            b = int(0 * (j - 4 * num_colors / 5) / (num_colors / 5))
        
        colors.append(f'rgb({r},{g},{b})')
    
    fig.add_trace(
        go.Bar(x=data_year['institution'], y=data_year['score'], name=f'{year} Scores', marker=dict(color=colors)),
        row=i, col=1
    )

# Update layout
fig.update_layout(height=3000, width=1000, title_text="Top 100 University Rankings from 2012 to 2015", showlegend=False)
fig.update_xaxes(tickangle=45)
fig.show()