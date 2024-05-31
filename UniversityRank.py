import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import numpy as np
from plotly.subplots import make_subplots

# Load data
universityList = pd.read_csv('cwurData.csv')

# Initialize the Dash app
app = Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Dropdown(
        id='metric-select',
        options=[
            {'label': 'Alumni Employment', 'value': 'alumni_employment'},
            {'label': 'Citations', 'value': 'citations'},
            {'label': 'Patents', 'value': 'patents'},
            {'label': 'Quality of Education', 'value': 'quality_of_education'}
        ],
        value='alumni_employment'
    ),
    dcc.Graph(id='bar-chart')
])

# Callback to update graph based on dropdown selection
@app.callback(
    Output('bar-chart', 'figure'),
    Input('metric-select', 'value')
)
def update_chart(selected_metric):
    figures = []
    # Generate 20 distinct colors
    colors = ['#' + ''.join([np.random.choice(list('0123456789ABCDEF')) for _ in range(6)]) for _ in range(20)]
    for i in range(2012, 2016):
        uniYear = universityList[universityList['year'] == i]
        topUniversities = uniYear.iloc[:20, :]
        xvals = topUniversities['world_rank'].values
        yvals = topUniversities[selected_metric].values
        university_names = topUniversities['institution'].values

        # Create a bar chart
        fig = go.Figure(data=[
            go.Bar(x=xvals, y=yvals, text=university_names, marker_color=colors)
        ])

        fig.update_layout(
            title=f'{selected_metric.replace("_", " ").title()} in {i}',
            xaxis_title='World Rank',
            yaxis_title=f'{selected_metric.replace("_", " ").title()} ({i})',  # Include year in y-axis title
            xaxis_tickangle=-45,
            height=450,  # Adjust height to maintain 16:9 aspect ratio
            width=800    # Adjust width to maintain 16:9 aspect ratio
        )
        figures.append(fig)

    # Combine all figures into one figure with subplots for each year
    final_fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for idx, fig in enumerate(figures, start=1):
        for trace in fig.data:
            final_fig.add_trace(trace, row=idx, col=1)
        final_fig.update_layout(height=1800, width=800)  # Adjust total size to maintain aspect ratio

    return final_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)