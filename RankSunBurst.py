import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
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
    dcc.Graph(id='sunburst-chart')
])

# Callback to update graph based on dropdown selection
@app.callback(
    Output('sunburst-chart', 'figure'),
    Input('metric-select', 'value')
)
def update_chart(selected_metric):
    # Determine the number of years to plot
    num_years = 2016 - 2012
    # Create a subplot figure with 1 column and a row for each year, specifying the type as 'domain' for sunburst charts
    final_fig = make_subplots(
        rows=num_years, 
        cols=1, 
        specs=[[{'type': 'domain'}] for _ in range(num_years)],  # Each subplot is of type 'domain'
        subplot_titles=[f'{selected_metric.replace("_", " ").title()} in {year}' for year in range(2012, 2016)]
    )

    row = 1
    for i in range(2012, 2016):
        uniYear = universityList[universityList['year'] == i]
        topUniversities = uniYear.iloc[:20, :]
        labels = topUniversities['institution'].values
        parents = [''] * len(labels)  # Sunburst charts require a parent array
        values = topUniversities[selected_metric].values

        # Create a sunburst chart
        fig = go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total"
        )

        # Add the sunburst chart to the appropriate subplot
        final_fig.add_trace(fig, row=row, col=1)
        row += 1

    final_fig.update_layout(height=900 * num_years, width=1600)  # Adjust total size to maintain aspect ratio
    return final_fig

# Run the app on a different port
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)  # Change 8051 to any available port
