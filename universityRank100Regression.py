import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to create and train a regression model
def regressionModel(df, predictors, target):
    # Invert the target scores
    max_score = df[target].max() + 1  # Ensure the maximum score is less than the subtracted value
    df[target] = max_score - df[target]

    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Check for columns with all missing values and remove them from predictors if necessary
    all_missing_columns = [col for col in predictors if df[col].isna().all()]
    if all_missing_columns:
        print(f"Removing columns with all missing values: {all_missing_columns}")
        predictors = [col for col in predictors if col not in all_missing_columns]

    # Impute missing values for remaining predictors
    if predictors:  # Ensure there are still predictors left after removal
        imputer = SimpleImputer(strategy='mean')
        df[predictors] = imputer.fit_transform(df[predictors])

    # Drop rows where the target is NaN
    df = df.dropna(subset=[target])
    
    # Check if DataFrame is still empty after handling missing values
    if df.empty:
        raise ValueError("DataFrame is empty after handling missing values")

    X = df[predictors]
    y = df[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check if any split part is empty
    if X_train.empty or X_test.empty:
        raise ValueError("Training or testing set is empty. Adjust the split parameters or data preprocessing.")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Invert predictions back to original scale for interpretation
    y_pred = max_score - y_pred
    y_test = max_score - y_test

    return y_pred, y_test

# Function to train individual regression models for each predictor
def train_individual_regressions_combined_plot(df, predictors, target):
    plt.figure(figsize=(14, 8))
    colors = sns.color_palette("hsv", len(predictors))
    
    # Ensure the target column has no missing values
    df = df.dropna(subset=[target])
    
    results = {}
    for i, predictor in enumerate(predictors):
        # Prepare the DataFrame for the current predictor
        data = df[[predictor, target]].dropna()
        
        # Check if DataFrame is empty after dropping missing values
        if data.empty:
            print(f"No data available for predictor {predictor}")
            continue
        
        # Split the data
        X = data[[predictor]]  # DataFrame format required by sklearn
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Store results
        results[predictor] = {
            'model': model,
            'mse': mse
        }
        
        # Plotting
        sns.regplot(x=X_test[predictor], y=y_test, scatter_kws={'color': colors[i]}, line_kws={'color': colors[i], 'label': f'{predictor}'})

    plt.title('Combined Regression Analysis for All Predictors')
    plt.xlabel('Predictor Value')
    plt.ylabel(target)
    plt.legend()
    plt.show()
    
    return results

# Load data
df = pd.read_csv('cwurData.csv')

# Filter top 100 universities for each year
top_100_per_year = df[df['world_rank'] <= 100]

# Prepare data for plotting
years = sorted(top_100_per_year['year'].unique())

# Expanded list of predictors
predictors = ['quality_of_education', 'alumni_employment', 'quality_of_faculty', 'publications', 'influence', 'citations', 'broad_impact', 'patents', ]
target = 'score'

# Create a subplot for each year
fig = make_subplots(rows=len(years), cols=1, subplot_titles=[f'Top 100 Universities in {year}' for year in years])

for i, year in enumerate(years, start=1):
    data_year = top_100_per_year[top_100_per_year['year'] == year].sort_values('world_rank', ascending=True)
    
    # Predict scores using the regression model
    predicted_scores, actual_scores = regressionModel(data_year, predictors, target)
    
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
    
    # Plot actual scores
    fig.add_trace(
        go.Bar(x=data_year['institution'], y=data_year['score'], name=f'Actual Scores {year}', marker=dict(color=colors)),
        row=i, col=1
    )
    
    # Plot predicted scores
    fig.add_trace(
        go.Scatter(x=data_year['institution'], y=predicted_scores, mode='markers', name=f'Predicted Scores {year}', marker=dict(color='black')),
        row=i, col=1
    )

# Update layout
fig.update_layout(height=3000, width=1000, title_text="Top 100 University Rankings from 2012 to 2015 with Predictions", showlegend=True)
fig.update_xaxes(tickangle=45)
fig.show()

# Train individual regression models for each predictor
results = train_individual_regressions_combined_plot(df, predictors, target)

# Print results
for predictor, info in results.items():
    print(f"Predictor: {predictor}, MSE: {info['mse']}")

def train_individual_regressions_plotly(df, predictors, target):
    # Ensure the target column has no missing values
    df = df.dropna(subset=[target])
    
    max_score = df[target].max() + 1
    df[target] = max_score - df[target]
    
    for predictor in predictors:
        # Prepare the DataFrame for the current predictor
        data = df[[predictor, target, 'institution']].dropna()  # Include 'institution' for labeling
        
        # Check if DataFrame is empty after dropping missing values
        if data.empty:
            print(f"No data available for predictor {predictor}")
            continue
        
        # Split the data
        X = data[[predictor]]  # DataFrame format required by sklearn
        y = data[target]
        institutions = data['institution']  # Capture the institution names for annotations
        X_train, X_test, y_train, y_test, institutions_train, institutions_test = train_test_split(X, y, institutions, test_size=0.2, random_state=42)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'Predictor': X_test[predictor],
            'Actual Score': y_test,
            'Predicted Score': y_pred,
            'Institution': institutions_test
        })
        
        # Invert predictions back to original scale for plotting
        plot_df['Predicted Score'] = max_score - plot_df['Predicted Score']
        plot_df['Actual Score'] = max_score - plot_df['Actual Score']
        
        # Plot using Plotly
        fig = px.scatter(plot_df, x='Predictor', y='Actual Score', hover_data=['Institution', 'Predicted Score'], title=f'Regression Analysis for {predictor}')
        fig.add_scatter(x=plot_df['Predictor'], y=plot_df['Predicted Score'], mode='lines', name='Regression Line')
        fig.update_traces(textposition='top center')
        fig.show()

# Example usage
df = pd.read_csv('cwurData.csv')
predictors = ['quality_of_education', 'alumni_employment', 'quality_of_faculty', 'publications', 'influence', 'citations', 'broad_impact', 'patents']
target = 'score'
train_individual_regressions_plotly(df, predictors, target)