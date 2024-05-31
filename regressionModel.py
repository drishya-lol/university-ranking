import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px

def regressionModel(file_path, predictors, target, year):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Filter data for the specified year
    df = df[df['year'] == year]
    
    # Selecting the predictors and target
    X = df[predictors]
    y = df[target]
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creating and training the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicting the target for test data
    y_pred = model.predict(X_test)
    
    # Calculating the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    
    # Plotting actual vs predicted values
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Score', 'y': 'Predicted Score'}, title=f'Actual vs Predicted Score for {year}')
    fig.show()
    
    # Returning the model and MSE as a tuple
    return model, mse

# Example usage
file_path = 'cwurData.csv'
predictors = ['quality_of_education', 'alumni_employment', 'quality_of_faculty', 'publications', 'influence', 'citations']
target = 'score'

# Creating models for each year from 2012 to 2015
for year in range(2012, 2016):
    model, mse = regressionModel(file_path, predictors, target, year)
    print(f"Model for {year} trained. Mean Squared Error: {mse}")
