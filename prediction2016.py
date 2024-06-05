import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data with the correct delimiter
data = pd.read_csv('cwurData.csv', delimiter=',')

print("Column names:", data.columns)

if 'year' in data.columns:
    # Filter data for the years 2012 to 2015
    data = data[data['year'].isin([2012, 2013, 2014, 2015])]
    print("Data filtered successfully.")
else:
    print("Column 'year' not found. Please check the data file and column names.")

# Prepare the data for modeling
# Pivot the data to have years as columns and ranks as values
pivot_data = data.pivot_table(index='institution', columns='year', values='world_rank')

# Drop any rows with NaN values to simplify the example
pivot_data.dropna(inplace=True)

# Train a linear regression model
X = pivot_data[[2012, 2013, 2014]].values
y = pivot_data[2015].values
model = LinearRegression()
model.fit(X, y)

# Predict the 2016 rankings
X_2016 = pivot_data[[2013, 2014, 2015]].values
predictions_2016 = model.predict(X_2016)

# Create a DataFrame for the 2016 predictions
predicted_ranks = pd.DataFrame({
    'institution': pivot_data.index,
    'predicted_rank_2016': np.floor(predictions_2016)
})

# Sort by predicted rank and get the top 15 universities
top_15_predictions = predicted_ranks.sort_values(by='predicted_rank_2016').head(15)

# Print the DataFrame without the index
print(top_15_predictions[['institution', 'predicted_rank_2016']].to_string(index=False))
