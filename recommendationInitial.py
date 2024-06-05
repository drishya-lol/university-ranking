import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def load_data(filepath):
    return pd.read_csv(filepath)

def prepare_data(df, predictors):
    imputer = SimpleImputer(strategy='mean')
    df[predictors] = imputer.fit_transform(df[predictors])
    return df.dropna(subset=predictors)

def train_model(df, predictors, target):
    X = df[predictors]
    y = df[target]
    model = LinearRegression()
    model.fit(X, y)
    return model

def recommend_universities(model, student_profile, df, predictors):
    student_df = pd.DataFrame([student_profile], columns=predictors)
    predicted_score = model.predict(student_df)
    df['predicted_difference'] = abs(df['score'] - predicted_score[0])
    recommended_df = df.sort_values(by='predicted_difference')
    return recommended_df[['institution', 'predicted_difference']]

# Example usage
filepath = 'cwurData.csv'
df = load_data(filepath)
predictors = ['quality_of_education', 'alumni_employment', 'quality_of_faculty', 'publications', 'influence', 'citations']
target = 'score'

df = prepare_data(df, predictors)
model = train_model(df, predictors, target)

# Student profile example: student excels in education quality and faculty quality
student_profile = {'quality_of_education': 100, 'alumni_employment': 75, 'quality_of_faculty': 25, 'publications': 100, 'influence': 50, 'citations': 30}

recommended_universities = recommend_universities(model, student_profile, df, predictors)
print(recommended_universities.head())