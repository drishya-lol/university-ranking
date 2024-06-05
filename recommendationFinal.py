import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def load_data(filepath):
    return pd.read_csv(filepath)

def prepare_data(df, predictors, target):
    # Invert the target scores
    max_score = df[target].max() + 1
    df[target] = max_score - df[target]

    imputer = SimpleImputer(strategy='mean')
    df[predictors] = imputer.fit_transform(df[predictors])
    return df.dropna(subset=[target] + predictors)

def train_model(df, predictors, target):
    X = df[predictors]
    y = df[target]
    model = LinearRegression()
    model.fit(X, y)
    return model

def recommend_universities(model, student_profile, df, predictors, target):
    student_df = pd.DataFrame([student_profile], columns=predictors)
    predicted_score = model.predict(student_df)
    # Invert predicted score back to original scale for comparison
    max_score = df[target].max() + 1
    predicted_score = max_score - predicted_score

    df['predicted_difference'] = abs(df[target] - predicted_score[0])
    recommended_df = df.sort_values(by='predicted_difference')
    return recommended_df[['institution', 'predicted_difference']]

# Example usage
filepath = 'cwurData.csv'
df = load_data(filepath)
predictors = ['quality_of_education', 'alumni_employment', 'quality_of_faculty', 'publications', 'influence', 'citations']
target = 'score'

df = prepare_data(df, predictors, target)
model = train_model(df, predictors, target)

# Student profile example: student excels in education quality and faculty quality
student_profile = {'quality_of_education': 90, 'alumni_employment': 100, 'quality_of_faculty': 100, 'publications': 100, 'influence': 100, 'citations': 100}

recommended_universities = recommend_universities(model, student_profile, df, predictors, target)
print(recommended_universities.head())