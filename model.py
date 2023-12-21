import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
import joblib as jb

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer as ms

# Importing the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
dt = {'mpg': float, 'cylinders': int, 'displacement': float, 'weight': float, 'acceleration': float, 'model year': int, 'origin': int}
df = pd.read_csv(url, sep="\s+", names=columns, dtype=dt)

# No missing values
# Normalizing the data using MinMaxScaler
sc = MinMaxScaler()
df[['mpg','weight','acceleration']] = sc.fit_transform(df[['mpg','weight','acceleration']])

# Categorical features
cf = ['model year', 'origin','cylinders']

# Numerical features
nf = ['displacement','weight','acceleration']

# Target feature
tf = 'mpg'

# Split data into training and test sets
x, y = df[['cylinders','acceleration','origin','model year','weight','displacement']].values, df['mpg'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Regression algorithm
alg = RandomForestRegressor()

# Define preprocessing for numeric columns (scale them)
numeric_features = [2,4,5]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Define preprocessing for categorical features (encode them)
categorical_features = [1,3,4]
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', alg)])

# fit the pipeline to train a linear regression model on the training set
model = pipeline.fit(x_train, y_train)

# Evaluate the predictions on the test set
predicted = model.predict(x_test)
mse = mean_squared_error(y_test,predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,predicted)
print('Mean squared error: ', mse, '\nRoot mean squared error: ', rmse, '\nR-squared: ', r2)

# Save the model as a pickle file
filename = './assets/model.pkl'
jb.dump(model, filename)

# Graph of predicted vs actual
mp.scatter(y_test, predicted)
mp.xlabel('Actual')
mp.ylabel('Predicted')
mp.title('Actual vs Predicted')

# Overlay the regression line
z = np.polyfit(y_test, predicted, 1)
p = np.poly1d(z)
mp.plot(y_test,p(y_test),"r--")
mp.show()