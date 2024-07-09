import pandas as pd
import numpy as np
import warnings 
import pickle
warnings.filterwarnings("ignore")

# Reading the dataset
bitcoin_df = pd.read_csv(r"C:\sudhanshu_projects\project-task-training-course\bitcoin_price_forecasting.csv")

# Splitting the date column into year, month, and day columns
split = bitcoin_df["Date"].str.split("-", expand=True)
bitcoin_df["Year"] = split[0].astype("int")
bitcoin_df["Month"] = split[1].astype("int")
bitcoin_df["Day"] = split[2].astype("int")

# Dropping the date column
bitcoin_df.drop("Date", axis=1, inplace=True)

# Creating the target column
bitcoin_df["Target"] = bitcoin_df["Close"].shift(-1)

# Dropping the last row because it contains NaN value
bitcoin_df.dropna(inplace=True)

# Creating additional features
bitcoin_df['is_quarter_end'] = np.where(bitcoin_df['Month'] % 3 == 0, 1, 0)
bitcoin_df["open-close"] = bitcoin_df["Open"] - bitcoin_df["Close"]
bitcoin_df["low-high"] = bitcoin_df["Low"] - bitcoin_df["High"]
bitcoin_df["price_diff"] = bitcoin_df["Close"].diff()
bitcoin_df["volatility"] = (bitcoin_df["High"] - bitcoin_df["Low"]) / bitcoin_df["Close"]
bitcoin_df["daily_return"] = bitcoin_df["Close"].pct_change()
bitcoin_df["ma7"] = bitcoin_df["Close"].rolling(window=7).mean()
bitcoin_df["ma21"] = bitcoin_df["Close"].rolling(window=21).mean()
bitcoin_df["ma63"] = bitcoin_df["Close"].rolling(window=63).mean()

# Dropping rows with NaN values created by rolling functions
bitcoin_df.dropna(inplace=True)

# Defining the feature set and target variable
x = bitcoin_df[["open-close", "low-high", "is_quarter_end", "price_diff", "volatility", "daily_return", "ma7", "ma21", "ma63"]]
y = bitcoin_df["Target"]

# Splitting the dataset into train and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Initialize the model
gb = GradientBoostingRegressor(random_state=0)

# Initialize Grid Search
grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(x_train_sc, y_train)

# Get the best model
best_gb = grid_search.best_estimator_

# Predict using the best model
y_pred_gb = best_gb.predict(x_test_sc)

# Calculate the accuracy
from sklearn.metrics import r2_score
score_gb = r2_score(y_test, y_pred_gb)
print(f"The accuracy of the best Gradient Boosting model is {score_gb}.")

# Try XGBoost
from xgboost import XGBRegressor

# Define the parameter grid
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Initialize the model
xgb = XGBRegressor(random_state=0)

# Initialize Grid Search
grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=2)

# Fit the model
grid_search_xgb.fit(x_train_sc, y_train)

# Get the best model
best_xgb = grid_search_xgb.best_estimator_

# Predict using the best model
y_pred_xgb = best_xgb.predict(x_test_sc)

# Calculate the accuracy
score_xgb = r2_score(y_test, y_pred_xgb)
print(f"The accuracy of the best XGBoost model is {score_xgb}.")

#-------------------------------------------
#Here we save the model.

pickle.dump(grid_search,open(r"C:\sudhanshu_projects\project-task-training-course\bitcoin_price_forecasting.pkl","wb"))

#-------------------------------------------
#Now we load the model.

model=pickle.load(open(r"C:\sudhanshu_projects\project-task-training-course\bitcoin_price_forecasting.pkl","rb"))

