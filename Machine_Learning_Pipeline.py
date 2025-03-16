#!/usr/bin/env python
# coding: utf-8

# 
# # Machine Learning Pipeline for Mycotoxin Levels Prediction
# 
# This notebook demonstrates a pipeline for predicting DON concentration in corn samples based on hyperspectral data. The pipeline covers data exploration, preprocessing, model training, evaluation, and production readiness.
# 
# ## Dataset Description
# - **Features**: Spectral reflectance values across multiple wavelengths.
# - **Target**: DON concentration (continuous numerical value).
# 
# ## Objectives
# 1. Preprocess the data (handle missing values, normalize features, and explore potential anomalies).
# 2. Visualize the spectral bands to understand data characteristics.
# 3. Train a regression model (e.g., a neural network) to predict DON concentration.
# 4. Evaluate the model using robust metrics and visualization tools.
# 5. Develop a production-ready pipeline.
# 

# In[1]:


# Let's load the dataset and inspect it to identify missing values, outliers, and inconsistencies.

import pandas as pd

# Load the dataset
file_path = 'MLE-Assignment.csv'
data = pd.read_csv(file_path)

# Display basic information and the first few rows of the dataset
data_info = data.info()
data_head = data.head()

# Check for missing values
missing_values = data.isnull().sum()

data_info, data_head, missing_values


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

# Exclude the 'hsi_id' column for numerical analysis
numerical_data = data.drop(columns=['hsi_id'])

# Summary statistics of the dataset
summary_statistics = numerical_data.describe()

# Visualizing distributions with histograms
plt.figure(figsize=(20, 10))
numerical_data.hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.show()

# Visualizing outliers with boxplots for a subset of columns (to avoid overplotting)
plt.figure(figsize=(15, 8))
sns.boxplot(data=numerical_data.iloc[:, -10:])  # Visualizing the last 10 features as an example
plt.xticks(rotation=90)
plt.title('Boxplot of the last 10 spectral bands')
plt.tight_layout()
plt.show()

summary_statistics


# In[3]:


# Generate summary statistics for the entire dataset
summary_statistics = numerical_data.describe()

# Visualizing distributions with histograms (subset of 10 columns to avoid overplotting)
plt.figure(figsize=(15, 8))
numerical_data.iloc[:, :10].hist(bins=50, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Visualizing outliers with boxplots for a subset of 10 columns
plt.figure(figsize=(15, 8))
sns.boxplot(data=numerical_data.iloc[:, :10])
plt.xticks(rotation=90)
plt.title('Boxplot of the first 10 spectral bands')
plt.tight_layout()
plt.show()

summary_statistics


# In[4]:


# Re-loading the dataset after environment reset to check for missing values and continue preprocessing.

import pandas as pd

# Load the dataset
file_path = 'MLE-Assignment.csv'
data = pd.read_csv(file_path)

# Exclude the 'hsi_id' column for numerical analysis
numerical_data = data.drop(columns=['hsi_id'])

# Check for missing values
missing_values = numerical_data.isnull().sum()

# No missing values should be present based on earlier exploration
missing_values


# In[5]:


from sklearn.preprocessing import StandardScaler

# Separating the target variable (vomitoxin_ppb) from the spectral data
X = numerical_data.drop(columns=['vomitoxin_ppb'])  # Features (spectral reflectance values)
y = numerical_data['vomitoxin_ppb']  # Target variable (DON concentration)

# Standardizing the spectral data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Creating a new DataFrame with normalized data
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Checking the summary statistics after scaling
scaled_summary_statistics = X_scaled_df.describe()

scaled_summary_statistics



# In[6]:


from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

# Detect anomalies using z-score (flagging samples with z-scores > 3 or < -3)
z_scores = X_scaled_df.apply(zscore)
z_score_anomalies = (z_scores > 3) | (z_scores < -3)
z_score_anomalous_samples = z_score_anomalies.any(axis=1)

# Detect anomalies using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)  # Assume 5% contamination rate
iso_forest.fit(X_scaled_df)
iso_forest_anomalies = iso_forest.predict(X_scaled_df)  # -1 for anomalies, 1 for normal data

# Summary of anomaly detection
z_score_anomalies_summary = z_score_anomalous_samples.sum()  # Total number of anomalies from z-score
iso_forest_anomalies_summary = (iso_forest_anomalies == -1).sum()  # Total number of anomalies from Isolation Forest

z_score_anomalies_summary, iso_forest_anomalies_summary



# In[7]:


# Re-import the necessary libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Generate a line plot for the average reflectance over wavelengths
avg_reflectance = X_scaled_df.mean()

# Line plot of average reflectance values across wavelengths
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(avg_reflectance)), avg_reflectance, color='blue', marker='o')
plt.title('Average Reflectance Across Wavelengths')
plt.xlabel('Wavelength Index')
plt.ylabel('Average Reflectance (Standardized)')
plt.grid(True)
plt.show()

# Generate a heatmap for sample comparisons
plt.figure(figsize=(12, 8))
sns.heatmap(X_scaled_df.corr(), cmap='coolwarm', annot=False)
plt.title('Heatmap of Correlations Between Wavelengths')
plt.show()


# In[8]:


from scipy.stats import kstest

# Automating checks for sensor drift using rolling mean and variance
window_size = 50  # Assuming 50 samples for rolling analysis

# Calculate rolling mean and variance for each wavelength (column)
rolling_mean = X_scaled_df.rolling(window=window_size).mean()
rolling_variance = X_scaled_df.rolling(window=window_size).var()

# Detect sensor drift using the Kolmogorov-Smirnov test (comparing first vs last window)
ks_results = {}
for col in X_scaled_df.columns:
    first_window = X_scaled_df[col][:window_size]
    last_window = X_scaled_df[col][-window_size:]
    ks_stat, p_value = kstest(first_window, last_window)
    ks_results[col] = p_value

# Flag wavelengths with significant drift (p-value < 0.05)
drift_flags = {col: p for col, p in ks_results.items() if p < 0.05}

# Summarizing drift checks
rolling_mean.head(), rolling_variance.head(), drift_flags


# In[9]:


from sklearn.decomposition import PCA

# Create a ratio-based spectral index (NDVI-like)
# For simplicity, we'll use the first and last wavelength bands to create a spectral index.
X_scaled_df['spectral_index_1'] = (X_scaled_df.iloc[:, 0] - X_scaled_df.iloc[:, -1]) / (X_scaled_df.iloc[:, 0] + X_scaled_df.iloc[:, -1])

# Apply PCA to reduce dimensionality and capture the most variance in the spectral data
pca = PCA(n_components=5)  # Reduce to 5 principal components
pca_features = pca.fit_transform(X_scaled_df)

# Add PCA components to the dataframe as new features
for i in range(pca_features.shape[1]):
    X_scaled_df[f'pca_{i+1}'] = pca_features[:, i]

# Check the explained variance of the PCA components
explained_variance = pca.explained_variance_ratio_

X_scaled_df.head(), explained_variance


# In[10]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load your dataset
data = pd.read_csv('MLE-Assignment.csv')  # Replace with your actual dataset path

# Assuming 'DON_concentration' is the target column and all other columns are features
X = data.drop(columns='DON_concentration')
y = data['DON_concentration']

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Input and first hidden layer
    Dense(32, activation='relu'),  # Second hidden layer
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=1)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss (MSE): {test_loss}')


# In[ ]:





# In[11]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create the XGBoost model
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)

# Train the model
xgboost_model.fit(X_train, y_train)

# Make predictions
y_pred = xgboost_model.predict(X_test)

# Evaluate the model using mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error for XGBoost: {mse}")


# In[12]:


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Base models
base_models = [
    ('ridge', RidgeCV()),
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
]

# Stacking model with a meta-model (linear regression)
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=RidgeCV()
)

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Make predictions
y_pred_stack = stacking_model.predict(X_test)

# Evaluate the model using mean squared error (MSE)
mse_stack = mean_squared_error(y_test, y_pred_stack)
print(f"Mean Squared Error for Stacking Model: {mse_stack}")


# In[13]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Custom neural network architecture
def create_custom_nn():
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Create the model
custom_nn = create_custom_nn()

# Train the model
custom_nn.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Make predictions
y_pred_nn = custom_nn.predict(X_test)

# Evaluate the model using mean squared error (MSE)
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f"Mean Squared Error for Custom Neural Network: {mse_nn}")


# In[16]:


from sklearn.model_selection import train_test_split

# Assuming 'X' contains your features and 'y' is the target variable (DON concentration)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[17]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the number of folds
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize your model (e.g., XGBoost or any regression model)
model = xgb.XGBRegressor()

# Cross-validation process
mse_scores = []
for train_index, test_index in kf.split(X_scaled):
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    
    # Train the model on the training fold
    model.fit(X_train_fold, y_train_fold)
    
    # Predict on the test fold
    y_pred_fold = model.predict(X_test_fold)
    
    # Compute the mean squared error for this fold
    mse_fold = mean_squared_error(y_test_fold, y_pred_fold)
    mse_scores.append(mse_fold)

# Average MSE score across all folds
mean_mse = np.mean(mse_scores)
print(f"Mean MSE across {k}-folds: {mean_mse}")


# In[18]:


import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint
import xgboost as xgb

# Load your dataset
data = pd.read_csv("MLE-Assignment.csv")

# Print the column names to identify the target column
print("Column names in the dataset:", data.columns)

# Define X (features) and y (target)
X = data.drop('vomitoxin_ppb', axis=1)   
y = data['vomitoxin_ppb']  

# Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = xgb.XGBRegressor()

# Define the hyperparameter search space
param_dist = {
    'n_estimators': randint(50, 300),  # Number of trees
    'learning_rate': uniform(0.01, 0.2),  # Learning rate
    'max_depth': randint(3, 10),  # Depth of the trees
    'subsample': uniform(0.5, 1.0),  # Percentage of rows used per tree
    'colsample_bytree': uniform(0.5, 1.0)  # Percentage of features used per tree
}

# Perform RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    model, 
    param_distributions=param_dist, 
    n_iter=100,  # Number of parameter settings sampled
    scoring='neg_mean_squared_error',  # Evaluation metric
    cv=3,  # Number of folds in cross-validation
    verbose=1,  # For displaying progress
    random_state=42,  # For reproducibility
    n_jobs=-1  # Use all available cores
)

# Fit the model
random_search.fit(X_train, y_train)

# Best hyperparameters
print("Best hyperparameters:", random_search.best_params_)

# Evaluate the model on the test set
y_pred = random_search.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse}")


# In[ ]:


import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Sample data (replace with your actual DataFrame)
data = pd.DataFrame({
    'hsi_id': ['imagoai_corn_0', 'imagoai_corn_1', 'imagoai_corn_2', 'imagoai_corn_3', 'imagoai_corn_4'],
    '0': [0.416181, 0.415797, 0.389023, 0.468837, 0.483352],
    '1': [0.396844, 0.402956, 0.371206, 0.473255, 0.487274],
    # add all your columns here...
    'vomitoxin_ppb': [1100.0, 1000.0, 1300.0, 1300.0, 220.0]
})

# Drop categorical 'hsi_id' column
X = data.drop(columns=['hsi_id', 'vomitoxin_ppb'])  # Input features (drop hsi_id and target)
y = data['vomitoxin_ppb']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Objective function for Optuna
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',  # Set tree method to 'hist' for better performance
    }

    model = xgb.XGBRegressor(**param)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Create a study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Best parameters found
print("Best parameters: ", study.best_params)


# In[19]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Generate sample data
np.random.seed(42)
X = np.random.rand(1000, 10)  # 1000 samples, 10 features
y = np.random.rand(1000) * 100  # Target variable between 0 and 100

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a custom asymmetric loss function
def custom_asymmetric_loss(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual > 0, 1 * residual, 10 * np.abs(residual))
    return np.sum(loss)

# Define a custom objective function for XGBoost
def custom_asymmetric_obj(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual > 0, -1, 10)
    hess = np.ones_like(y_true)  # Hessian is constant (approximated as 1 here)
    return grad, hess

# Initialize the XGBoost Regressor
model = XGBRegressor(objective=custom_asymmetric_obj, n_estimators=100, learning_rate=0.1, max_depth=3)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Separate the features (X) and target (y)
X = data.drop(columns=['hsi_id', 'vomitoxin_ppb'])
y = data['vomitoxin_ppb']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
mae


# In[21]:


# Calculating the R² Score
from sklearn.metrics import r2_score

# Calculate R² score
r2 = r2_score(y_test, y_pred)
r2



# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create scatter plot of actual vs predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linewidth=2)  # Line representing perfect predictions
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values (Scatter Plot)")
plt.grid(True)
plt.show()


# In[23]:


# Perform residual analysis
# Residuals are the difference between actual and predicted values
residuals = y_test - y_pred

# Plotting residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color="blue", edgecolor="w", alpha=0.7)
plt.axhline(y=0, color="red", linestyle="--")
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()


# In[25]:


import shap

# Create a SHAP explainer for the trained model (using TreeExplainer as it's well-suited for tree-based models)
explainer = shap.Explainer(model)

# Get SHAP values for the test data
shap_values = explainer(X_test)

# Global feature importance plot
shap.summary_plot(shap_values, X_test)


# In[26]:


import shap
import matplotlib.pyplot as plt
import numpy as np

# Train the model (assuming the model has already been trained)
# Let's say 'model' is your trained model and 'X_test' is your test data

# Create a SHAP explainer
explainer = shap.Explainer(model)

# Get SHAP values for the test set
shap_values = explainer(X_test)

# Summary plot for global feature importance
shap.summary_plot(shap_values, X_test)

# Feature importance based on mean absolute SHAP values
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Limitations and insights
print("\nModel Limitations:")
print("1. Bias: If training data is biased, predictions may not generalize well.")
print("2. Feature Correlation: The model may struggle with highly correlated features.")
print("3. Overfitting: High accuracy on training data but poor test performance may indicate overfitting.")
print("4. Interpretability: Despite using SHAP, the model might still be hard to fully interpret.")
print("5. Outliers and Variance: The model might be sensitive to outliers or data with high variance.")
print("6. Generalization: Test the model on entirely new data to assess its robustness.")



# In[36]:


import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        # Sample data setup
        self.df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [2, 4, 6, 8],
            'target': [1, 2, 3, 4]
        })
        self.X = self.df[['feature1', 'feature2']]
        self.y = self.df['target']
        self.model = RandomForestRegressor()

    def test_train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        self.assertGreaterEqual(score, 0, "Model score should be non-negative")

    def test_predict(self):
        self.model.fit(self.X, self.y)
        preds = self.model.predict(self.X)
        self.assertEqual(len(preds), len(self.y), "Predictions should match the number of input rows")
   
    


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[37]:


import logging

# Step 1: Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Log all levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp, log level, and message
    handlers=[
        logging.FileHandler("ml_pipeline.log"),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)

# Example function: Train model with logging
def train_model():
    try:
        # Log the start of model training
        logging.info("Starting model training...")

        # Assume you have some code here that loads data
        # For example:
        X_train, X_test, y_train, y_test = load_data()
        logging.debug(f"Data loaded: {len(X_train)} training samples, {len(X_test)} test samples")

        # Train the model (replace with your actual training logic)
        model = SomeModelClass()
        model.fit(X_train, y_train)

        logging.info("Model training completed.")
        return model, X_test, y_test

    except Exception as e:
        logging.error(f"Error during model training: {e}", exc_info=True)
        raise

# Example unit test with logging
import unittest

class TestMLPipeline(unittest.TestCase):
    def test_train_model(self):
        logging.info("Starting test for model training")

        try:
            model, X_test, y_test = train_model()

            # Check if there's enough data for evaluation
            if len(y_test) < 2:
                logging.warning("Not enough data to calculate R^2 score, skipping test")
                self.skipTest("Not enough data to calculate R^2 score")

            # Evaluate the model
            score = model.score(X_test, y_test)
            logging.debug(f"Model score: {score}")

            # Assert the score is non-negative
            self.assertGreaterEqual(score, 0, "Model score should be non-negative")
            logging.info("Model test passed with a valid score.")

        except Exception as e:
            logging.error(f"Test failed: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    unittest.main()



# In[ ]:





# In[ ]:




