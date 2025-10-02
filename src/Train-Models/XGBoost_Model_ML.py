import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Updated dataset name to match Create_Games.py output
dataset = "dataset_nfl_new"
con = sqlite3.connect("../../Data/dataset.sqlite")

try:
    data = pd.read_sql_query(f"SELECT * FROM `{dataset}`", con)
    print(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Available tables:")
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)
    print(tables)
    con.close()
    exit(1)

con.close()

# Extract target variable
margin = data['Home-Team-Win']

# Define columns to exclude from features
exclude_columns = [
    'Score', 'Home-Team-Win', 'TEAM_NAME', 'TEAM_NAME.1', 
    'OU-Cover', 'OU', 'Days-Rest-Home', 'Days-Rest-Away',
    'index', 'TEAM_ID', 'TEAM_ID.1', 'SEASON', 'SEASON.1'
]

# Remove non-feature columns
feature_columns = [col for col in data.columns if col not in exclude_columns]
data_features = data[feature_columns]

print(f"Using {len(feature_columns)} features for training")
print(f"Feature columns: {feature_columns[:10]}...")  # Show first 10 features

# Convert features to numpy array and ensure proper data types
data = data_features.values

# Handle any non-numeric values by converting to numeric and filling NaN
data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').fillna(0).values
data = data.astype(float)

print(f"Data shape: {data.shape}")
print(f"Number of features: {data.shape[1]}")
print(f"Target distribution: {np.bincount(margin)}")
print(f"Feature columns used: {feature_columns}")

# Check for any remaining issues
if np.isnan(data).any():
    print("Warning: NaN values found in data, replacing with 0")
    data = np.nan_to_num(data)

if np.isinf(data).any():
    print("Warning: Infinite values found in data, replacing with 0")
    data = np.nan_to_num(data, posinf=0, neginf=0)

acc_results = []
best_accuracy = 0

print("Starting XGBoost training...")
for x in tqdm(range(300), desc="Training iterations"):
    try:
        x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=0.1, random_state=x)
        
        # Ensure we have both classes in training data
        if len(np.unique(y_train)) < 2:
            print(f"Iteration {x}: Skipping - only one class in training data")
            continue

        train = xgb.DMatrix(x_train, label=y_train)
        test = xgb.DMatrix(x_test, label=y_test)

        # Updated parameters for better performance
        param = {
            'max_depth': 4,  # Increased depth for more complex patterns
            'eta': 0.05,     # Increased learning rate
            'objective': 'multi:softprob',
            'num_class': 2,
            'eval_metric': 'mlogloss',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': x
        }
        epochs = 500  # Reduced epochs due to higher learning rate

        model = xgb.train(param, train, epochs, verbose_eval=False)
        predictions = model.predict(test)
        
        # Convert predictions to class labels
        y_pred = []
        for z in predictions:
            y_pred.append(np.argmax(z))

        acc = round(accuracy_score(y_test, y_pred) * 100, 1)
        
        # Only print every 50 iterations to reduce output
        if x % 50 == 0 or acc > best_accuracy:
            print(f"Iteration {x}: {acc}%")
        
        acc_results.append(acc)
        
        # Save model if it's the best so far
        if acc > best_accuracy:
            best_accuracy = acc
            # Ensure the directory exists
            import os
            model_dir = '../../Models/XGBoost_Models'
            os.makedirs(model_dir, exist_ok=True)
            model_path = f'{model_dir}/XGBoost_{acc}%_ML.json'
            model.save_model(model_path)
            print(f"New best model saved: {model_path}")
            
    except Exception as e:
        print(f"Iteration {x}: Error - {e}")
        continue

# Print final results
print(f"\nTraining completed!")
print(f"Best accuracy: {max(acc_results):.1f}%")
print(f"Average accuracy: {np.mean(acc_results):.1f}%")
print(f"Standard deviation: {np.std(acc_results):.1f}%")
print(f"Accuracy range: {min(acc_results):.1f}% - {max(acc_results):.1f}%")
