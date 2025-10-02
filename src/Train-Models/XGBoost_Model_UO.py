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

# Extract target variables for Over/Under prediction
OU = data['OU-Cover']  # 0 = under, 1 = over, 2 = push
total = data['OU']     # Over/Under line value

print(f"OU-Cover distribution: {np.bincount(OU)}")
print(f"OU values range: {total.min():.1f} - {total.max():.1f}")

# Define columns to exclude from features
exclude_columns = [
    'Score', 'Home-Team-Win', 'TEAM_NAME', 'TEAM_NAME.1', 
    'OU-Cover', 'Days-Rest-Home', 'Days-Rest-Away',
    'index', 'TEAM_ID', 'TEAM_ID.1', 'SEASON', 'SEASON.1'
]

# Remove non-feature columns but keep OU as a feature
feature_columns = [col for col in data.columns if col not in exclude_columns]
data_features = data[feature_columns]

print(f"Using {len(feature_columns)} features for OU prediction")
print(f"Feature columns: {feature_columns[:10]}...")  # Show first 10 features

# Convert features to numpy array and ensure proper data types
data = data_features.values

# Handle any non-numeric values by converting to numeric and filling NaN
data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').fillna(0).values
data = data.astype(float)

print(f"Data shape: {data.shape}")

# Check for any remaining issues
if np.isnan(data).any():
    print("Warning: NaN values found in data, replacing with 0")
    data = np.nan_to_num(data)

if np.isinf(data).any():
    print("Warning: Infinite values found in data, replacing with 0")
    data = np.nan_to_num(data, posinf=0, neginf=0)

acc_results = []
best_accuracy = 0

print("Starting XGBoost OU prediction training...")
for x in tqdm(range(100), desc="Training iterations"):
    try:
        x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=0.1, random_state=x)
        
        # Ensure we have all three classes in training data (under, over, push)
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 3:
            print(f"Iteration {x}: Skipping - only {len(unique_classes)} classes in training data")
            continue

        train = xgb.DMatrix(x_train, label=y_train)
        test = xgb.DMatrix(x_test)

        # Updated parameters for OU prediction (3-class problem)
        param = {
            'max_depth': 6,  # Moderate depth for OU prediction
            'eta': 0.1,      # Higher learning rate for faster convergence
            'objective': 'multi:softprob',
            'num_class': 3,  # Under, Over, Push
            'eval_metric': 'mlogloss',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': x
        }
        epochs = 300  # Reduced epochs due to higher learning rate

        model = xgb.train(param, train, epochs, verbose_eval=False)

        predictions = model.predict(test)
        
        # Convert predictions to class labels
        y_pred = []
        for z in predictions:
            y_pred.append(np.argmax(z))

        acc = round(accuracy_score(y_test, y_pred) * 100, 1)
        
        # Only print every 20 iterations to reduce output
        if x % 20 == 0 or acc > best_accuracy:
            print(f"Iteration {x}: {acc}%")
        
        acc_results.append(acc)
        
        # Save model if it's the best so far
        if acc > best_accuracy:
            best_accuracy = acc
            # Ensure the directory exists
            import os
            model_dir = '../../Models/XGBoost_Models'
            os.makedirs(model_dir, exist_ok=True)
            model_path = f'{model_dir}/XGBoost_{acc}%_UO.json'
            model.save_model(model_path)
            print(f"New best OU model saved: {model_path}")
            
    except Exception as e:
        print(f"Iteration {x}: Error - {e}")
        continue

# Print final results
print(f"\nOU Training completed!")
print(f"Best accuracy: {max(acc_results):.1f}%")
print(f"Average accuracy: {np.mean(acc_results):.1f}%")
print(f"Standard deviation: {np.std(acc_results):.1f}%")
print(f"Accuracy range: {min(acc_results):.1f}% - {max(acc_results):.1f}%")

# Print class distribution analysis
print(f"\nClass distribution analysis:")
print(f"Under (0): {np.sum(OU == 0)} games ({np.sum(OU == 0)/len(OU)*100:.1f}%)")
print(f"Over (1): {np.sum(OU == 1)} games ({np.sum(OU == 1)/len(OU)*100:.1f}%)")
print(f"Push (2): {np.sum(OU == 2)} games ({np.sum(OU == 2)/len(OU)*100:.1f}%)")
