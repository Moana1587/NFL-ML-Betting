import copy
import glob
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

init()

def load_best_model(model_type):
    """Load the best available model for the given type (ML or UO)"""
    # Try multiple possible model directory locations
    possible_dirs = [
        'Models/XGBoost_Models/',
        '../../Models/XGBoost_Models/',
        '../Models/XGBoost_Models/',
        'src/Models/XGBoost_Models/'
    ]
    
    model_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            model_dir = dir_path
            break
    
    if model_dir is None:
        print(f"Warning: Model directory not found in any of these locations:")
        for dir_path in possible_dirs:
            print(f"  - {dir_path}")
        print("Please train models first by running:")
        print("  python src/Train-Models/XGBoost_Model_ML.py")
        print("  python src/Train-Models/XGBoost_Model_UO.py")
        return None
    
    # Find all models of the specified type
    pattern = f"XGBoost_*%_{model_type}.json"
    model_files = glob.glob(os.path.join(model_dir, pattern))
    
    if not model_files:
        print(f"Warning: No {model_type} models found in {model_dir}")
        return None
    
    # Sort by accuracy (extract from filename) and get the best one
    def extract_accuracy(filename):
        try:
            # Extract accuracy from filename like "XGBoost_68.7%_ML.json"
            basename = os.path.basename(filename)
            accuracy_str = basename.split('_')[1].replace('%', '')
            return float(accuracy_str)
        except:
            return 0.0
    
    best_model = max(model_files, key=extract_accuracy)
    accuracy = extract_accuracy(best_model)
    
    print(f"Loading best {model_type} model: {os.path.basename(best_model)} ({accuracy}%)")
    
    try:
        booster = xgb.Booster()
        booster.load_model(best_model)
        return booster
    except Exception as e:
        print(f"Error loading model {best_model}: {e}")
        return None

# Load the best available models
xgb_ml = load_best_model('ML')
xgb_uo = load_best_model('UO')

if xgb_ml is None or xgb_uo is None:
    print("Error: Could not load required models. Please train models first.")
    print("Run: python src/Train-Models/XGBoost_Model_ML.py")
    print("Run: python src/Train-Models/XGBoost_Model_UO.py")


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    """Run XGBoost predictions for today's games"""
    
    if xgb_ml is None or xgb_uo is None:
        print("Error: Models not loaded. Cannot make predictions.")
        return
    
    if data is None or len(data) == 0:
        print("Error: No data provided for predictions.")
        return
    
    print(f"Making predictions for {len(games)} games...")
    print(f"Data shape: {data.shape} (rows: {data.shape[0]}, features: {data.shape[1]})")
    
    # Make ML predictions
    ml_predictions_array = []
    try:
        for i, row in enumerate(data):
            if i == 0:  # Debug first row
                print(f"First row shape: {row.shape}, sample values: {row[:5]}")
            prediction = xgb_ml.predict(xgb.DMatrix(np.array([row])))
            ml_predictions_array.append(prediction)
    except Exception as e:
        print(f"Error making ML predictions: {e}")
        print(f"Data shape: {data.shape}")
        print(f"First row shape: {data[0].shape if len(data) > 0 else 'No data'}")
        print(f"Expected features: Check model training logs")
        return

    # Prepare OU data with same feature selection as ML data
    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    
    # Use the same feature selection logic as in training
    exclude_columns = [
        'Score', 'Home-Team-Win', 'TEAM_NAME', 'TEAM_NAME.1', 
        'OU-Cover', 'Days-Rest-Home', 'Days-Rest-Away',
        'index', 'TEAM_ID', 'TEAM_ID.1', 'SEASON', 'SEASON.1',
        'Date', 'Date.1', 'index.1'
    ]
    
    # Get feature columns (same as ML prediction)
    feature_columns = [col for col in frame_uo.columns if col not in exclude_columns]
    
    print(f"OU data - Using {len(feature_columns)} feature columns")
    print(f"OU feature columns: {feature_columns[:10]}...")
    
    # Create OU data with only feature columns
    ou_feature_data = frame_uo[feature_columns].values
    
    # Convert to float, handling any remaining non-numeric values
    try:
        ou_data = ou_feature_data.astype(float)
        print("Successfully converted OU data to float")
    except ValueError as e:
        print(f"Error converting OU data to float: {e}")
        print("Attempting fallback conversion method...")
        
        # Fallback: convert each column individually and handle errors
        ou_data_df = pd.DataFrame(ou_feature_data, columns=feature_columns)
        ou_data_df = ou_data_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        ou_data = ou_data_df.values.astype(float)
        print("Used fallback conversion method for OU data")

    # Make OU predictions
    print(f"OU data shape: {ou_data.shape} (rows: {ou_data.shape[0]}, features: {ou_data.shape[1]})")
    ou_predictions_array = []
    try:
        for i, row in enumerate(ou_data):
            if i == 0:  # Debug first row
                print(f"OU first row shape: {row.shape}, sample values: {row[:5]}")
            prediction = xgb_uo.predict(xgb.DMatrix(np.array([row])))
            ou_predictions_array.append(prediction)
    except Exception as e:
        print(f"Error making OU predictions: {e}")
        print(f"OU data shape: {ou_data.shape}")
        print(f"OU first row shape: {ou_data[0].shape if len(ou_data) > 0 else 'No data'}")
        return

    # Display predictions
    print("\n" + "="*60)
    print("XGBOOST PREDICTIONS")
    print("="*60)
    
    count = 0
    for game in games:
        if count >= len(ml_predictions_array) or count >= len(ou_predictions_array):
            print(f"Warning: Not enough predictions for game {count + 1}")
            break
            
        home_team = game[0]
        away_team = game[1]
        
        # Get predictions
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        un_confidence = ou_predictions_array[count]
        
        # Format confidence percentages with proper rounding
        if winner == 1:  # Home team wins
            winner_confidence = round(winner_confidence[0][1] * 100, 1)
            if under_over == 0:  # Under
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence:.1f}%)" + Style.RESET_ALL + 
                    ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(todays_games_uo[count]) + 
                    Style.RESET_ALL + Fore.CYAN + f" ({un_confidence:.1f}%)" + Style.RESET_ALL)
            else:  # Over
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence:.1f}%)" + Style.RESET_ALL + 
                    ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(todays_games_uo[count]) + 
                    Style.RESET_ALL + Fore.CYAN + f" ({un_confidence:.1f}%)" + Style.RESET_ALL)
        else:  # Away team wins
            winner_confidence = round(winner_confidence[0][0] * 100, 1)
            if under_over == 0:  # Under
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                print(
                    Fore.RED + home_team + Style.RESET_ALL + ' vs ' + 
                    Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence:.1f}%)" + Style.RESET_ALL + 
                    ': ' + Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(todays_games_uo[count]) + 
                    Style.RESET_ALL + Fore.CYAN + f" ({un_confidence:.1f}%)" + Style.RESET_ALL)
            else:  # Over
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                print(
                    Fore.RED + home_team + Style.RESET_ALL + ' vs ' + 
                    Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence:.1f}%)" + Style.RESET_ALL + 
                    ': ' + Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(todays_games_uo[count]) + 
                    Style.RESET_ALL + Fore.CYAN + f" ({un_confidence:.1f}%)" + Style.RESET_ALL)
        count += 1

    # Expected Value and Kelly Criterion Analysis
    if kelly_criterion:
        print("\n" + "="*60)
        print("EXPECTED VALUE & KELLY CRITERION ANALYSIS")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("EXPECTED VALUE ANALYSIS")
        print("="*60)
    
    count = 0
    for game in games:
        if count >= len(ml_predictions_array):
            print(f"Warning: Not enough predictions for EV analysis of game {count + 1}")
            break
            
        home_team = game[0]
        away_team = game[1]
        
        # Calculate expected values
        ev_home = ev_away = 0
        if (count < len(home_team_odds) and count < len(away_team_odds) and 
            home_team_odds[count] and away_team_odds[count]):
            try:
                ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
                ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
            except Exception as e:
                print(f"Error calculating EV for {home_team} vs {away_team}: {e}")
                ev_home = ev_away = 0
        
        # Color coding for expected values
        expected_value_colors = {
            'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
            'away_color': Fore.GREEN if ev_away > 0 else Fore.RED
        }
        
        # Kelly Criterion calculations
        if kelly_criterion:
            try:
                bankroll_descriptor = ' Fraction of Bankroll: '
                bankroll_fraction_home = bankroll_descriptor + str(kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1])) + '%'
                bankroll_fraction_away = bankroll_descriptor + str(kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0])) + '%'
            except Exception as e:
                print(f"Error calculating Kelly Criterion: {e}")
                bankroll_fraction_home = bankroll_fraction_away = ' Error calculating Kelly %'
        else:
            bankroll_fraction_home = bankroll_fraction_away = ''

        # Display results
        print(f"\n{home_team} vs {away_team}:")
        print(f"  {home_team} EV: " + expected_value_colors['home_color'] + f"{ev_home:.3f}" + Style.RESET_ALL + (bankroll_fraction_home if kelly_criterion else ''))
        print(f"  {away_team} EV: " + expected_value_colors['away_color'] + f"{ev_away:.3f}" + Style.RESET_ALL + (bankroll_fraction_away if kelly_criterion else ''))
        
        # Add betting recommendation
        if ev_home > 0 or ev_away > 0:
            if ev_home > ev_away and ev_home > 0:
                print(f"  Recommendation: " + Fore.GREEN + f"Bet on {home_team}" + Style.RESET_ALL)
            elif ev_away > ev_home and ev_away > 0:
                print(f"  Recommendation: " + Fore.GREEN + f"Bet on {away_team}" + Style.RESET_ALL)
        else:
            print(f"  Recommendation: " + Fore.YELLOW + "No positive EV bets" + Style.RESET_ALL)
        
        count += 1

    print("\n" + "="*60)
    deinit()
