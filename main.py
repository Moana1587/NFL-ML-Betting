import argparse
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import tensorflow as tf
from colorama import Fore, Style

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games

# Updated data sources to use database tables
todays_games_url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'

# NFL team name mapping for consistent team identification
NFL_TEAM_MAPPING = {
    # AFC East
    'Buffalo Bills': 'BUF', 'Miami Dolphins': 'MIA', 'New England Patriots': 'NE', 'New York Jets': 'NYJ',
    # AFC North
    'Baltimore Ravens': 'BAL', 'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Pittsburgh Steelers': 'PIT',
    # AFC South
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX', 'Tennessee Titans': 'TEN',
    # AFC West
    'Denver Broncos': 'DEN', 'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    # NFC East
    'Dallas Cowboys': 'DAL', 'New York Giants': 'NYG', 'Philadelphia Eagles': 'PHI', 'Washington Commanders': 'WAS',
    # NFC North
    'Chicago Bears': 'CHI', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB', 'Minnesota Vikings': 'MIN',
    # NFC South
    'Atlanta Falcons': 'ATL', 'Carolina Panthers': 'CAR', 'New Orleans Saints': 'NO', 'Tampa Bay Buccaneers': 'TB',
    # NFC West
    'Arizona Cardinals': 'ARI', 'Los Angeles Rams': 'LAR', 'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA'
}


def get_team_index(team_name, team_df):
    """Get team index from team dataframe based on team name"""
    # Try exact match first
    exact_match = team_df[team_df['TEAM_NAME'] == team_name]
    if not exact_match.empty:
        return exact_match.index[0]
    
    # Try abbreviation match
    team_abbrev = NFL_TEAM_MAPPING.get(team_name)
    if team_abbrev:
        abbrev_match = team_df[team_df['TEAM_NAME'].str.contains(team_abbrev, case=False, na=False)]
        if not abbrev_match.empty:
            return abbrev_match.index[0]
    
    # Try partial name match
    for idx, row in team_df.iterrows():
        if team_name.lower() in row['TEAM_NAME'].lower() or row['TEAM_NAME'].lower() in team_name.lower():
            return idx
    
    return None

def createTodaysGames(games, team_df, odds):
    """Create today's games data using the new database structure"""
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []

    print(f"Processing {len(games)} games for today...")

    for game in games:
        home_team = game[0]
        away_team = game[1]
        
        print(f"Processing: {away_team} @ {home_team}")
        
        # Get team indices using the new mapping function
        home_team_idx = get_team_index(home_team, team_df)
        away_team_idx = get_team_index(away_team, team_df)
        
        if home_team_idx is None or away_team_idx is None:
            print(f"  Skipping - team not found in database")
            print(f"    Home team '{home_team}' found: {home_team_idx is not None}")
            print(f"    Away team '{away_team}' found: {away_team_idx is not None}")
            continue

        # Handle odds data
        if odds is not None:
            try:
                game_key = home_team + ':' + away_team
                if game_key in odds:
                    game_odds = odds[game_key]
                    todays_games_uo.append(game_odds.get('under_over_odds', 0))
                    home_team_odds.append(game_odds.get(home_team, {}).get('money_line_odds', 0))
                    away_team_odds.append(game_odds.get(away_team, {}).get('money_line_odds', 0))
                else:
                    print(f"  No odds found for {game_key}")
                    todays_games_uo.append(0)
                    home_team_odds.append(0)
                    away_team_odds.append(0)
            except Exception as e:
                print(f"  Error processing odds: {e}")
                todays_games_uo.append(0)
                home_team_odds.append(0)
                away_team_odds.append(0)
        else:
            # Manual input fallback
            try:
                ou_input = input(f"{away_team} @ {home_team} OU: ")
                todays_games_uo.append(float(ou_input))
                home_team_odds.append(float(input(f"{home_team} ML odds: ")))
                away_team_odds.append(float(input(f"{away_team} ML odds: ")))
            except ValueError:
                print("  Invalid input, using default values")
                todays_games_uo.append(45.0)
                home_team_odds.append(0)
                away_team_odds.append(0)

        # Calculate days rest (simplified for now)
        home_days_off = 7  # Default
        away_days_off = 7  # Default
        
        # Get team stats
        home_team_series = team_df.iloc[home_team_idx]
        away_team_series = team_df.iloc[away_team_idx]
        
        # Create game record by combining home and away team stats
        stats = pd.concat([
            home_team_series, 
            away_team_series.rename(index={col: f"{col}.1" for col in team_df.columns.values})
        ])
        
        # Add days rest information
        stats['Days-Rest-Home'] = home_days_off
        stats['Days-Rest-Away'] = away_days_off
        
        match_data.append(stats)
        print(f"  Added game data successfully")

    if not match_data:
        print("No valid games found!")
        return None, None, None, None, None

    # Create final dataframe
    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    # Remove team ID columns if they exist
    columns_to_drop = [col for col in games_data_frame.columns if 'TEAM_ID' in col]
    if columns_to_drop:
        frame_ml = games_data_frame.drop(columns=columns_to_drop)
    else:
        frame_ml = games_data_frame

    # Convert to numpy array, handling non-numeric columns
    # Use the same feature selection as in training scripts
    exclude_columns = [
        'Score', 'Home-Team-Win', 'TEAM_NAME', 'TEAM_NAME.1', 
        'OU-Cover', 'OU', 'Days-Rest-Home', 'Days-Rest-Away',
        'index', 'TEAM_ID', 'TEAM_ID.1', 'SEASON', 'SEASON.1',
        'Date', 'Date.1', 'index.1'
    ]
    
    # Get feature columns (same as training)
    feature_columns = [col for col in frame_ml.columns if col not in exclude_columns]
    
    print(f"Excluding {len(exclude_columns)} non-feature columns")
    print(f"Using {len(feature_columns)} feature columns for predictions")
    print(f"Feature columns: {feature_columns[:10]}...")  # Show first 10
    
    # Create data with only feature columns
    if not feature_columns:
        print("Warning: No feature columns found! Using all numeric columns...")
        # Fallback to numeric columns only
        numeric_exclude = ['TEAM_NAME', 'TEAM_NAME.1', 'Date', 'Date.1', 'index', 'index.1', 'SEASON', 'SEASON.1']
        feature_columns = [col for col in frame_ml.columns if col not in numeric_exclude]
    
    feature_data = frame_ml[feature_columns].values
    
    # Convert to float, handling any remaining non-numeric values
    try:
        data = feature_data.astype(float)
        print("Successfully converted data to float")
    except ValueError as e:
        print(f"Error converting to float: {e}")
        print("Attempting fallback conversion method...")
        
        # Fallback: convert each column individually and handle errors
        data_df = pd.DataFrame(feature_data, columns=feature_columns)
        data_df = data_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        data = data_df.values.astype(float)
        print("Used fallback conversion method successfully")

    print(f"Created dataset with {len(data)} games and {data.shape[1]} features")
    print(f"Feature count validation: {data.shape[1]} features (should match model training)")
    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def main():
    print("NFL ML Betting Prediction System")
    print("=" * 50)
    
    # Load team data from database
    try:
        con = sqlite3.connect("Data/TeamData.sqlite")
        
        # Get the most recent team stats table
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)
        team_tables = [t[0] for t in tables.values if 'nfl_team_stats' in t[0]]
        
        if not team_tables:
            print("No team stats tables found in database!")
            print("Please run Get_Data.py first to populate team statistics.")
            return
        
        # Use the most recent table (highest year)
        latest_table = sorted(team_tables)[-1]
        print(f"Using team stats from: {latest_table}")
        
        team_df = pd.read_sql_query(f"SELECT * FROM `{latest_table}`", con)
        con.close()
        
        if team_df.empty:
            print("No team data found in database!")
            return
            
        print(f"Loaded {len(team_df)} teams from database")
        
    except Exception as e:
        print(f"Error loading team data: {e}")
        print("Please ensure Get_Data.py has been run to populate the database.")
        return

    # Handle odds and games
    odds = None
    if args.odds:
        try:
            odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
            games = create_todays_games_from_odds(odds)
            if len(games) == 0:
                print("No games found from odds provider.")
                return
            if (games[0][0] + ':' + games[0][1]) not in list(odds.keys()):
                print(games[0][0] + ':' + games[0][1])
                print(Fore.RED,"--------------Games list not up to date for todays games!!! Scraping disabled until list is updated.--------------")
                print(Style.RESET_ALL)
                odds = None
            else:
                print(f"------------------{args.odds} odds data------------------")
                for g in odds.keys():
                    home_team, away_team = g.split(":")
                    print(f"{away_team} ({odds[g][away_team]['money_line_odds']}) @ {home_team} ({odds[g][home_team]['money_line_odds']})")
        except Exception as e:
            print(f"Error loading odds: {e}")
            odds = None
    else:
        try:
            data = get_todays_games_json(todays_games_url)
            games = create_todays_games(data)
        except Exception as e:
            print(f"Error loading today's games: {e}")
            print("Please provide odds data using -odds parameter or check ESPN API connection.")
            return

    # Create today's games data
    result = createTodaysGames(games, team_df, odds)
    if result[0] is None:
        print("Failed to create games data. Exiting.")
        return
    
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = result
    if args.nn:
        print("------------Neural Network Model Predictions-----------")
        data = tf.keras.utils.normalize(data, axis=1)
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
    if args.xgb:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
    if args.A:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
        data = tf.keras.utils.normalize(data, axis=1)
        print("------------Neural Network Model Predictions-----------")
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from. (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    args = parser.parse_args()
    main()
