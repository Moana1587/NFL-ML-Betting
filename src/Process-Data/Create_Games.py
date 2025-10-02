import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import toml

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

config = toml.load("../../config.toml")

# NFL team name mapping for consistent team identification
NFL_TEAM_MAPPING = {
    # AFC East
    'Buffalo Bills': 'BUF',
    'Miami Dolphins': 'MIA', 
    'New England Patriots': 'NE',
    'New York Jets': 'NYJ',
    
    # AFC North
    'Baltimore Ravens': 'BAL',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Pittsburgh Steelers': 'PIT',
    
    # AFC South
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Tennessee Titans': 'TEN',
    
    # AFC West
    'Denver Broncos': 'DEN',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    
    # NFC East
    'Dallas Cowboys': 'DAL',
    'New York Giants': 'NYG',
    'Philadelphia Eagles': 'PHI',
    'Washington Commanders': 'WAS',
    
    # NFC North
    'Chicago Bears': 'CHI',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Minnesota Vikings': 'MIN',
    
    # NFC South
    'Atlanta Falcons': 'ATL',
    'Carolina Panthers': 'CAR',
    'New Orleans Saints': 'NO',
    'Tampa Bay Buccaneers': 'TB',
    
    # NFC West
    'Arizona Cardinals': 'ARI',
    'Los Angeles Rams': 'LAR',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA'
}

df = pd.DataFrame
scores = []
win_margin = []
OU = []
OU_Cover = []
games = []
days_rest_away = []
days_rest_home = []
teams_con = sqlite3.connect("../../Data/TeamData.sqlite")
odds_con = sqlite3.connect("../../Data/OddsData.sqlite")

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

# Process each season in the create-games config
for key, value in config['create-games'].items():
    print(f"Processing season: {key}")
    
    # Get odds data - table name format: odds_{key}
    odds_table_name = f"odds_{key}" if key.isdigit() else key
    try:
        odds_df = pd.read_sql_query(f"SELECT * FROM `{odds_table_name}`", odds_con)
        print(f"Found {len(odds_df)} odds records for {key}")
    except Exception as e:
        print(f"Error reading odds data for {key}: {e}")
        continue
    
    # Get team stats data - table name format: nfl_team_stats_{year}
    season_year = value.get('start_year', key)
    team_table_name = f"nfl_team_stats_{season_year}"
    try:
        team_df = pd.read_sql_query(f"SELECT * FROM `{team_table_name}`", teams_con)
        print(f"Found {len(team_df)} team records for {season_year}")
    except Exception as e:
        print(f"Error reading team data for {season_year}: {e}")
        continue
    
    # Process each game in the odds data
    for index, row in odds_df.iterrows():
        home_team = row['Home']
        away_team = row['Away']
        game_date = row['Date']
        points = row['Points']  # Total points scored
        ou_value = row['OU']    # Over/Under value
        win_margin_value = row['Win_Margin']  # Home team win margin
        days_rest_home_value = row['Days_Rest_Home']
        days_rest_away_value = row['Days_Rest_Away']
        
        # Skip if we don't have complete data
        if pd.isna(points) or pd.isna(ou_value) or pd.isna(win_margin_value):
            print(f"Skipping game {away_team} @ {home_team} - missing data")
            continue
        
        # Get team indices
        home_team_idx = get_team_index(home_team, team_df)
        away_team_idx = get_team_index(away_team, team_df)
        
        if home_team_idx is None or away_team_idx is None:
            print(f"Skipping game {away_team} @ {home_team} - team not found in stats")
            print(f"  Home team '{home_team}' found: {home_team_idx is not None}")
            print(f"  Away team '{away_team}' found: {away_team_idx is not None}")
            continue
        
        # Add game data
        scores.append(points)
        OU.append(ou_value)
        days_rest_home.append(days_rest_home_value)
        days_rest_away.append(days_rest_away_value)
        
        # Win margin: 1 if home team wins, 0 if away team wins
        if win_margin_value > 0:
            win_margin.append(1)
        else:
            win_margin.append(0)
        
        # OU Cover: 0 = under, 1 = over, 2 = push
        if points < ou_value:
            OU_Cover.append(0)
        elif points > ou_value:
            OU_Cover.append(1)
        else:
            OU_Cover.append(2)
        
        # Get team stats
        home_team_series = team_df.iloc[home_team_idx]
        away_team_series = team_df.iloc[away_team_idx]
        
        # Create game record by combining home and away team stats
        game = pd.concat([
            home_team_series, 
            away_team_series.rename(index={col: f"{col}.1" for col in team_df.columns.values})
        ])
        games.append(game)
odds_con.close()
teams_con.close()

# Create final dataset
if games:
    print(f"Creating dataset with {len(games)} games")
    season = pd.concat(games, ignore_index=True, axis=1)
    season = season.T
    
    # Remove team ID columns if they exist
    columns_to_drop = [col for col in season.columns if 'TEAM_ID' in col]
    if columns_to_drop:
        frame = season.drop(columns=columns_to_drop)
    else:
        frame = season
    
    # Add game outcome columns
    frame['Score'] = np.asarray(scores)
    frame['Home-Team-Win'] = np.asarray(win_margin)
    frame['OU'] = np.asarray(OU)
    frame['OU-Cover'] = np.asarray(OU_Cover)
    frame['Days-Rest-Home'] = np.asarray(days_rest_home)
    frame['Days-Rest-Away'] = np.asarray(days_rest_away)
    
    # Fix data types for numeric columns
    for field in frame.columns.values:
        if 'TEAM_' in field or 'Date' in field or field in ['Score', 'Home-Team-Win', 'OU', 'OU-Cover', 'Days-Rest-Home', 'Days-Rest-Away']:
            continue
        try:
            frame[field] = pd.to_numeric(frame[field], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not convert column {field} to numeric: {e}")
    
    # Save to database
    con = sqlite3.connect("../../Data/dataset.sqlite")
    frame.to_sql("dataset_nfl_new", con, if_exists="replace", index=False)
    con.close()
    
    print(f"Dataset created successfully with {len(frame)} games and {len(frame.columns)} features")
    print(f"Sample of dataset:")
    print(frame[['TEAM_NAME', 'TEAM_NAME.1', 'Score', 'OU', 'Home-Team-Win', 'OU-Cover']].head())
    
else:
    print("No games found to create dataset")
