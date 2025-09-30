import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import toml
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.tools import get_json_data, to_data_frame

config = toml.load("../../config.toml")

def get_nfl_team_ids():
    """Get all NFL team IDs from ESPN API"""
    print("Getting NFL team IDs...")
    teams_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
    
    raw_data = get_json_data(teams_url)
    team_ids = []
    
    if isinstance(raw_data, dict) and 'sports' in raw_data:
        for sport in raw_data['sports']:
            if sport.get('name') == 'Football':
                for league in sport.get('leagues', []):
                    if league.get('shortName') == 'NFL':
                        for team in league.get('teams', []):
                            team_info = team.get('team', {})
                            team_ids.append({
                                'id': team_info.get('id'),
                                'name': team_info.get('displayName'),
                                'abbreviation': team_info.get('abbreviation'),
                                'slug': team_info.get('slug')
                            })
    
    print(f"Found {len(team_ids)} NFL teams")
    return team_ids

def get_team_stats(team_id, season_year):
    """Get team statistics for a specific team and season"""
    stats_url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season_year}/types/2/teams/{team_id}/statistics"
    
    print(f"Getting stats for team {team_id} for season {season_year}")
    raw_data = get_json_data(stats_url)
    
    if isinstance(raw_data, dict) and 'splits' in raw_data:
        # Extract team statistics from the API response
        stats_data = []
        for split in raw_data['splits']:
            if 'stats' in split:
                for stat in split['stats']:
                    stats_data.append({
                        'team_id': team_id,
                        'season': season_year,
                        'stat_name': stat.get('label', ''),
                        'stat_value': stat.get('value', 0),
                        'stat_display_value': stat.get('displayValue', ''),
                        'stat_category': stat.get('category', ''),
                        'split_type': split.get('type', ''),
                        'split_value': split.get('value', '')
                    })
        return pd.DataFrame(stats_data)
    
    return pd.DataFrame()

def get_team_stats_formatted(team_id, team_name, season_year):
    """Get team statistics in a formatted table structure with comprehensive stats"""
    stats_url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season_year}/types/2/teams/{team_id}/statistics"
    
    print(f"Getting formatted stats for {team_name} (ID: {team_id}) for season {season_year}")
    raw_data = get_json_data(stats_url)
    
    if isinstance(raw_data, dict) and 'splits' in raw_data:
        # Initialize comprehensive team stats dictionary
        team_stats = {
            # Basic Info
            'TEAM_ID': team_id,
            'TEAM_NAME': team_name,
            'SEASON': season_year,
            'GP': 0,            
            'TP/G': 0.0,
            'PYDS/G': 0.0,
            'YDS/G': 0.0,
            'NYDS/PA': 0.0,
            'YDS/RA': 0.0,
            'CMP%': 0.0,
            'INT%': 0.0,
            'INT': 0,
            'SACKS_PASSING': 0,
            'SACKS_DEFENSIVE': 0,
            'SYL': 0.0,
            'TFL': 0.0,
            'PD': 0.0,
            'TT': 0.0,
            'TGV': 0.0,
            'DIFF': 0.0,
            '3RDC%': 0.0,
            '4THC%': 0.0,
            'RZTD%': 0.0,
            'RZ%': 0.0,
            'TPEN': 0.0,
            'TPY': 0.0,
            'POSS': 0.0,
            'TOP': 0.0,
            'FD/G': 0.0,
            'FIRST': 0.0,
            'NYDS/G': 0.0,
            'CAR': 0,
            'TGT': 0,
            'LST': 0
        }
        
        # Process stats from API response - iterate through categories
        for category in raw_data['splits'].get('categories', []):
            category_name = category.get('name', '')
            stats_list = category.get('stats', [])
            
            for stat in stats_list:
                stat_name = stat.get('name', '')
                value = stat.get('value', 0)
                
                # Map all available stats to our columns
                if stat_name == 'gamesPlayed':
                    team_stats['GP'] = int(value)
                elif stat_name == 'totalPointsPerGame':
                    team_stats['TP/G'] = int(value)
                elif stat_name == 'passingYardsPerGame':
                    team_stats['PYDS/G'] = int(value)
                elif stat_name == 'rushingYardsPerGame':
                    team_stats['YDS/G'] = float(value)
                elif stat_name == 'netYardsPerPassAttempt':
                    team_stats['NYDS/PA'] = int(value)
                elif stat_name == 'yardsPerRushAttempt':
                    team_stats['YDS/RA'] = float(value)
                elif stat_name == 'completionPct':
                    team_stats['CMP%'] = int(value)
                elif stat_name == 'interceptionPct':
                    team_stats['INT%'] = float(value)
                elif stat_name == 'interceptions':
                    team_stats['INT'] = int(value)
                elif stat_name == 'sacks' and category_name == 'passing':
                    team_stats['SACKS_PASSING'] = float(value)
                elif stat_name == 'sacks' and category_name == 'defensive':
                    team_stats['SACKS_DEFENSIVE'] = int(value)
                elif stat_name == 'sackYardsLost':
                    team_stats['SYL'] = float(value)
                elif stat_name == 'tacklesForLoss':
                    team_stats['TFL'] = float(value)
                elif stat_name == 'passesDefended':
                    team_stats['PD'] = float(value)
                elif stat_name == 'totalTakeaways':
                    team_stats['TT'] = float(value)
                elif stat_name == 'totalGiveaways':
                    team_stats['TGV'] = float(value)
                elif stat_name == 'turnOverDifferential':
                    team_stats['DIFF'] = float(value)
                elif stat_name == 'thirdDownConvPct':
                    team_stats['3RDC%'] = float(value)
                elif stat_name == 'fourthDownConvPct':
                    team_stats['4THC%'] = float(value)
                elif stat_name == 'redzoneTouchdownPct':
                    team_stats['RZTD%'] = float(value)
                elif stat_name == 'redzoneScoringPct':
                    team_stats['RZ%'] = float(value)
                elif stat_name == 'totalPenalties':
                    team_stats['TPEN'] = float(value)
                elif stat_name == 'totalPenaltyYards':
                    team_stats['TPY'] = float(value)
                elif stat_name == 'possessionTimeSeconds':
                    team_stats['POSS'] = float(value)
                elif stat_name == 'totalOffensivePlays':
                    team_stats['TOP'] = float(value)
                elif stat_name == 'firstDownsPerGame':
                    team_stats['FD/G'] = float(value)
                elif stat_name == 'firstDowns':
                    team_stats['FIRST'] = float(value)
                elif stat_name == 'netPassingYardsPerGame':
                    team_stats['NYDS/G'] = int(value)
                elif stat_name == 'rushingAttempts':
                    team_stats['CAR'] = int(value)
                elif stat_name == 'receivingTargets':
                    team_stats['TGT'] = int(value)
                elif stat_name == 'fumblesLost':
                    team_stats['LST'] = int(value)
                
        
        return team_stats
    
    return None

def get_team_standings(team_id, season_year):
    """Get team standings for a specific team and season"""
    standings_url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season_year}/types/1/teams/{team_id}/statistics"
    
    print(f"Getting standings for team {team_id} for season {season_year}")
    raw_data = get_json_data(standings_url)
    
    if isinstance(raw_data, dict) and 'splits' in raw_data:
        standings_data = []
        for split in raw_data['splits']:
            if 'stats' in split:
                for stat in split['stats']:
                    standings_data.append({
                        'team_id': team_id,
                        'season': season_year,
                        'stat_name': stat.get('label', ''),
                        'stat_value': stat.get('value', 0),
                        'stat_display_value': stat.get('displayValue', ''),
                        'stat_category': stat.get('category', ''),
                        'split_type': split.get('type', ''),
                        'split_value': split.get('value', '')
                    })
        return pd.DataFrame(standings_data)
    
    return pd.DataFrame()

# Main execution
if __name__ == "__main__":
    # Get all NFL team IDs
    team_ids = get_nfl_team_ids()
    
    # Connect to database
    con = sqlite3.connect("../../Data/TeamData.sqlite")
    
    # Get team statistics for each season in config
    for key, value in config['get-data'].items():
        season_year = value['start_year']
        print(f"\nProcessing season {season_year}")
        
        # Get formatted stats for each team
        all_team_stats = []
        
        for index, team in enumerate(team_ids):
            team_id = team['id']
            team_name = team['name']
            
            # Get formatted team statistics
            team_stats = get_team_stats_formatted(team_id, team_name, season_year)
            if team_stats:
                # Add index column (0-based like in the basketball table)
                team_stats['index'] = index
                all_team_stats.append(team_stats)
            
            # Add delay to avoid rate limiting
            time.sleep(random.randint(1, 3))
        
        # Combine all team stats for this season
        if all_team_stats:
            combined_stats_df = pd.DataFrame(all_team_stats)
            
            # Reorder columns to match basketball table format (index first, then team info, then stats)
            column_order = [
                'index',
                'TEAM_ID',
                'TEAM_NAME',
                'SEASON',
                'GP',            
                'TP/G',
                'PYDS/G',
                'YDS/G',
                'NYDS/PA',
                'YDS/RA',
                'CMP%',
                'INT%',
                'INT',
                'SACKS_PASSING',
                'SACKS_DEFENSIVE',
                'SYL',
                'TFL',
                'PD',
                'TT',
                'TGV',
                'DIFF',
                '3RDC%',
                '4THC%',
                'RZTD%',
                'RZ%',
                'TPEN',
                'TPY',
                'POSS',
                'TOP',
                'FD/G',
                'FIRST',
                'NYDS/G',
                'CAR',
                'TGT',
                'LST'
            ]
            
            # Only include columns that exist in the dataframe
            existing_columns = [col for col in column_order if col in combined_stats_df.columns]
            combined_stats_df = combined_stats_df[existing_columns]
            
            # Save to database
            combined_stats_df.to_sql(f'nfl_team_stats_{season_year}', con, if_exists='replace', index=False)
            print(f"Saved formatted team stats for season {season_year} with {len(combined_stats_df)} teams")
            
            # Display sample of the data
            print(f"\nSample data for season {season_year}:")
            print(combined_stats_df[['index', 'TEAM_NAME', 'GP', 'TP/G', 'TFL', 'TT', 'TGV', 'YDS/RA',
                'CMP%', 'INT%']].head())
    
    con.close()
    print("Data collection completed!")
