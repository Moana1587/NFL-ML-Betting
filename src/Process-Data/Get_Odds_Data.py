import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import toml

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.tools import get_json_data

df_data = []

config = toml.load("../../config.toml")
odds_url_template = config['odds_url_template']
odds_api_key = config['odds_api_key']

# Preferred bookmaker for consistent odds data
PREFERRED_BOOKMAKER = "draftkings"  # Options: draftkings, fanduel, betmgm, etc.

# OU calculation parameters
DEFAULT_OU_ESTIMATE = 45.0  # Default OU when no data available
OU_CALCULATION_ENABLED = True  # Enable OU calculation when NULL
OU_SPREAD_MULTIPLIER = 0.5  # Multiplier for spread-based OU calculation
OU_ML_BASE = 42.0  # Base OU for moneyline-based calculation
OU_ML_RANGE = 8.0  # Range for moneyline-based calculation

# ESPN API for scores
scores_url_template = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

con = sqlite3.connect("../../Data/OddsData.sqlite")

# Function to clean up database indexes and tables
def cleanup_database(connection):
    """Clean up any existing database indexes and tables to avoid conflicts"""
    try:
        cursor = connection.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        # Get all index names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = cursor.fetchall()
        
        print(f"Found {len(tables)} existing tables and {len(indexes)} indexes in database")
        
        # Clean up any problematic indexes
        for index in indexes:
            index_name = index[0]
            if 'ix_' in index_name and 'index' in index_name:
                try:
                    cursor.execute(f"DROP INDEX IF EXISTS `{index_name}`")
                    print(f"Dropped index: {index_name}")
                except Exception as e:
                    print(f"Could not drop index {index_name}: {e}")
        
        connection.commit()
        print("Database cleanup completed")
        
    except Exception as e:
        print(f"Error during database cleanup: {e}")

# Clean up database before starting
cleanup_database(con)

# Helper function to convert American odds to decimal
def american_to_decimal(american_odds):
    """Convert American odds to decimal odds"""
    if american_odds is None:
        return None
    
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

# Helper function to get odds from a specific bookmaker for a specific team
def get_bookmaker_odds(bookmakers, market_type, team_name, preferred_bookmaker="draftkings"):
    """Get odds from a specific bookmaker for a specific team"""
    # Try preferred bookmaker first
    for bookmaker in bookmakers:
        if bookmaker.get('key') == preferred_bookmaker:
            markets = bookmaker.get('markets', [])
            for market in markets:
                if market.get('key') == market_type:
                    outcomes = market.get('outcomes', [])
                    for outcome in outcomes:
                        if outcome.get('name') == team_name:
                            price = outcome.get('price')
                            if price is not None:
                                return price
    
    # If preferred bookmaker not found, use first available bookmaker
    for bookmaker in bookmakers:
        markets = bookmaker.get('markets', [])
        for market in markets:
            if market.get('key') == market_type:
                outcomes = market.get('outcomes', [])
                for outcome in outcomes:
                    if outcome.get('name') == team_name:
                    price = outcome.get('price')
                    if price is not None:
                            return price
    return None

# Helper function to get spread from a specific bookmaker
def get_bookmaker_spread(bookmakers, home_team, preferred_bookmaker="draftkings"):
    """Get spread from a specific bookmaker for the home team"""
    # Try preferred bookmaker first
    for bookmaker in bookmakers:
        if bookmaker.get('key') == preferred_bookmaker:
            markets = bookmaker.get('markets', [])
            for market in markets:
                if market.get('key') == 'spreads':
                    outcomes = market.get('outcomes', [])
                    for outcome in outcomes:
                        if outcome.get('name') == home_team:
                            point = outcome.get('point')
                            if point is not None:
                                return point
    
    # If preferred bookmaker not found, use first available bookmaker
    for bookmaker in bookmakers:
        markets = bookmaker.get('markets', [])
        for market in markets:
            if market.get('key') == 'spreads':
                outcomes = market.get('outcomes', [])
                for outcome in outcomes:
                    if outcome.get('name') == home_team:
                    point = outcome.get('point')
                    if point is not None:
                                    return point
    return None

# Helper function to get total from a specific bookmaker
def get_bookmaker_total(bookmakers, preferred_bookmaker="draftkings"):
    """Get total from a specific bookmaker"""
    # Try preferred bookmaker first
    for bookmaker in bookmakers:
        if bookmaker.get('key') == preferred_bookmaker:
            markets = bookmaker.get('markets', [])
            for market in markets:
                if market.get('key') == 'totals':
                    outcomes = market.get('outcomes', [])
                    for outcome in outcomes:
                        point = outcome.get('point')
                        if point is not None:
                            return point
    
    # If preferred bookmaker not found, use first available bookmaker
    for bookmaker in bookmakers:
        markets = bookmaker.get('markets', [])
        for market in markets:
            if market.get('key') == 'totals':
                outcomes = market.get('outcomes', [])
                for outcome in outcomes:
                    point = outcome.get('point')
                    if point is not None:
                        return point
    return None

# Helper function to calculate OU when it's NULL
def calculate_ou_estimate(home_team, away_team, game_date, spread=None, home_ml=None, away_ml=None):
    """Calculate OU estimate when bookmaker data is NULL"""
    
    if not OU_CALCULATION_ENABLED:
        return None
    
    print(f"  Calculating OU estimate for {away_team} @ {home_team}")
    
    # Method 1: Use spread-based estimation
    if spread is not None:
        # Basic formula: OU ≈ 45 + (|spread| * multiplier)
        # This assumes higher spreads correlate with higher scoring games
        spread_ou = DEFAULT_OU_ESTIMATE + (abs(spread) * OU_SPREAD_MULTIPLIER)
        spread_ou = round(spread_ou, 1)  # Round to 1 decimal place
        print(f"    Spread-based OU estimate: {spread_ou:.1f} (spread: {spread})")
        
        # Method 2: Use moneyline-based estimation
        if home_ml is not None and away_ml is not None:
            # Convert ML to implied probabilities
            home_prob = 1 / american_to_decimal(home_ml) if home_ml else 0.5
            away_prob = 1 / american_to_decimal(away_ml) if away_ml else 0.5
            
            # Higher probability games tend to have higher OUs
            ml_ou = OU_ML_BASE + (max(home_prob, away_prob) * OU_ML_RANGE)
            ml_ou = round(ml_ou, 1)  # Round to 1 decimal place
            print(f"    ML-based OU estimate: {ml_ou:.1f} (home_prob: {home_prob:.3f}, away_prob: {away_prob:.3f})")
            
            # Average the two methods
            final_ou = (spread_ou + ml_ou) / 2
            final_ou = round(final_ou, 1)  # Round to 1 decimal place
            print(f"    Combined OU estimate: {final_ou:.1f}")
            return final_ou
        else:
            return spread_ou
    
    # Method 3: Use moneyline only if no spread
    elif home_ml is not None and away_ml is not None:
        home_prob = 1 / american_to_decimal(home_ml) if home_ml else 0.5
        away_prob = 1 / american_to_decimal(away_ml) if away_ml else 0.5
        
        ml_ou = OU_ML_BASE + (max(home_prob, away_prob) * OU_ML_RANGE)
        ml_ou = round(ml_ou, 1)  # Round to 1 decimal place
        print(f"    ML-only OU estimate: {ml_ou:.1f}")
        return ml_ou
    
    # Method 4: Use default based on game date (season progression)
    else:
        try:
            # NFL season typically runs September to January
            # Early season games tend to have lower OUs, late season higher
            if game_date.month >= 9 and game_date.month <= 12:
                # Early season (Sep-Oct): lower OU
                if game_date.month <= 10:
                    season_ou = 43.0
                # Late season (Nov-Dec): higher OU
                else:
                    season_ou = 47.0
            else:
                # Playoffs/other: use default
                season_ou = DEFAULT_OU_ESTIMATE
            
            season_ou = round(season_ou, 1)  # Round to 1 decimal place
            print(f"    Season-based OU estimate: {season_ou:.1f} (month: {game_date.month})")
            return season_ou
            
        except Exception as e:
            print(f"    Error in season-based calculation: {e}")
            print(f"    Using default OU: {DEFAULT_OU_ESTIMATE}")
            return round(DEFAULT_OU_ESTIMATE, 1)  # Round to 1 decimal place

# Helper function to get game scores from ESPN API
def get_game_scores(start_date, end_date, start_processing_time=None, max_processing_time=None):
    """Get game scores from ESPN API for the given date range"""
    print(f"Fetching scores from ESPN API for {start_date} to {end_date}")
    
    # Process each date in the range
    scores_dict = {}
    current_date = start_date
    
    while current_date <= end_date:
        # Check if we've exceeded maximum processing time
        if start_processing_time and max_processing_time:
            elapsed_time = time.time() - start_processing_time
            if elapsed_time > max_processing_time:
                print(f"Maximum processing time exceeded during score fetching. Stopping.")
                break
            
        # Format date for ESPN API (YYYYMMDD)
        date_str = current_date.strftime("%Y%m%d")
        
        # ESPN API URL with date parameter
        scores_url = f"{scores_url_template}?dates={date_str}&lang=en&region=us"
        
        print(f"Fetching scores from: {scores_url}")
        print(f"Progress: Processing {current_date} ({current_date - start_date + timedelta(days=1)} days processed)")
        
        try:
            raw_data = get_json_data(scores_url, timeout=15, max_retries=2)
        except Exception as e:
            print(f"Error fetching scores for {current_date}: {e}")
            current_date += timedelta(days=1)
            continue
        
        if not raw_data or 'events' not in raw_data:
            print(f"No scores data found for {current_date}")
            current_date += timedelta(days=1)
            continue
        
        # Check if events array is empty (no games on this date)
        if not raw_data['events'] or len(raw_data['events']) == 0:
            print(f"No games found for {current_date} (empty events array)")
            current_date += timedelta(days=1)
            continue
        
        # Process scores data for this date
        for event in raw_data['events']:
            try:
                # Extract game date
                game_time_str = event.get('date', '')
                if not game_time_str:
                    continue
                    
                game_datetime = datetime.fromisoformat(game_time_str.replace('Z', '+00:00'))
                game_date = game_datetime.date()
                
                # Skip games outside our date range
                if game_date < start_date or game_date > end_date:
                    continue
                
                # Extract team information and scores from competitions
                competitions = event.get('competitions', [])
                if not competitions:
                    continue
                    
                competition = competitions[0]
                competitors = competition.get('competitors', [])
                
                if len(competitors) < 2:
                    continue
                
                # Find home and away teams
                home_team = None
                away_team = None
                home_score = None
                away_score = None
                
                for competitor in competitors:
                    if competitor.get('homeAway') == 'home':
                        home_team = competitor.get('team', {}).get('displayName', '')
                        home_score = int(competitor.get('score', 0))
                    elif competitor.get('homeAway') == 'away':
                        away_team = competitor.get('team', {}).get('displayName', '')
                        away_score = int(competitor.get('score', 0))
                
                # Check if scores are available
                if home_team and away_team and home_score is not None and away_score is not None:
                    # Create key for lookup
                    key = f"{game_date}_{away_team}_{home_team}"
                    total_points = home_score + away_score
                    scores_dict[key] = {
                        'date': game_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'win_margin': home_score - away_score,
                        'points': total_points
                    }
                    print(f"Found score: {away_team} @ {home_team} on {game_date}: {away_score}-{home_score} (Win Margin: {home_score - away_score}, Total Points: {total_points})")
                    print(f"  Score key: {key}")
                else:
                    print(f"Missing score data for game on {game_date}: home_team='{home_team}', away_team='{away_team}', home_score={home_score}, away_score={away_score}")
            
            except Exception as e:
                print(f"Error processing score data: {e}")
                continue
        
        current_date += timedelta(days=1)
        time.sleep(1)  # Be respectful to the API
    
    print(f"Retrieved {len(scores_dict)} game scores")
    return scores_dict

# Filter to only process recent years (2020 and later) to avoid endless runs
def should_process_year(year_key):
    """Determine if we should process this year based on availability of odds data"""
    try:
        year = int(year_key)
        # Only process years 2020 and later to avoid endless runs
        return year >= 2020
    except (ValueError, TypeError):
        return False

# Process only recent years to avoid endless runs
print("Filtering to process only recent years (2020 and later) to avoid endless runs...")
filtered_config = {k: v for k, v in config['get-odds-data'].items() if should_process_year(k)}

if not filtered_config:
    print("No recent years found in config. Please add years 2020 or later.")
    sys.exit(1)

print(f"Processing {len(filtered_config)} recent years: {list(filtered_config.keys())}")

for key, value in filtered_config.items():
    start_date = datetime.strptime(value['start_date'], "%Y-%m-%d").date()
    end_date = datetime.strptime(value['end_date'], "%Y-%m-%d").date()
    teams_last_played = {}
    
    # Create a safe table name (add prefix to avoid numeric table name issues)
    safe_table_name = f"odds_{key}" if key.isdigit() else key
    
    # Set maximum processing time (30 minutes per year since we're making fewer API calls)
    max_processing_time = 30 * 60  # 30 minutes in seconds
    start_processing_time = time.time()

    print(f"Getting odds data for {key} from {start_date} to {end_date}")
    print(f"Table name: {safe_table_name}")
    print(f"Maximum processing time: {max_processing_time/60:.1f} minutes")
    print(f"Optimization: Using single API call per season instead of per date")
    
    # Get game scores from ESPN API
    game_scores = get_game_scores(start_date, end_date, start_processing_time, max_processing_time)
    
    if not game_scores:
        print(f"Warning: No game scores found for the date range {start_date} to {end_date}")
        print("Continuing with odds data only (win_margin and points will be 0)")

    # Make single API call per season using season start date
    print(f"Making single odds API call for season {key} using start date: {start_date}")
    
    # Format start date for The Odds API (ISO format with timezone)
    start_date_str = start_date.strftime("%Y-%m-%dT12:00:00Z")
    
    # Construct API URL with parameters - single call for the entire season
    api_url = f"{odds_url_template}/?apiKey={odds_api_key}&regions=us&markets=h2h,spreads,totals&oddsFormat=american&date={start_date_str}"
    
    print(f"Fetching odds from: {api_url}")
    
    # Get odds data from The Odds API with timeout
    try:
        raw_data = get_json_data(api_url, timeout=30, max_retries=2)
    except Exception as e:
        print(f"Error fetching odds data for season {key}: {e}")
        continue
        
        if not raw_data or 'data' not in raw_data:
            print(f"No odds data found for season {key}")
            continue

    # Check if we have any games to process
    if not raw_data['data'] or len(raw_data['data']) == 0:
        print(f"No games found in odds data for season {key}")
        continue

    print(f"Found {len(raw_data['data'])} games for season {key}")
    
    # Filter games to only include those within our date range
    all_games_data = []
        for game in raw_data['data']:
        try:
            # Extract game date
            game_time_str = game.get('commence_time', '')
            if not game_time_str:
                continue
            
            # Parse game time
            try:
                game_datetime = datetime.fromisoformat(game_time_str.replace('Z', '+00:00'))
                game_date = game_datetime.date()
            except (ValueError, TypeError):
                continue
            
            # Only include games within our date range
            if start_date <= game_date <= end_date:
                all_games_data.append(game)
                
        except Exception as e:
            print(f"Error processing game date: {e}")
            continue

    print(f"Total games collected for season {key}: {len(all_games_data)} (filtered from {len(raw_data['data'])} total games)")

    # Process each game in the collected odds data
    for game_index, game in enumerate(all_games_data):
        # Check if we've exceeded maximum processing time
        elapsed_time = time.time() - start_processing_time
        if elapsed_time > max_processing_time:
            print(f"Maximum processing time exceeded ({elapsed_time/3600:.1f} hours). Stopping processing.")
            break
            
        print(f"Processing game {game_index + 1}/{len(all_games_data)} (Elapsed: {elapsed_time/60:.1f} minutes)")
        
            try:
                # Extract game information
                game_time_str = game.get('commence_time', '')
                if not game_time_str:
                    continue
                
                # Parse game time
                try:
                    game_datetime = datetime.fromisoformat(game_time_str.replace('Z', '+00:00'))
                    game_date = game_datetime.date()
                except (ValueError, TypeError):
                    print(f"Could not parse game time: {game_time_str}")
                    continue
                
                # Skip games outside our date range
                if game_date < start_date or game_date > end_date:
                    continue
                
                print(f"Processing game on {game_date}")
                
                # Extract team information
                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')
                
                if not home_team or not away_team:
                    continue
                
                # Calculate days rest for teams
                if home_team not in teams_last_played:
                    teams_last_played[home_team] = game_date
                    home_games_rested = timedelta(days=7)  # start of season, big number
                else:
                    home_games_rested = game_date - teams_last_played[home_team]
                    teams_last_played[home_team] = game_date

                if away_team not in teams_last_played:
                    teams_last_played[away_team] = game_date
                    away_games_rested = timedelta(days=7)  # start of season, big number
                else:
                    away_games_rested = game_date - teams_last_played[away_team]
                    teams_last_played[away_team] = game_date

                # Extract odds data from bookmakers
                bookmakers = game.get('bookmakers', [])
                
            # Debug: Print sample bookmaker data for first game
            if len(df_data) == 0 and bookmakers:
                print(f"Sample bookmaker data for {away_team} @ {home_team}:")
                for i, bookmaker in enumerate(bookmakers[:1]):  # Show first bookmaker only
                    print(f"  Bookmaker {i+1}: {bookmaker.get('title', 'Unknown')}")
                    for market in bookmaker.get('markets', []):
                        print(f"    Market: {market.get('key')}")
                        for outcome in market.get('outcomes', []):
                            print(f"      Outcome: {outcome}")
                            # Specifically look at spreads and totals
                            if market.get('key') in ['spreads', 'totals']:
                                print(f"        Point: {outcome.get('point')}")
                                print(f"        Price: {outcome.get('price')}")
                                print(f"        Name: {outcome.get('name')}")
            
            # Get odds from specific bookmaker (preferred bookmaker, fallback to first available)
            home_ml = get_bookmaker_odds(bookmakers, 'h2h', home_team, PREFERRED_BOOKMAKER)
            away_ml = get_bookmaker_odds(bookmakers, 'h2h', away_team, PREFERRED_BOOKMAKER)
            spread = get_bookmaker_spread(bookmakers, home_team, PREFERRED_BOOKMAKER)
            total = get_bookmaker_total(bookmakers, PREFERRED_BOOKMAKER)
            
            # Debug: Show which bookmaker was used
            used_bookmaker = "Unknown"
            if bookmakers:
                # Check if preferred bookmaker was found
                for bookmaker in bookmakers:
                    if bookmaker.get('key') == PREFERRED_BOOKMAKER:
                        used_bookmaker = bookmaker.get('title', PREFERRED_BOOKMAKER)
                        break
                else:
                    # If preferred not found, show first available
                    used_bookmaker = bookmakers[0].get('title', 'Unknown')
                print(f"  Using bookmaker: {used_bookmaker} (preferred: {PREFERRED_BOOKMAKER})")
            
            # Debug: Check for NULL OU values and calculate if needed
            if total is None:
                print(f"WARNING: No totals (OU) data found for {away_team} @ {home_team}")
                print(f"  Bookmakers available: {len(bookmakers)}")
                for i, bookmaker in enumerate(bookmakers[:2]):  # Check first 2 bookmakers
                    print(f"    Bookmaker {i+1}: {bookmaker.get('title')}")
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'totals':
                            print(f"      Totals market found with {len(market.get('outcomes', []))} outcomes")
                        else:
                            print(f"      Market: {market.get('key')}")
                
                # Calculate OU estimate when NULL
                print(f"  Attempting to calculate OU estimate...")
                calculated_ou = calculate_ou_estimate(home_team, away_team, game_date, spread, home_ml, away_ml)
                if calculated_ou is not None:
                    total = calculated_ou
                    print(f"  Using calculated OU: {total:.1f}")
                else:
                    print(f"  Could not calculate OU, will remain NULL")
                
                # Convert American odds to decimal for moneyline
            home_ml_decimal = american_to_decimal(home_ml) if home_ml else None
            away_ml_decimal = american_to_decimal(away_ml) if away_ml else None
                
                # Convert decimal odds back to American format for consistency
                def decimal_to_american(decimal_odds):
                    if decimal_odds is None:
                        return None
                    if decimal_odds >= 2:
                        return int((decimal_odds - 1) * 100)
                    else:
                        return int(-100 / (decimal_odds - 1))
                
                home_ml_american = decimal_to_american(home_ml_decimal)
                away_ml_american = decimal_to_american(away_ml_decimal)
            
            # Look up win margin and points from scores
            score_key = f"{game_date}_{away_team}_{home_team}"
            win_margin = 0  # Default to 0 if no score found
            points = 0  # Default to 0 if no score found
            if score_key in game_scores:
                win_margin = game_scores[score_key]['win_margin']
                points = game_scores[score_key]['points']
                print(f"Found win margin for {away_team} @ {home_team}: {win_margin}, Points: {points}")
            else:
                print(f"No score found for {away_team} @ {home_team} on {game_date}")
                # Debug: Show available score keys for this date
                available_keys = [key for key in game_scores.keys() if str(game_date) in key]
                if available_keys:
                    print(f"Available score keys for {game_date}: {available_keys[:5]}...")  # Show first 5
                else:
                    print(f"No scores available for {game_date}")
            
            # Debug: Print calculated values with type information
            print(f"Calculated values for {away_team} @ {home_team}:")
            print(f"  OU: {total} (type: {type(total)})")
            print(f"  Spread: {spread} (type: {type(spread)})")
            print(f"  ML_Home: {home_ml_american} (type: {type(home_ml_american)})")
            print(f"  ML_Away: {away_ml_american} (type: {type(away_ml_american)})")
            print(f"  Win_Margin: {win_margin}")
            print(f"  Points: {points}")
            
            # Debug: Check for NaN values
            if total is not None:
                import math
                if math.isnan(total):
                    print(f"  WARNING: OU value is NaN!")
                    total = None
                
                # Only add data if we have at least some odds information
            if total is not None or spread is not None or home_ml_american is not None or away_ml_american is not None:
                # Clean data before adding to DataFrame
                import math
                
                # Ensure no NaN values are added
                clean_ou = total if total is not None and not math.isnan(total) else None
                clean_spread = spread if spread is not None and not math.isnan(spread) else None
                clean_ml_home = home_ml_american if home_ml_american is not None and not math.isnan(home_ml_american) else None
                clean_ml_away = away_ml_american if away_ml_american is not None and not math.isnan(away_ml_american) else None
                
                    df_data.append({
                        'Date': game_date,
                        'Home': home_team,
                        'Away': away_team,
                    'OU': clean_ou,
                    'Spread': clean_spread,
                    'ML_Home': clean_ml_home,
                    'ML_Away': clean_ml_away,
                    'Win_Margin': win_margin,  # Actual win margin from scores API
                    'Points': points,  # Total points scored (home + away)
                        'Days_Rest_Home': home_games_rested.days,
                        'Days_Rest_Away': away_games_rested.days
                    })
                    print(f"Added odds data for {away_team} @ {home_team} on {game_date}")
            print(f"  Cleaned OU: {clean_ou} (was: {total})")

            except Exception as e:
                print(f"Error processing game {away_team} @ {home_team}: {e}")
        
    # Summary of processing
    print(f"\n=== Processing Summary for {key} ===")
    print(f"Total games collected: {len(all_games_data)}")
    print(f"Games with valid odds data: {len(df_data)}")
    print(f"Games with scores data: {len(game_scores)}")
    print(f"Date range processed: {start_date} to {end_date}")

    df = pd.DataFrame(df_data)
    if not df.empty:
        # Final data validation and cleaning
        print(f"DataFrame created with {len(df)} rows")
        print(f"OU column info:")
        print(f"  Non-null count: {df['OU'].notna().sum()}")
        print(f"  Null count: {df['OU'].isna().sum()}")
        print(f"  Sample OU values: {df['OU'].head().tolist()}")
        
        # Replace any remaining NaN values with None for proper SQLite handling
        df = df.where(pd.notna(df), None)
        
        print(f"After cleaning - OU column info:")
        print(f"  Non-null count: {df['OU'].notna().sum()}")
        print(f"  Null count: {df['OU'].isna().sum()}")
        print(f"  Sample OU values: {df['OU'].head().tolist()}")
        try:
            # Use a more robust approach to handle database operations
            cursor = con.cursor()
            
            # Drop the table and all associated indexes (quote table names to handle numeric names)
            cursor.execute(f"DROP TABLE IF EXISTS `{safe_table_name}`")
            
            # Drop any indexes that might be associated with this table
            cursor.execute(f"DROP INDEX IF EXISTS `ix_{safe_table_name}_index`")
            cursor.execute(f"DROP INDEX IF EXISTS `{safe_table_name}_index`")
            
            # Commit the cleanup
            con.commit()
            
            # Create the table with fresh data, ensuring no index conflicts
            # Use smaller chunksize to avoid SQL variable limits
            chunk_size = 500  # Process in smaller chunks
            df.to_sql(safe_table_name, con, if_exists="replace", index=False, method='multi', chunksize=chunk_size)
            print(f"Saved {len(df)} odds records to database for {key} (table: {safe_table_name})")
            
        except Exception as e:
            print(f"Error saving data to database for {key}: {e}")
            print(f"Error details: {str(e)}")
            
            # Try alternative approach - create table manually first
            try:
                print(f"Attempting alternative database save method...")
                
                # Create table with explicit schema
                cursor = con.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS `{safe_table_name}`")
                con.commit()
                
                # Use a simple approach without pandas index
                # Use even smaller chunksize to avoid SQL variable limits
                df.to_sql(safe_table_name, con, if_exists="replace", index=False, chunksize=200)
                print(f"Saved {len(df)} odds records to database for {key} (table: {safe_table_name}, alternative method)")
                
            except Exception as e2:
                print(f"Failed to save data with alternative method: {e2}")
                print(f"Attempting row-by-row insertion...")
                
                # Try row-by-row insertion to avoid SQL variable limits
                try:
                    cursor = con.cursor()
                    cursor.execute(f"DROP TABLE IF EXISTS `{safe_table_name}`")
                    con.commit()
                    
                    # Create table with proper schema
                    cursor.execute(f"""
                        CREATE TABLE `{safe_table_name}` (
                            Date TEXT,
                            Home TEXT,
                            Away TEXT,
                            OU REAL,
                            Spread REAL,
                            ML_Home INTEGER,
                            ML_Away INTEGER,
                            Win_Margin INTEGER,
                            Points INTEGER,
                            Days_Rest_Home INTEGER,
                            Days_Rest_Away INTEGER
                        )
                    """)
                    
                    # Insert data row by row
                    insert_query = f"""
                        INSERT INTO `{safe_table_name}` 
                        (Date, Home, Away, OU, Spread, ML_Home, ML_Away, Win_Margin, Points, Days_Rest_Home, Days_Rest_Away)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    records_saved = 0
                    for _, row in df.iterrows():
                        try:
                            # Debug: Print row data before insertion
                            if records_saved < 5:  # Only print first 5 rows for debugging
                                print(f"  Row {records_saved + 1} data:")
                                print(f"    OU: {row['OU']} (pd.notna: {pd.notna(row['OU'])})")
                                print(f"    Spread: {row['Spread']} (pd.notna: {pd.notna(row['Spread'])})")
                            
                            # Convert values properly, handling NaN and None
                            ou_value = row['OU'] if pd.notna(row['OU']) and not pd.isna(row['OU']) else None
                            spread_value = row['Spread'] if pd.notna(row['Spread']) and not pd.isna(row['Spread']) else None
                            ml_home_value = int(row['ML_Home']) if pd.notna(row['ML_Home']) and not pd.isna(row['ML_Home']) else None
                            ml_away_value = int(row['ML_Away']) if pd.notna(row['ML_Away']) and not pd.isna(row['ML_Away']) else None
                            win_margin_value = int(row['Win_Margin']) if pd.notna(row['Win_Margin']) and not pd.isna(row['Win_Margin']) else None
                            points_value = int(row['Points']) if pd.notna(row['Points']) and not pd.isna(row['Points']) else None
                            days_rest_home_value = int(row['Days_Rest_Home']) if pd.notna(row['Days_Rest_Home']) and not pd.isna(row['Days_Rest_Home']) else None
                            days_rest_away_value = int(row['Days_Rest_Away']) if pd.notna(row['Days_Rest_Away']) and not pd.isna(row['Days_Rest_Away']) else None
                            
                            cursor.execute(insert_query, (
                                str(row['Date']),
                                str(row['Home']),
                                str(row['Away']),
                                ou_value,
                                spread_value,
                                ml_home_value,
                                ml_away_value,
                                win_margin_value,
                                points_value,
                                days_rest_home_value,
                                days_rest_away_value
                            ))
                            records_saved += 1
                            
                            # Commit every 100 records to avoid memory issues
                            if records_saved % 100 == 0:
                                con.commit()
                                print(f"  Saved {records_saved}/{len(df)} records...")
                                
                        except Exception as row_error:
                            print(f"  Error inserting row {records_saved + 1}: {row_error}")
                            print(f"    Row data: OU={row['OU']}, Spread={row['Spread']}, ML_Home={row['ML_Home']}")
                            continue
                    
                    # Final commit
                    con.commit()
                    print(f"Successfully saved {records_saved} odds records to database for {key} (table: {safe_table_name}, row-by-row method)")
                    
                except Exception as e3:
                    print(f"Failed to save data with row-by-row method: {e3}")
                    print(f"Continuing without saving this dataset...")
    else:
        print(f"No odds data to save for {key}")

    time.sleep(2)

con.close()