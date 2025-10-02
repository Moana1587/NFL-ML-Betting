from datetime import date
import json
import os
import sys
from flask import Flask, render_template, jsonify
from functools import lru_cache
import subprocess, requests, re, time


@lru_cache()
def fetch_fanduel(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="fanduel")

@lru_cache()
def fetch_draftkings(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="draftkings")

@lru_cache()
def fetch_betmgm(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="betmgm")

def fetch_game_data(sportsbook="fanduel"):
    try:
        # Ensure we're in the correct directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        cmd = ["python", "main.py", "-xgb", f"-odds={sportsbook}"]
        stdout = subprocess.check_output(
            cmd, 
            cwd=parent_dir, 
            timeout=60,
            stderr=subprocess.PIPE,
            text=True
        )
    except subprocess.TimeoutExpired:
        print(f"Timeout running prediction for {sportsbook}")
        return {}
    except subprocess.CalledProcessError as e:
        print(f"Error running prediction for {sportsbook}: {e}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Working directory: {parent_dir}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return {}
    except FileNotFoundError:
        print(f"Python or main.py not found. Current directory: {parent_dir}")
        return {}
    except Exception as e:
        print(f"Unexpected error for {sportsbook}: {e}")
        return {}
    
    # Updated regex patterns to match the new output format
    # Pattern for: "Los Angeles Rams (67.8%) vs San Francisco 49ers: OVER 45.5 (85.9%)"
    data_re = re.compile(r'(?P<home_team>[\w\s]+)\s+\((?P<home_confidence>[\d\.]+)%\)\s+vs\s+(?P<away_team>[\w\s]+):\s+(?P<ou_pick>OVER|UNDER)\s+(?P<ou_value>[\d\.]+)\s+\((?P<ou_confidence>[\d\.]+)%\)', re.MULTILINE)
    
    # Pattern for Expected Value: "Kansas City Chiefs EV: 0.045"
    ev_re = re.compile(r'(?P<team>[\w\s]+)\s+EV:\s+(?P<ev>[-\d\.]+)', re.MULTILINE)
    
    # Pattern for odds: "Kansas City Chiefs (150) @ Buffalo Bills (-180)"
    odds_re = re.compile(r'(?P<away_team>[\w\s]+)\s+\((?P<away_team_odds>-?\d+)\)\s+@\s+(?P<home_team>[\w\s]+)\s+\((?P<home_team_odds>-?\d+)\)', re.MULTILINE)
    
    # Pattern for betting recommendation: "Recommendation: Bet on Kansas City Chiefs"
    recommendation_re = re.compile(r'Recommendation:\s+(?P<recommendation>Bet on [\w\s]+|No positive EV bets)', re.MULTILINE)
    
    games = {}
    
    # Debug: Print a sample of the output to help with regex debugging
    print(f"Sample output for {sportsbook}:")
    print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
    
    # Parse prediction data
    matches = list(data_re.finditer(stdout))
    print(f"Found {len(matches)} prediction matches")
    
    # If no matches found, try alternative pattern
    if len(matches) == 0:
        print("No matches found with main pattern, trying alternative...")
        # Alternative pattern for different output format
        alt_data_re = re.compile(r'(?P<home_team>[\w\s]+)\s+\((?P<home_confidence>[\d\.]+)%\)\s+vs\s+(?P<away_team>[\w\s]+):\s+(?P<ou_pick>OVER|UNDER)\s+(?P<ou_value>[\d\.]+)\s+\((?P<ou_confidence>[\d\.]+)%\)', re.MULTILINE)
        matches = list(alt_data_re.finditer(stdout))
        print(f"Found {len(matches)} matches with alternative pattern")
    
    for match in matches:
        game_dict = {
            'away_team': match.group('away_team').strip(),
            'home_team': match.group('home_team').strip(),
            'home_confidence': match.group('home_confidence'),
            'ou_pick': match.group('ou_pick'),
            'ou_value': match.group('ou_value'),
            'ou_confidence': match.group('ou_confidence'),
            'away_confidence': None,  # Will be filled from EV data
            'away_team_ev': None,
            'home_team_ev': None,
            'away_team_odds': None,
            'home_team_odds': None,
            'recommendation': None
        }
        
        # Extract Expected Values
        for ev_match in ev_re.finditer(stdout):
            team_name = ev_match.group('team').strip()
            ev_value = ev_match.group('ev')
            
            if team_name == game_dict['away_team']:
                game_dict['away_team_ev'] = ev_value
            elif team_name == game_dict['home_team']:
                game_dict['home_team_ev'] = ev_value
        
        # Extract odds data
        for odds_match in odds_re.finditer(stdout):
            away_team = odds_match.group('away_team').strip()
            home_team = odds_match.group('home_team').strip()
            
            if (away_team == game_dict['away_team'] and 
                home_team == game_dict['home_team']):
                game_dict['away_team_odds'] = odds_match.group('away_team_odds')
                game_dict['home_team_odds'] = odds_match.group('home_team_odds')
                break
        
        # Extract betting recommendation
        for rec_match in recommendation_re.finditer(stdout):
            # Find recommendation that matches this game
            recommendation = rec_match.group('recommendation')
            if (game_dict['away_team'] in recommendation or 
                game_dict['home_team'] in recommendation):
                game_dict['recommendation'] = recommendation
                break
        
        # Calculate away team confidence (inverse of home team confidence)
        if game_dict['home_confidence']:
            try:
                home_conf = float(game_dict['home_confidence'])
                away_conf = round(100 - home_conf, 1)
                game_dict['away_confidence'] = str(away_conf)
            except ValueError:
                pass
        
        print(json.dumps(game_dict, sort_keys=True, indent=4))
        games[f"{game_dict['away_team']}:{game_dict['home_team']}"] = game_dict
    
    return games


def get_ttl_hash(seconds=600):
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)


app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')


@app.route("/")
def index():
    try:
        fanduel = fetch_fanduel(ttl_hash=get_ttl_hash())
        draftkings = fetch_draftkings(ttl_hash=get_ttl_hash())
        betmgm = fetch_betmgm(ttl_hash=get_ttl_hash())

        return render_template('index.html', today=date.today(), data={"fanduel": fanduel, "draftkings": draftkings, "betmgm": betmgm})
    except Exception as e:
        print(f"Error in index route: {e}")
        return f"Error loading predictions: {str(e)}", 500

@app.route("/test")
def test():
    """Simple test route to verify Flask is working"""
    return jsonify({
        "status": "success",
        "message": "Flask app is running",
        "platform": sys.platform,
        "python_version": sys.version
    })




def get_player_data(team_abv):
    """Fetch player data for a given team abbreviation"""
    url = "https://tank01-fantasy-stats.p.rapidapi.com/getNFLTeamRoster"
    headers = {
        "x-rapidapi-key": "a0f0cd0b5cmshfef96ed37a9cda6p1f67bajsnfcdd16f37df8",
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }
    querystring = {"teamAbv": team_abv}
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        
        if data.get('statusCode') == 200:
            formatted_players = []
            roster = data.get('body', {}).get('roster', [])
            
            for player in roster:
                # Format injury status
                injury_status = "Healthy"
                if player.get('injury'):
                    injury_info = player['injury']
                    if injury_info.get('designation'):
                        injury_status = injury_info['designation']
                        if injury_info.get('description'):
                            injury_status += f" - {injury_info['description']}"
                
                formatted_player = {
                    'name': player.get('longName'),
                    'shortName': player.get('shortName'),
                    'headshot': player.get('nbaComHeadshot'),
                    'injury': injury_status,
                    'position': player.get('pos'),
                    'height': player.get('height'),
                    'weight': player.get('weight'),
                    'college': player.get('college'),
                    'experience': player.get('exp'),
                    'jerseyNum': player.get('jerseyNum'),
                    'playerId': player.get('playerID'),
                    'birthDate': player.get('bDay')
                }
                formatted_players.append(formatted_player)
            
            return {
                'success': True,
                'players': formatted_players
            }
        
        return {
            'success': False,
            'error': 'Failed to fetch team data'
        }
        
    except Exception as e:
        print(f"Error in get_player_data: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


@app.route("/team-data/<team_name>")
def team_data(team_name):
    # Convert full team name to abbreviation using the existing dictionary
    team_abv = team_abbreviations.get(team_name)
    
    if not team_abv:
        return jsonify({
            'success': False,
            'error': f'Team abbreviation not found for {team_name}'
        })
    
    # Fetch and return the player data
    result = get_player_data(team_abv)
    return jsonify(result)


    
@app.route("/player-stats/<player_id>")
def player_stats(player_id):
    headers = {
        "x-rapidapi-key": "a0f0cd0b5cmshfef96ed37a9cda6p1f67bajsnfcdd16f37df8",
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }
    
    # First get player info
    info_url = "https://tank01-fantasy-stats.p.rapidapi.com/getNFLPlayerInfo"
    info_querystring = {"playerID": player_id}
    
    # Then get game stats
    games_url = "https://tank01-fantasy-stats.p.rapidapi.com/getNFLGamesForPlayer"
    games_querystring = {
        "playerID": player_id,
        "season": "2024",
    }
    
    try:
        # Get both responses
        info_response = requests.get(info_url, headers=headers, params=info_querystring)
        games_response = requests.get(games_url, headers=headers, params=games_querystring)
        
        info_data = info_response.json()
        games_data = games_response.json()
        
        if info_data.get('statusCode') == 200 and games_data.get('statusCode') == 200:
            # Process games data
            games = list(games_data['body'].values())
            games.sort(key=lambda x: x['gameID'], reverse=True)
            recent_games = games[:10]
            
            # Get player info
            player_info = info_data['body']
            
            # Format injury info
            injury_status = "Healthy"
            if player_info.get('injury'):
                injury_info = player_info['injury']
                injury_status = injury_info

            # Combine and return all data
            return jsonify({
                'success': True,
                'games': recent_games,
                'player': {
                    'name': player_info.get('longName'),
                    'position': player_info.get('pos'),
                    'number': player_info.get('jerseyNum'),
                    'height': player_info.get('height'),
                    'weight': player_info.get('weight'),
                    'team': player_info.get('team'),
                    'college': player_info.get('college'),
                    'experience': player_info.get('exp'),
                    'headshot': player_info.get('nbaComHeadshot'),
                    'injury': injury_status
                }
            })
            
        return jsonify({
            'success': False,
            'error': 'Failed to fetch player data'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

        
team_abbreviations = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS'
}

if __name__ == '__main__':
    # Windows-specific fixes
    if sys.platform.startswith('win'):
        # Disable Flask's reloader on Windows to avoid socket issues
        app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)
    else:
        # Normal operation for other platforms
        app.run(debug=True, host='127.0.0.1', port=5000)