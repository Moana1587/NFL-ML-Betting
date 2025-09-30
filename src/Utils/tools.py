import re
from datetime import datetime

import pandas as pd
import requests

from .Dictionaries import team_index_current

games_header = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/57.0.2987.133 Safari/537.36',
    'Dnt': '1',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en',
    'origin': 'https://site.api.espn.com',
    'Referer': 'https://www.espn.com/'
}

data_headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Host': 'stats.nfl.com',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nfl.com/',
    'Connection': 'keep-alive'
}


def get_json_data(url):
    raw_data = requests.get(url, headers=data_headers)
    try:
        json = raw_data.json()
    except Exception as e:
        print(e)
        return {}
    # ESPN API returns data directly, not in resultSets format
    return json


def get_todays_games_json(url):
    raw_data = requests.get(url, headers=games_header)
    json = raw_data.json()
    return json.get('events', [])


def to_data_frame(data):
    try:
        # ESPN NFL teams API returns data in specific format
        if isinstance(data, dict) and 'sports' in data:
            teams_data = []
            for sport in data['sports']:
                if sport.get('name') == 'Football':
                    for league in sport.get('leagues', []):
                        if league.get('name') == 'NFL':
                            for team in league.get('teams', []):
                                team_info = team.get('team', {})
                                # Extract basic team stats - you may need to adjust based on actual API response
                                team_data = {
                                    'TEAM_ID': team_info.get('id', ''),
                                    'TEAM_NAME': team_info.get('displayName', ''),
                                    'WINS': 0,  # These would need to be fetched from stats API
                                    'LOSSES': 0,
                                    'TIES': 0,
                                    'PCT': 0.0,
                                    'POINTS_FOR': 0,
                                    'POINTS_AGAINST': 0,
                                    'POINT_DIFF': 0
                                }
                                teams_data.append(team_data)
            return pd.DataFrame(teams_data)
        elif isinstance(data, list) and len(data) > 0:
            return pd.DataFrame(data=data)
        else:
            return pd.DataFrame(data={})
    except Exception as e:
        print(e)
        return pd.DataFrame(data={})


def create_todays_games(input_list):
    games = []
    for game in input_list:
        if game.get('status', {}).get('type', {}).get('name') == 'STATUS_SCHEDULED':
            competitions = game.get('competitions', [])
            if competitions:
                competitors = competitions[0].get('competitors', [])
                if len(competitors) >= 2:
                    home_team = competitors[0].get('team', {}).get('displayName', '')
                    away_team = competitors[1].get('team', {}).get('displayName', '')
                    if home_team and away_team:
                        games.append([home_team, away_team])
    return games


def create_todays_games_from_odds(input_dict):
    games = []
    for game in input_dict.keys():
        home_team, away_team = game.split(":")
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        games.append([home_team, away_team])
    return games


def get_date(date_string):
    year1, month, day = re.search(r'(\d+)-\d+-(\d\d)(\d\d)', date_string).groups()
    year = year1 if int(month) > 8 else int(year1) + 1
    return datetime.strptime(f"{year}-{month}-{day}", '%Y-%m-%d')
