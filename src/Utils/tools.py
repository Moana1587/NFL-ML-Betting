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
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'en-US,en;q=0.9',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
    'Referer': 'https://gist.github.com/',
    'Sec-Ch-Ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'cross-site',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36'
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
