import pandas as pd
import json

from NHLGameState import NHLGameState

def load_game_data(file_path):
    """ Loads a single JSON file of NHL game data and transforms it into a pandas DataFrame of events."""
    with open(file_path, 'r') as f:
        games_data = json.load(f)
        
    all_events = []
    
    for game in games_data:
        game_id = game.get('id')
        plays = game.get('plays', [])
        home_team_id = game['homeTeam']['id']
        away_team_id = game['awayTeam']['id']
        
        prev_time = 3600
        current_home_score = 0
        current_away_score = 0
        
        for play in plays:
            details = play.get('details', {})
                
            state = NHLGameState(
                play, home_team_id, away_team_id, 
                current_home_score, current_away_score)
            
            vector = state.get_state_vector()
            current_time = vector['time_remaining']
            vector['game_id'] = game_id
            
            # Exposure for this row is elapsed time since the previous event.
            vector['duration_seconds'] = max(0, prev_time - current_time)
            
            is_goal = play.get('typeDescKey') == 'goal'
            if is_goal:
                scoring_team = details.get('eventOwnerTeamId')
                if scoring_team == home_team_id:
                    vector['is_home_goal'] = 1
                    vector['is_away_goal'] = 0
                    current_home_score += 1
                elif scoring_team == away_team_id:
                    vector['is_home_goal'] = 0
                    vector['is_away_goal'] = 1
                    current_away_score += 1
            else:
                vector['is_home_goal'] = 0
                vector['is_away_goal'] = 0
            
            is_penalty = play.get('typeDescKey') == 'penalty'
            vector['is_penalty'] = 1 if is_penalty else 0
                
            all_events.append(vector)
            prev_time = current_time
            
    return pd.DataFrame(all_events)