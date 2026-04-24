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
        
        # Reset game states at puck drop
        prev_time = 3600
        current_home_score = 0
        current_away_score = 0
        current_home_sog = 0
        current_away_sog = 0
        
        for play in plays:
            details = play.get('details', {})
            
            # update running totals for shots on goal if available in the play details
            if 'homeSOG' in details:
                current_home_sog = details['homeSOG']
            if 'awaySOG' in details:
                current_away_sog = details['awaySOG']
                
            # instantiate state object with current cumulative scores and SOG
            state = NHLGameState(
                play, home_team_id, away_team_id, 
                current_home_score, current_away_score, 
                current_home_sog, current_away_sog)
            
            vector = state.get_state_vector()
            current_time = vector['time_remaining']
            vector['game_id'] = game_id
            
            # Calculate how long the previous state lasted before this new event
            vector['duration_seconds'] = max(0, prev_time - current_time)
            
            # check for goals and update scores accordingly
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