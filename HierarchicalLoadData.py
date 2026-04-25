import json
import pandas as pd
from NHLGameState import NHLGameState

def load_h_data(json_files):
    # Arizona, Utah Hockey Club, and Utah Mammoth are all the same team, so the multiple IDs should be used as one.
    TEAM_ALIASES = {59: 53, 68: 53}
    all_events = []
    
    all_tids = set()
    for fp in json_files:
        with open(fp, 'r') as f:
            data = json.load(f)
            for g in data:
                h_id = int(g['homeTeam']['id'])
                a_id = int(g['awayTeam']['id'])
                all_tids.add(TEAM_ALIASES.get(h_id, h_id))
                all_tids.add(TEAM_ALIASES.get(a_id, a_id))
    
    team_mapping = {int(tid): i for i, tid in enumerate(sorted(list(all_tids)))}
    
    for alias, main_id in TEAM_ALIASES.items():
        if main_id in team_mapping:
            team_mapping[alias] = team_mapping[main_id]
    
    # Persist mapping so inference and simulation share identical team codes.
    with open('h_team_mapping.json', 'w') as f:
        json.dump(team_mapping, f)

    for fp in json_files:
        with open(fp, 'r') as f:
            games_data = json.load(f)
            
        for game in games_data:
            game_id = game.get('id')
            g_date = game.get('gameDate')
            h_id, a_id = int(game['homeTeam']['id']), int(game['awayTeam']['id'])
            
            prev_time = 3600
            curr_h_score, curr_a_score = 0, 0
            
            for play in game.get('plays', []):
                details = play.get('details', {})
                
                state = NHLGameState(play, h_id, a_id, curr_h_score, curr_a_score)
                vector = state.get_state_vector()
                
                vector['game_id'] = game_id
                vector['game_date'] = g_date
                vector['home_team_id'] = h_id
                vector['away_team_id'] = a_id
                vector['home_score'] = curr_h_score
                vector['away_score'] = curr_a_score
                vector['h_team_code'] = team_mapping[h_id]
                vector['a_team_code'] = team_mapping[a_id]
                vector['duration_seconds'] = max(0, prev_time - vector['time_remaining'])
                
                type_key = play.get('typeDescKey', '')
                vector['is_penalty'] = 1 if type_key == 'penalty' else 0
                vector['is_h_goal'], vector['is_a_goal'] = 0, 0
                
                if type_key == 'goal':
                    scorer = details.get('eventOwnerTeamId')
                    if scorer == h_id:
                        vector['is_h_goal'] = 1
                        curr_h_score += 1
                    else:
                        vector['is_a_goal'] = 1
                        curr_a_score += 1
                
                all_events.append(vector)
                prev_time = vector['time_remaining']
                
    return pd.DataFrame(all_events), team_mapping