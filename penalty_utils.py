import json

import numpy as np

def estimate_home_penalty_share(json_files, default_share=0.52, min_events=50):
    """Estimate probability that a future penalty gives home team a power play.

    Uses penalty events where details.eventOwnerTeamId is present.
    Interprets eventOwnerTeamId as the penalized team ID.
    """
    home_adv = 0
    away_adv = 0

    for file_path in json_files:
        with open(file_path, 'r') as f:
            games = json.load(f)

        for game in games:
            home_id = game['homeTeam']['id']
            away_id = game['awayTeam']['id']

            for play in game.get('plays', []):
                if play.get('typeDescKey') != 'penalty':
                    continue

                penalized_team = play.get('details', {}).get('eventOwnerTeamId')
                if penalized_team is None:
                    continue

                if penalized_team == away_id:
                    home_adv += 1
                elif penalized_team == home_id:
                    away_adv += 1

    total = home_adv + away_adv
    if total < min_events:
        return float(default_share)

    share = (home_adv + 1.0) / (total + 2.0)
    return float(np.clip(share, 1e-3, 1 - 1e-3))
