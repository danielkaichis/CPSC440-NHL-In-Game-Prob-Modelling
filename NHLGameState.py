class NHLGameState:
    def __init__(self, play_data, home_team_id, away_team_id, 
                 home_strength, away_strength, home_advantage):
        """
        Initialize from a single play's data 
        """
        self.play_data = play_data
        self.home_team_id = home_team_id
        self.away_team_id = away_team_id
        
        # Core state from play data
        self.time_remaining = play_data['timeRemaining']
        self.period = play_data['periodDescriptor']['number']
        self.situation_code = play_data.get('situationCode')
        
        # Scores and shots (already cumulative in data)
        details = play_data.get('details', {})
        self.home_score = details.get('homeScore', 0)
        self.away_score = details.get('awayScore', 0)
        self.home_sog = details.get('homeSOG', 0)
        self.away_sog = details.get('awaySOG', 0)
        
        # Pre-computed strength metrics
        self.home_strength = home_strength
        self.away_strength = away_strength
        self.home_advantage = home_advantage
    
    def get_manpower_state(self):
        """Decode situation code to human-readable format.
        
        NHL Situation Codes:
        - 1551: 5v5 (even strength, most common)
        - 1541: 5v4 (one team has power play)
        - 1451: 4v5 (other team has power play)
        - 1560: 5v5 with empty net
        - 0651: Special situations (double penalty, shootout, etc.)
        """
        if not self.situation_code:
            return 'unknown'
        
        # Convert to int if it's a string (JSON stores numbers as strings sometimes)
        try:
            code_int = int(self.situation_code)
        except (ValueError, TypeError):
            return f'unknown_{self.situation_code}'
        
        # Known situation code mappings
        situation_map = {
            1551: '5v5',      # Even strength
            1541: '5v4',      # Power play (one team)
            1451: '4v5',      # Power play (other team)
            1560: '5v5_EN',   # Empty net
            651: 'special'    # Double penalty, shootout, etc.
        }
        
        return situation_map.get(code_int, f'unknown_{code_int}')
    
    def get_score_differential(self):
        """Home score minus away score."""
        return self.home_score - self.away_score
    
    def get_state_vector(self):
        """Return state for modeling."""
        return {
            "time_remaining": self.time_remaining,
            "period": self.period,
            "score_differential": self.get_score_differential(),
            "manpower_state": self.get_manpower_state(),
            "home_sog": self.home_sog,
            "away_sog": self.away_sog,
            "home_strength": self.home_strength,
            "away_strength": self.away_strength,
            "home_advantage": self.home_advantage,
            "situation_code": self.situation_code
        }

    def get_time_seconds(self):
        """Convert 'MM:SS' string from timeRemaining to total seconds remaining."""

        if isinstance(self.time_remaining, str):
            try:
                minutes, seconds = map(int, self.time_remaining.split(':'))
                return minutes * 60 + seconds
            except ValueError:
                return None
        return None
    
    def get_game_time_remaining(self):
        """Calculate total game time remaining in seconds."""
        seconds_remaining = self.get_time_seconds()
        if seconds_remaining is None:
            return None
        
        # NHL games have 3 periods of 20 minutes each (1200 seconds)
        total_game_seconds = 3 * 20 * 60
        
        # Calculate elapsed time based on current period and time remaining
        elapsed_time = (self.period - 1) * 20 * 60 + (total_game_seconds - seconds_remaining)
        
        return total_game_seconds - elapsed_time



    