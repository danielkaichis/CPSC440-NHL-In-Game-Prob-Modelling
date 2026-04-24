class NHLGameState:
    def __init__(self, play_data, home_team_id, away_team_id, current_home_score, current_away_score):
        """
        Initialize from a single play's data 
        """
        self.play_data = play_data
        self.home_team_id = home_team_id
        self.away_team_id = away_team_id
        
        self.time_remaining = play_data['timeRemaining']
        self.period = play_data['periodDescriptor']['number']
        self.situation_code = play_data.get('situationCode')
        
        self.home_score = current_home_score
        self.away_score = current_away_score
    
    def get_manpower_state(self):
        if not self.situation_code: return 'unknown'
        
        situation_str = str(self.situation_code).zfill(4)
        
        if len(situation_str) != 4: return 'special'
            
        away_goalie = int(situation_str[0])
        away_skaters = int(situation_str[1])
        home_skaters = int(situation_str[2])
        home_goalie = int(situation_str[3])
        
        if home_goalie == 0: return 'home_empty_net'
        if away_goalie == 0: return 'away_empty_net'
            
        skater_diff = home_skaters - away_skaters
        
        if skater_diff == 0:
            if home_skaters == 5: return '5v5'
            if home_skaters == 4: return '4v4'
            if home_skaters == 3: return '3v3' # Overtime
        elif skater_diff == 1: 
            return 'home_PP_1' # 5v4 and 4v3
        elif skater_diff == 2: 
            return 'home_PP_2' # 5v3
        elif skater_diff == -1: 
            return 'away_PP_1' # 4v5 and 3v4
        elif skater_diff == -2: 
            return 'away_PP_2' # 3v5
            
        return 'special'
    
    def get_score_differential(self):
        """Home score minus away score."""
        return self.home_score - self.away_score
    

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
        seconds_in_period = self.get_time_seconds()
        if seconds_in_period is None:
            return 0
        
        return (3 - self.period) * 1200 + seconds_in_period

    def get_state_vector(self):
        """Return state for modeling."""
        return {
            "time_remaining": self.get_game_time_remaining(),
            "period": self.period,
            "score_differential": self.get_score_differential(),
            "manpower_state": self.get_manpower_state(),
            "home_score": self.home_score,
            "away_score": self.away_score,
            "situation_code": self.situation_code
        }
