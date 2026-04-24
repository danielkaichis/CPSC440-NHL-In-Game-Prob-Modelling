import requests
import time
import json
import os

def download_dynamic_seasons(seasons, base_dir="nhl_raw_data"):
    """
    Downloads play-by-play data until the season ends, then dynamically splits
    the total gathered games into two equal files.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for season_year in seasons:
        print(f"\n{'='*50}")
        print(f"Scraping the {season_year}-{season_year+1} season (Dynamic Split)")
        print(f"{'='*50}")
        
        season_data = []
        consecutive_404s = 0
        game_num = 1 # Start at Game 1
        
        # Keep scraping forever until we trigger the break condition
        while True:
            game_id = f"{season_year}02{str(game_num).zfill(4)}"
            url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
            
            try:
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    data['game_id'] = game_id 
                    season_data.append(data)
                    
                    print(f"[{game_id}] Downloaded.")
                    consecutive_404s = 0 # Reset our failure counter
                    
                elif response.status_code == 404:
                    print(f"[{game_id}] Not Found (404).")
                    consecutive_404s += 1
                    
                    # We wait for 5 consecutive missing games to confirm the season has ended.
                    # This protects against weird mid-season postponements or cancellations.
                    if consecutive_404s >= 5:
                        print(f"\nHit {consecutive_404s} missing games in a row. End of schedule detected.")
                        break 
                else:
                    print(f"[{game_id}] Failed - Status Code: {response.status_code}")
                    
            except Exception as e:
                print(f"[{game_id}] Error: {e}")
                
            game_num += 1
            
        # split file to meet github size limits
        total_games = len(season_data)
        
        if total_games > 0:
            midpoint = total_games // 2
            
            part1_file = os.path.join(base_dir, f"{season_year}_{season_year+1}_part1.json")
            part2_file = os.path.join(base_dir, f"{season_year}_{season_year+1}_part2.json")
            
            print(f"\nTotal valid games collected for {season_year}: {total_games}")
            print(f"Splitting dynamically at game {midpoint}...")
            
            # Slice the first half and save
            with open(part1_file, 'w') as f:
                json.dump(season_data[:midpoint], f)
            print(f"Saved Part 1 -> {part1_file} ({len(season_data[:midpoint])} games)")
            
            # Slice the second half and save
            with open(part2_file, 'w') as f:
                json.dump(season_data[midpoint:], f)
            print(f"Saved Part 2 -> {part2_file} ({len(season_data[midpoint:])} games)")
            
        else:
            print(f"\nNo data found for the {season_year} season.")

if __name__ == "__main__":
    target_seasons = [2023]
    print("Initializing dynamic scraper...")
    download_dynamic_seasons(target_seasons)
