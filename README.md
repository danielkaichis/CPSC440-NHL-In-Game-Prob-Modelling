# CPSC440-NHL-In-Game-Prob-Modelling
Repo for final UBC CPSC 440 project

Code prefixed with "league" is for the model that estimates league wide scoring rates.

Code prefixed with "team" is for the model that estimates team-specific scoring rates.

## Usage
First install dependencies
```
pip install -r requirements.txt
```

To run the baseline model that estimates league wide scoring rates, from the main folder run:
```
# fit model
python3 league_advi.py
# evaluate model by running monte carlo simulation
python3 league_eval_model.py
```

To run the model that estimates team-specific scoring rates, from the main folder run:
```
# fit model
python3 -m team_models.team_advi
# evaluate model by running monte carlo simulation
python3 -m team_models.team_eval_model
```

To run the team-specific model that is retrained on 20 day windows to factor in recent team performance, run:
```
# fit model and run monte carlo simulations
python3 -m team_models.team_20_day_model
```

To generate plots showing the Brier score, Logistic Loss, and Accuracy of each model once evaluated, run:
```
python3 plot_results.py
``` 
