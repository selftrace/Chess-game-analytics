# Chess Game Analytics
from raw data to predicting wins

So basically I took this massive Lichess dataset, stopped it from looking like a mess, engineered some actually useful features, threw a bunch of graphs at it, and then trained a model to guess who's gonna win.

Here’s the full chaos breakdown of how it went down.
1. Data cleaning & storytelling (the classic boring cleanup)
First off, raw data is always a headache. Had to unzip everything, check the rows, look for missing values, and handle dupes so the stats wouldn't be totally fake.
Unzipped and ingested the CSV straight into Pandas.
Ran standard checks (info(), isnull(), duplicated()) so nothing weird breaks later.
Did an IQR sweep to catch weird outlier games that make no sense.

2. Feature engineering (Making the data actually useful, you know)
The default columns weren't cutting it, so I cooked up a few custom features to actually give the data some context:
Timestamps to normal time: Converted created_at and last_move_at from epoch timestamps into actual datetime format. Now we can see when people are playing and how long games actually last in minutes.
Splitting increment codes: Took things like "15+2" and split them into base_time (15) and increment (2). Now we can group games by Blitz, Rapid, Classical, etc.
Rating differential: Computed rating_diff = white_rating - black_rating. Simple math, but literally the strongest feature to tell if a game is a total mismatch or a sweatfest.

Python
# Quick snippet of feature creation
import pandas as pd

# Datetime fix
df['created_at_dt'] = pd.to_datetime(df['created_at'], unit='ms')
df['game_duration_min'] = (df['last_move_at'] - df['created_at']) / (1000 * 60)

# Rating diff
df['rating_diff'] = df['white_rating'] - df['black_rating']

# Increment split
df[['base_time', 'increment']] = df['increment_code'].str.split('+', expand=True).astype(int)

3. Expanded EDA (Visual chaos)
Instead of just printing giant text tables nobody reads, plotted everything out to see what’s actually happening on the board:
Opening win-rates: Plotted top openings vs. who actually wins (does White opening advantage even matter at normal ELOs?).
Rating distributions: Visualized ELO spread to see where most players cluster.
Game lengths: Boxplots and histograms comparing game duration (turns and time) against victory_status (resign vs. mate vs. outoftime).

4. Machine learning (Predicting the winner)
Toss all the cleaned features into a baseline classification model to predict winner (White / Black / Draw).
Model: Random Forest Classifier / XGBoost baseline.
Features used: rating_diff, white_rating, black_rating, base_time, turns, encoded opening features.
Metrics: Accuracy score, confusion matrix, and feature importance to see what the model actually cares about (spoiler: rating gap carries hard).

5. Key Insights & Takeaways
Rating Gap is King: If the ELO difference is >150 points, opening choice barely saves you. The model relies heavily on rating_diff.

Time Scrambles: Short base_time games (1-3 min) end on outoftime way more often than resignations, while longer Rapid games almost always end in resignation or mate.

White Advantage is Real (Sort of): White maintains a subtle win-rate edge overall, but it’s much more noticeable in middle-to-high ELO brackets than lower ones.

Opening Traps: Certain sharp openings yield insanely high win rates in lower ELOs simply because people don't know the refutations yet.
