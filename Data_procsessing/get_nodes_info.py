import pandas as pd
from sqlalchemy import create_engine


# Directory path
code_dir = '/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/'

# List of homogeneous game names
homo_game_names = ["001ISR", "002USP", "003HK", "003NTU", "005USP", "006AZ", "006NTU", "006USP", "006ZAM", "007HK", "007NTU", "009SB", "010NTU", "010UMD", "011NTU", "011SB", "012HK", "015ZAM"]

# Database connection details
# read from other files

# Connect to the MySQL database
engine = create_engine(f"mysql+mysqlconnector://{dbUsername}:{dbPassword}@{dbHostname}:{port}/{dbName}")

# Read data from the database into pandas DataFrame
play_b4_query = 'SELECT Player_ID, play_b4 FROM postreport'
play_b4 = pd.read_sql(play_b4_query, engine)

player_query = 'SELECT Player_ID, Player_Number, Game_Role, game_name, WinLose, sex, Eng_nativ FROM player'
player = pd.read_sql(player_query, engine)

# Merge DataFrames
player = pd.merge(player,play_b4,  on='Player_ID',how='left')

# Add missing data
missing_data = {'Player_ID':'na','play_b4':'', 'Player_Number': "3", 'Game_Role': "Villager", 'game_name': "002USP", 'WinLose': "Lose", 'sex': "Female", 'Eng_nativ': "not native speaker"}
player = pd.concat([player, pd.DataFrame(missing_data, index=[0])], ignore_index=True)


##game info

# Read game names from CSV file
game_info = pd.read_csv(code_dir + 'Data/game_names.csv', header=None)

# Add homogeneous column based on game
game_info['homogeneous'] = game_info[0].isin(homo_game_names).replace({True: 'Yes', False: 'No'})
# rename the game_info column
game_info.columns=['game_name','homogeneous']

# Query for round numbers
round_number_query = 'SELECT gamename, predicted_round FROM app_team_rounds'
round_number = pd.read_sql(round_number_query, engine)

# Group by and get max round
round_number = round_number.groupby('gamename').agg(max_round=('predicted_round', 'max')).reset_index()
#change gamename to game_name
round_number.columns=['game_name','max_round']

# Cleaning game names
round_number['game_name'] = round_number['game_name'].str.upper().str.strip()
game_info['game_name'] = game_info['game_name'].str.upper().str.strip()
player['game_name'] = player['game_name'].str.upper().str.strip()

# combine game_info and round_number
game_info=pd.merge(game_info,round_number, on='game_name',how='left')
# Filter player DataFrame based on game names
player = player[player['game_name'].isin(game_info['game_name'])]

# Merge player with game name information
player = pd.merge(player, game_info, on='game_name')

# Drop the row names equivalent in pandas
player.reset_index(drop=True, inplace=True)

# Write to CSV
player.to_csv(code_dir+"Data/all_nodes.csv", index=False)

print("Data processing complete.")

# run statistics for the player data
# total number of players
print("total number of players:",len(player))
# total number of games
print("total number of games:",len(player['game_name'].unique()))
# Convert Player_Number to numeric if it's not already
player['Player_Number'] = pd.to_numeric(player['Player_Number'], errors='coerce')
# Convert Player_Number to numeric if it's not already
player['Player_Number'] = pd.to_numeric(player['Player_Number'], errors='coerce')

# Group by game_name and calculate the maximum Player_Number for each game
max_players_per_game = player.groupby('game_name')['Player_Number'].max()

# Ensure all values in max_players_per_game are numeric
max_players_per_game = pd.to_numeric(max_players_per_game, errors='coerce')

# Now calculate the mean, min, max, and standard deviation
mean_players = max_players_per_game.mean()
min_players = max_players_per_game.min()
max_players = max_players_per_game.max()
std_players = max_players_per_game.std()

print(f"Mean number of players in each game: {mean_players}")
print(f"Minimum number of players in each game: {min_players}")
print(f"Maximum number of players in each game: {max_players}")
print(f"Standard deviation of number of players in each game: {std_players}")

# max_rounds counts
player['max_round'].value_counts()

# game roles counts
player['Game_Role'].value_counts()

#winlose counts
player['WinLose'].value_counts()

#sex counts
player['sex'].value_counts()

#eng_nativ counts
player['Eng_nativ'].value_counts()

#play_b4 counts
player['play_b4'].value_counts()

#homogeneous counts
player['homogeneous'].value_counts()

player.describe()