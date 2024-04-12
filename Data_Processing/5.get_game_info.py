'''
This script is used to get the game information, including the game name, the number of players, the number of rounds,
the game result, and the number of spy and villager in each game. The game information is saved to a csv file.
'''
import pandas as pd

Code_dir = '/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/'
# get the player(nodes) information
game_nodes = pd.read_csv(Code_dir + 'Data/all_nodes.csv')

# get game info, including the game name, the number of players, and the number of rounds
# aggregate the game info by game name, include game result
game_info = game_nodes.groupby('game_name').agg(
    {'max_round': 'max', 'Player_Number': 'count', 'game_result': 'first','homogeneous':'first'}).reset_index()
# include count the number of spy and villager in each game
game_info = game_info.merge(game_nodes.groupby(['game_name', 'Game_Role']).size().unstack().reset_index(),
                            on='game_name')
#change the column names
game_info.columns = ['game_name', 'max_round', 'num_players', 'game_result','homogeneous', 'Num0fSpy', 'Num0fVillager']

# change the game round of 006AZ to 4 and 013HK to 5 , since we only have the transcript for 4 rounds
game_info.loc[game_info['game_name'] == '006AZ', 'max_round'] = 4
game_info.loc[game_info['game_name'] == '013HK', 'max_round'] = 5

#save the game_info to a csv file
game_info.to_csv(Code_dir + 'Data/' + 'game_info.csv', index=False)

