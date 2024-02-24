"""
Extract links from raw labels.

This script processes the raw labels from a CSV file, categorizes them based on sentiment,
and extracts 'from', 'to', and 'sentiment_score' for each link.


input:  the labels
output: the csv file that saved all the linked information

"""

import pandas as pd

# remove the warning
pd.options.mode.chained_assignment = None  # default='warn'

# Directory path
Code_dir = '/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/'

# Read the raw labels from CSV
raw_labels = pd.read_csv(Code_dir + 'Data/Private_Data/Reviewed/combined_labels_w_transcripts.csv')

# Remove rows with empty labels and the 'trans' column
raw_labels = raw_labels[raw_labels['raw_labels'].notna()].drop(columns=['trans'])


# Function to categorize labels and extract link information
def categorize_and_update_row(row):
    """
    Categorize labels and extract 'from', 'to', and sentiment score from each row.

    :param row: A pandas Series representing a row of the DataFrame.
    :return: A DataFrame with extracted link information for the given row.
    """
    # row=raw_labels.iloc[0]
    positive_labels = {'AG', 'TT', 'NM', 'DF'}
    negative_labels = {'DAG', 'DTT', 'SPY', 'CH'}

    # DataFrame to store links for
    links = pd.DataFrame(columns=['from', 'to', 'sentiment_score', 'round', 'game', 'indx'])

    # Split and clean the labels
    labels = row['raw_labels'].translate(str.maketrans('', '', '[]\'{} ')).split(',')

    for label in labels:
        # Extract the digit (representing 'to' speaker) from the label
        digits = ''.join(filter(str.isdigit, label))
        if digits:
            to_speaker = int(digits)
            # Determine sentiment based on the label prefix
            sentiment = 1 if any(label.startswith(pos_label) for pos_label in positive_labels) else -1 if any(
                label.startswith(neg_label) for neg_label in negative_labels) else 0

            # Create a dictionary for each link and append to the list
            link = {'from': row['speaker'], 'to': to_speaker, 'sentiment_score': sentiment, 'round': row['round'],
                    'game': row['game'], 'indx': row['indx']}
            links = pd.concat([links, pd.DataFrame(link, index=[0])], ignore_index=True)

            # drop duplicates
            # if the same sentiment to the same person is recorded multiple times, only keep one
            links = links.drop_duplicates()


    return links


# Initialize DataFrame to store all links
all_links = pd.DataFrame()

# Apply the function to each row and concatenate the results
for index, row in raw_labels.iterrows():
    links = categorize_and_update_row(row)
    all_links = pd.concat([all_links, links], ignore_index=True)

# save the links to a csv file
all_links.to_csv(Code_dir + 'Data/all_links.csv', index=False)

# find outliers
# show the number of links in each game, sort by the number of links
print(all_links['game'].unique())
print(all_links.groupby('game').count() )
#find nan in game
# all_links[all_links['game'].isna()]
# print(all_links.groupby('game').count().sort_values(by='from', ascending=False))

# show the average number of links
print(all_links.groupby('game').count().mean())

#check the game name not in the game name list from a file
# game_name_file_path = '/Users/saiyingge/**Research Projects/Nomination-network/data/Cleaned_2021/Game_process_track.xlsx'
# old_game_names = pd.read_excel(game_name_file_path)['GameName'].tolist()
# new_game_names = all_links['game'].unique().tolist()
# for game in new_game_names:
#     if game not in old_game_names:
#         print(game)

# print(all_links.groupby('game').count())
all_links.groupby('from').count()
all_links.groupby('to').count()

# find the outlier
# all_links[all_links['to'] == 24]
#find nan in links
