"""
compare the reviewed transcripts with the another set of labelings

"""

import os
import csv
import glob
import pandas as pd
import difflib
import re


def get_out_of_order_rows(df, round_column_name):
    """
    Identifies and returns rows where the round number is out of order in the specified column.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - round_column_name: The name of the column in the DataFrame that contains round numbers.

    Returns:
    - A DataFrame containing only the rows where the round numbers are out of order.
    """
    # Extract round numbers as integers from the specified column
    round_numbers = df[round_column_name].apply(lambda x: int(x.replace('round', '')))

    # Find indices where the round number is greater than the round number in the next row
    out_of_order_indices = round_numbers[round_numbers.diff() < 0].index

    # Return rows that are out of order
    # Since diff() leads to checking current row against next, we include next row by index+1 if it's part of the sequence
    return df.loc[out_of_order_indices.union(out_of_order_indices + 1).intersection(df.index)]


def find_similar_transcripts(old_file, new_file, similarity_threshold=0.7):
    # Filter old_file where 'To' and 'Action' are not empty
    filtered_old_df = old_file[old_file['To'].notna() & old_file['Action'].notna()]

    #also from not equal to to
    filtered_old_df = filtered_old_df[filtered_old_df['From'] != filtered_old_df['To']]

    matches = []  # To store the matches with details

    # Iterate through each row in the filtered_old_df
    for old_index, old_row in filtered_old_df.iterrows():

        # Find matching rows in new_file based on 'speaker' and 'round'
        matching_rows_new_df = new_file[
            (new_file['speaker'] == old_row['speaker']) & (new_file['round'] == old_row['round'])]

        # For each potential match, calculate similarity for 'trans'
        for new_index, new_row in matching_rows_new_df.iterrows():
            # Calculate the similarity score
            similarity = difflib.SequenceMatcher(None, old_row['trans'], new_row['trans']).ratio()

            # Check if similarity is above the threshold
            if similarity >= similarity_threshold:
                # If a match is found, store the details
                matches.append((
                               new_index,  new_row['round'], new_row['speaker'], old_row['From'], new_row['trans'],
                               new_row['raw_labels'], old_row['To'], old_row['Action'], old_row['trans'], similarity,
                               old_index))

    # Return matches as a DataFrame with specified column names
    return pd.DataFrame(matches,
                        columns=['New Index', 'Round', 'Speaker', 'From', 'New Transcript', 'New Labels', 'To', 'Action',
                                 'Old Transcript', 'Similarity', 'Old Index'])


# Ensure you have the correct column names in your old_file and new_file DataFrames when using this function.

# read the previous labels
old_data_dir = '/Users/saiyingge/**Research Projects/Nomination-network/data/Cleaned_2021/'
old_game_names = pd.read_excel(old_data_dir + 'Game_process_track.xlsx')['GameName'].tolist()

# read the new labels
new_data_dir = '/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/Data/Private_Data/Reviewed/'
new_game_file = pd.read_csv(new_data_dir + 'combined_labels_w_transcripts.csv')

# read the game names from the csv file
new_game_names = new_game_file['game'].unique().tolist()

# # for each file in the old fold find the one in new folder
# for game_name in old_game_names:
#     if game_name in new_game_names:
#         print(game_name)

# for game_name in old_game_names:
#     if game_name in new_game_names:
game_name="018AZ"
print(f"Processing game: {game_name}")
        # Read the old and new files
old_file_path = f"{old_data_dir}{game_name}.xls"
old_file = pd.read_excel(old_file_path, engine='xlrd')
new_file = new_game_file[new_game_file['game'] == game_name]

# reset the index of the new file
new_file = new_file.reset_index(drop=True)

# change the round and transcript column name to match the new file
old_file = old_file.rename(columns={
    'roundname': 'round',
    'newphrase': 'trans'
})

# change round to lowercase, and change speaker to int
old_file['round'] = old_file['round'].str.lower()
# modify speaker to int
new_file['speaker'] = new_file['speaker'].astype(int)

# Check for non-numeric values in 'speaker'
is_numeric = pd.to_numeric(old_file['speaker'], errors='coerce').notna()
non_numeric_entries = old_file[~is_numeric]

print("Non-numeric entries in 'speaker' column:")
print(non_numeric_entries)
# Replace non-numeric values with -1 (or another placeholder value of your choice)
old_file.loc[~is_numeric, 'speaker'] = -1

# Now attempt conversion to int
old_file['speaker'] = old_file['speaker'].astype(int)

# drop roundname that is Intro, not contain "round"
# old_file = old_file[old_file['round'].str.contains("round")]

# check if the rounds are in order
out_of_order_new = get_out_of_order_rows(new_file, 'round')
if not out_of_order_new.empty:
    print(out_of_order_new)
else:
    print(f"No out of order rounds found in {game_name}")

# Find marked in old file that match to new file
matches = find_similar_transcripts(old_file, new_file)

print(f"Updated game: {game_name}")

#
# # for 002usp,005NTU
# # keep indx round speaker trans and game
# # drop the rest
#
# old_file_to_save= old_file[['round', 'speaker','trans','From',"To","Action"]]
# #add indx and game
# old_file_to_save['indx'] = old_file.index
# old_file_to_save['game'] = game_name
# #change the column order
# old_file_to_save = old_file_to_save[['indx', 'round', 'speaker', 'trans', 'game', 'From', 'To', 'Action']]
# # save the file
# old_file_to_save.to_csv(f"{new_data_dir}{game_name}_fix.csv", index=False)
