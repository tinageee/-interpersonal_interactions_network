"""
This script processes game data to extract and consolidate labels from different sources. It performs the following steps:

1. Reads a list of game names from a CSV file.
2. For each game name, it finds the corresponding reviewed file and loads its data.
3. Applies a custom function 'compare_label_sets' to each row of the data:
   - This function extracts labels from 'matched_labels' and consolidated label columns.
   - It checks if each label ends with a digit and includes it in the set if so.
   - It also checks if the label's prefix is in a predefined list of acceptable prefixes.
4. The resulting set of labels is then combined with the first four columns of the original data.
5. The combined data is saved to a CSV file for further use.

input: raw consolidated labels from the reviewed files
output: combined_labels_w_transcripts.csv
"""

import os
import csv
import glob
import pandas as pd
import ast
import re


# def compare_label_sets(row):
#     """
#     This function extracts labels from 'matched_labels' and consolidated label columns.
#     # Extract labels from the matched_labels and consolidated labels, put them into a set for each raw
#     # checked if the label ends with a digit, if so, add it to the set
#     # if the label is not in the acceptable list, print the row
#     :param row:     Each row of the raw_labels
#     :return:    The row with a set of labels, or NaN if the set is empty, or the row if the label is not in the acceptable list
#     """
#
#     # row=raw_labels.iloc[10]
#     # Initialize a set for labels
#     combined_labels = set()
#
#     # Function to check if the label ends with a digit and it is not 0
#     def ends_with_digit(label):
#         return label[-1].isdigit() and label[-1] != '0'
#
#     # Function to extract prefix from label
#     def extract_prefix(label):
#         return ''.join(filter(lambda x: not x.isdigit(), label.split('_')[0]))
#
#     # Handle 'matched_labels' column
#     if isinstance(row['matched_labels'], str):
#         # Convert string representation of list to actual list
#         try:
#             matched_labels = ast.literal_eval(row['matched_labels'])
#             if isinstance(matched_labels, list):
#                 # Filter out labels ending with digits
#                 combined_labels.update(label for label in matched_labels if ends_with_digit(label))
#         except ValueError:
#             # Handle the case where the string is not a list
#             if not ends_with_digit(row['matched_labels']):
#                 combined_labels.add(row['matched_labels'])
#
#     # Columns that contain consolidated labels
#     consolidated_cols = ['L1', 'L2', 'L3', 'L4']
#
#     # Iterate through each consolidated label column
#     for col in consolidated_cols:
#         # Check if the column value is a string and not empty
#         if isinstance(row[col], str) and row[col]:
#
#             # Remove unwanted characters (brackets and quotes)
#             cleaned_labels = row[col].translate(str.maketrans('', '', '[]\''))
#
#             # Split by comma or dot
#             labels = re.split('[,.]', cleaned_labels)
#
#             # Filter out labels ending with digits
#             combined_labels.update(label.strip() for label in labels if label.strip() and ends_with_digit(label.strip()))
#
#
#     # Check if any label's prefix is not in the acceptable list
#     for label in combined_labels:
#         if extract_prefix(label) not in label_prefix_list:
#             # if the label is not include "Q'   print the row
#             if extract_prefix(label) != 'Q':
#                 print(row)
#
#
#
#
#     # Return NaN if the set is empty, else return the set
#     return combined_labels if combined_labels else pd.NA

def compare_label_sets(row, label_prefix_list, consolidated_cols=['L1', 'L2', 'L3', 'L4']):
    """
    Extracts and processes labels from a DataFrame row.

    Parameters:
    row (pd.Series): A row from a DataFrame.
    label_prefix_list (list): List of acceptable label prefixes.
    consolidated_cols (list): List of column names with consolidated labels.

    Returns:
    set or pd.NA: A set of processed labels or pd.NA if the set is empty.
    """

    def ends_with_digit(label):
        """Check if the label ends with a digit (not 0)."""
        return label[-1].isdigit() and label[-1] != '0'

    def extract_prefix(label):
        """Extract prefix from label."""
        return ''.join(filter(lambda x: not x.isdigit(), label.split('_')[0]))

    combined_labels = set()

    # Process 'matched_labels' column
    if isinstance(row['matched_labels'], str):
        try:
            matched_labels = ast.literal_eval(row['matched_labels'])
            if isinstance(matched_labels, list):
                combined_labels.update(label for label in matched_labels if ends_with_digit(label))
        except (ValueError, SyntaxError):
            if ends_with_digit(row['matched_labels']):
                combined_labels.add(row['matched_labels'])

    # Process consolidated label columns
    for col in consolidated_cols:
        if isinstance(row[col], str) and row[col]:
            cleaned_labels = row[col].translate(str.maketrans('', '', '[]\''))
            labels = re.split('[,.]', cleaned_labels)
            combined_labels.update(
                label.strip() for label in labels if label.strip() and ends_with_digit(label.strip()))

    # Filter out labels starting with "Q"
    combined_labels = {label for label in combined_labels if not label.startswith('Q')}

    # Check if any label's prefix is not in the acceptable list
    for label in combined_labels:
        if extract_prefix(label) not in label_prefix_list:
            print(row)

    return combined_labels if combined_labels else pd.NA


label_prefix_list = ['AG', 'DAG', 'TT', 'DTT', 'NM', 'SPY', 'CH', 'DF']
# Get the current working directory
code_dir = '/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/'

# Construct the full path to the file
game_name_file_path = os.path.join(code_dir, 'Data/game_names.csv')
Reviewed_files = os.path.join(code_dir, 'Data/Private_Data/Reviewed/')

# create a df to save the result
processed_labels = pd.DataFrame()
# initialize a counter to count the number of files that are successfully processed
countSuccess = 0

# Get all game names
game_names = []
with open(game_name_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        game_names.append(row[0])

print(game_names)


# process each game
for game_name in game_names:
    print(game_name)
    # game_name = '003NTU'
    try:
        # find the corresponding files
        file_path = glob.glob(os.path.join(Reviewed_files, f'{game_name}.xlsx'))[0]

        # read the files, using try and except to handle the error

        raw_labels = pd.read_excel(file_path)

        # Apply the compare_label_sets function to each row
        try:
            raw_labels['combined_labels'] = raw_labels.apply(compare_label_sets, axis=1,
                                                             label_prefix_list=label_prefix_list)

            # only keep the columns that are needed,first 4 columns and the combined labels
            raw_labels = raw_labels.iloc[:, :5].join(raw_labels['combined_labels'])
            raw_labels.columns = ['indx', 'round', 'speaker', 'trans', 'game', 'raw_labels']

            # add the labels to the processed_labels,included the index
            processed_labels = pd.concat([processed_labels, raw_labels], ignore_index=False)
            print(f'Finished processing {game_name}.xlsx')
            countSuccess += 1
        except:
            print(f'Error using compare_label_sets on {game_name}')
            continue
    except:
        print(f'Error processing {game_name}.xlsx')
        continue

print(f'Finished processing {countSuccess} files')

# drop the last column, keep the first columns
processed_labels = processed_labels.iloc[:, :6]
processed_labels.columns = ['indx', 'round', 'speaker', 'trans', 'game', 'raw_labels']

processed_labels['round'].unique()
# fill the na in round with same round as the previous row
processed_labels['round'] = processed_labels['round'].ffill()

# save the labels in csv file
processed_labels.to_csv(Reviewed_files + 'combined_labels_w_transcripts.csv', index=False)

# # error handling
# check = processed_labels[(processed_labels['game'] == '007NTU') & (~processed_labels['raw_labels'].isna())]
# check = processed_labels[(processed_labels['game'] == '008ISR') & (~processed_labels['raw_labels'].isna())]
check = processed_labels[(processed_labels['game'] == '007ISR') & (~processed_labels['raw_labels'].isna())]
