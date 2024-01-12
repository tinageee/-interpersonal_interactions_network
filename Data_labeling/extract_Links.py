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

label_prefix_list= ['AG', 'DAG', 'TT', 'DTT', 'NM', 'SPY', 'CH', 'DF']
# Get the current working directory
parent_dir = os.path.dirname(os.getcwd())

# Construct the full path to the file
game_name_file_path = os.path.join(parent_dir, 'Data/game_names.csv')
Reviewed_files = os.path.join(parent_dir, 'Data/Private_Data/Reviewed/')

# Get all game names
game_names = []
with open(game_name_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        game_names.append(row[0])

print(game_names)

game_name= game_names[0]
# find the corresponding files
file_path = glob.glob(os.path.join(Reviewed_files, f'{game_name}.xlsx'))[0]

# read the files
raw_labels = pd.read_excel(file_path)

def compare_label_sets(row):
    # Extract labels from the matched_labels and consolidated labels, put them into a set for each raw
    # checked if the label ends with a digit, if so, add it to the set
    # if the label is not in the acceptable list, print the row

    # Initialize a set for labels
    combined_labels = set()

    # Function to check if the label ends with a digit
    def ends_with_digit(label):
        return re.search(r'\d+$', label) is not None

    # Function to extract prefix from label
    def extract_prefix(label):
        return ''.join(filter(lambda x: not x.isdigit(), label.split('_')[0]))

    # Handle 'matched_labels' column
    if isinstance(row['matched_labels'], str):
        # Convert string representation of list to actual list
        try:
            matched_labels = ast.literal_eval(row['matched_labels'])
            if isinstance(matched_labels, list):
                # Filter out labels ending with digits
                combined_labels.update(label for label in matched_labels if ends_with_digit(label))
        except ValueError:
            # Handle the case where the string is not a list
            if not ends_with_digit(row['matched_labels']):
                combined_labels.add(row['matched_labels'])

    # Columns that contain consolidated labels
    consolidated_cols = ['L1', 'L2', 'L3', 'L4']

    # Iterate through each consolidated label column
    for col in consolidated_cols:
        # Check if the column value is a string and not empty
        if isinstance(row[col], str) and row[col]:
            # Remove unwanted characters, split by comma, and add to the set
            labels = row[col].translate(str.maketrans('', '', '[]\'')).split(',')
            # Filter out labels ending with digits
            combined_labels.update(label.strip() for label in labels if label.strip() and ends_with_digit(label.strip()))


    # Check if any label's prefix is not in the acceptable list
    for label in combined_labels:
        if extract_prefix(label) not in label_prefix_list:
            print(row)


    # Return NaN if the set is empty, else return the set
    return combined_labels if combined_labels else pd.NA

# Apply the compare_label_sets function to each row
raw_labels['combined_labels'] = raw_labels.apply(compare_label_sets, axis=1)

# only keep the columns that are needed,first 4 columns and the combined labels
raw_labels = raw_labels.iloc[:, :5].join(raw_labels['combined_labels'])
#change the column name
raw_labels.columns = ['indx', 'round', 'speaker', 'trans', 'game', 'raw_labels']

#save the labels in csv file
raw_labels.to_csv(Reviewed_files+'combined_labels_w_transcripts.csv', index=False)
