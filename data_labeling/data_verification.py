"""
This script is designed for the cross-validation and coding of conversational data between two coders. It includes the following processes:

1. Identifying and loading game files from specified directories.
2. Comparing the coding done by two different coders and consolidating their tags.
3. Converting and normalizing labels to a predefined set of acceptable tags.
4. Identifying rows with labels that require review based on certain conditions.
5. Generating a report of these identified rows and saving them for further analysis.

The script is structured to process Excel files containing conversational data, which have been coded by two separate coders. Each coder's tags are compared, discrepancies are identified, and rows that need further review are marked. The output is an Excel file that contains these marked rows, ready for review.

Instructions:
- The script uses global variables to define directories and file lists. These should be set to the appropriate paths before running the script.
- Ensure that pandas and other required libraries are installed in the Python environment where the script is run.
- The script contains several commented-out sections for optional functionality, such as renaming files or extracting game names. Uncomment these as needed.

input: two folders containing the files from two coders
output: a folder containing the files that need to be reviewd

created by: Saiying Ge on Jan 2023

"""


import os
import csv
import glob
import pandas as pd
import re

# def extra_game_names(directory):
#     # List all files in the directory
#     file_names = os.listdir(directory)
#
#     # Assuming game names are the full filenames without extensions
#     game_names = [os.path.splitext(file)[0] for file in file_names]
#
#     return game_names


# Directory containing the game files
directory_Coder1 = '/Users/saiyingge/**Research Projects/Nomination-network/data/Tags_two_coders/XC_tags'
directory_Coder2 = '/Users/saiyingge/**Research Projects/Nomination-network/data/Tags_two_coders/KG_tags'

# save directory
directory_save = '/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/Data/Private_Data'

# the file need to restructure
file_restructure = ['003USP', '003ZAM', '007ISR', '010HK','003ISR', '008USP', '013HK']
# coder1 columns
coder1_columns = ['L1_1', 'L2_1', 'L3_1', 'L4_1', 'L5_1']
# coder2 columns
coder2_columns = ['L1_2', 'L2_2', 'L3_2', 'L4_2', 'L5_2']
# List of label columns
label_columns = coder1_columns + coder2_columns

# Define the list of acceptable labels
acceptable_label_prefixes = ['AG', 'DAG', 'TT', 'DTT', 'SPY', 'NM', 'CH', 'DF', 'Q', 'POS', 'NEG', 'RC', 'SG', 'EXP',
                             'CF'
    , 'POS', 'SPECIAL', 'PRETEND', 'WEAK', 'CONFIDENCE', 'STRATEGY','GR_','TEAM','POINT','SWEAR','PROMISE',"VL",'SELF','SHOULD',
                             'UC',"DWT",'VILL','DNW','DNW',"EXO",'NC']

# Define the list of interested labels
interested_label_prefixes = ['AG', 'DAG', 'TT', 'DTT', 'NM', 'SPY', 'CH', 'DF', 'Q']

# Initialize an empty DataFrame to store rows with unacceptable labels
rows_with_unacceptable_prefixes = pd.DataFrame()


# # Extract game names
# game_names = extra_game_names(directory_Coder1)
# # List all files in the directory
# print(game_names)
# # save the game names to a txt file
# csv.writer(open(f'{directory_save}/game_names.csv', 'w', newline='', encoding='utf-8')).writerows([[name] for name in game_names])

# # reformat the file names in coder1 KG directory
# for file_path in glob.glob(os.path.join(directory_Coder2, '*-T.xlsx')):
#     # Construct new file name
#     new_file_path = file_path.replace('-T.xlsx', '.xlsx')
#
#     # Rename the file
#     os.rename(file_path, new_file_path)

# # reformat the file type in coder2 XC directory
# # Iterate over all CSV files in the directory
# for csv_file in glob.glob(os.path.join(directory_Coder1, '*.csv')):
#     # Read the CSV file
#     df = pd.read_csv(csv_file, encoding='ISO-8859-1')
#
#     # Create the XLSX file name
#     xlsx_file = os.path.splitext(csv_file)[0] + '.xlsx'
#
#     # Write to an XLSX file
#     df.to_excel(xlsx_file, index=False)

# check if the files can be read and if the first four columns are same
# file_notmatch = []
# error_file = []
# # for all the games check if first four columns are same
# for game_name in game_names:
#     # find the corresponding files
#     game_name=game_names[0]
#     file_path_Coder1 = glob.glob(os.path.join(directory_Coder1, f'{game_name}.xlsx'))[0]
#     file_path_Coder2 = glob.glob(os.path.join(directory_Coder2, f'{game_name}.xlsx'))[0]
#
#     # read the files
#     # use try except to avoid the error of empty file
#     try:
#         df_Coder1 = pd.read_excel(file_path_Coder1)
#         df_Coder2 = pd.read_excel(file_path_Coder2)
#         # if not same, save the game name to a list
#         if not df_Coder1.iloc[:, 0:5].equals(df_Coder2.iloc[:, 0:5]):
#             print(game_name)
#             file_notmatch.append(game_name)
#     except:
#         error_file.append(game_name)
#
# for game_name in file_notmatch:
#     file_path_Coder1 = glob.glob(os.path.join(directory_Coder1, f'{game_name}.xlsx'))[0]
#     file_path_Coder2 = glob.glob(os.path.join(directory_Coder2, f'{game_name}.xlsx'))[0]
#
#     df_Coder1 = pd.read_excel(file_path_Coder1)
#     df_Coder2 = pd.read_excel(file_path_Coder2)
#
#     print(df_Coder1.iloc[:, 0:3].equals(df_Coder2.iloc[:, 0:3]))
#     # these file missing the first column(index), add the first column
#     df_Coder1.insert(0, 'indx', range(1, 1 + len(df_Coder1)))

# convert the label in L1_1 to L5_1 to same name write in function, using disctionary
def convert_label(label):
    if not isinstance(label, str):
        return label  # Return the label as is if it's not a string
    #Trim whitespace
    label = label.strip()

    label_dict = {
        'AGREE': 'AG', 'DISAGREE': 'DAG',
    'TRUST': 'TT', 'DISTRUST': 'DTT',
    'SPY': 'SPY', 'SP': 'SPY', 'SP?': 'SPY?',
    'ZHIYI': 'CH', 'BIANJIE': 'DF',
    'QUESTION': 'Q', 'QUES': 'Q',
    'POSITIVE': 'POS', 'NEGATIVE': 'NEG', 'SUGGESTION': 'SG',  'SUGG': 'SG',
    'RECALL': 'RC',
    'REASONING': 'EXP', 'CONFUSE': 'CF', 'C': 'CONFIDENCE',
    'DDT7': 'DTT7', 'BIANJIEW': 'DF', 'ZX': 'CH',
                      #error fix
                      'DDT7':'DTT7', 'BIANJIEW':'DF','BAINJIE':'DF','ZX':'CH', 'STT3':'TT3', 'DISTRU2':'DTT2',
    'TRUSTY': 'TT','NSPY':'DF','DISTURST':'DTT','TRUSTP8':'TT8',"M3":'NM3','SP8':'SPY8'}
    # First, try to find a direct match in the dictionary
    if label in label_dict:
        return label_dict[label]

    # If no direct match, apply pattern matching for TRUST, DISTRUST, and SP followed by a digit
    pattern = re.compile(r'^(TRUST|DISTRUST|SP)(\d+)$')
    match = pattern.match(label)
    if match:
        prefix, digit = match.groups()
        if prefix == 'TRUST':
            return f'TT{digit}'
        elif prefix == 'DISTRUST':
            return f'DTT{digit}'
        elif prefix == 'SP':
            return f'SPY{digit}'

    return label


# Function to check if a label does not start with any of the acceptable prefixes
def does_not_start_with_acceptable_prefix(label):
    if pd.isna(label):
        return False  # Exclude NaN values
    return not any(label.startswith(prefix) for prefix in acceptable_label_prefixes)


# Function to check if a label does not start with any of the interested prefixes
def does_not_start_with_interested_prefix(label):
    if pd.isna(label) or not isinstance(label, str):
        return False  # Exclude NaN and non-string values
    return any(label.startswith(prefix) for prefix in interested_label_prefixes)


# Function to check if a label starts with any of the interested prefixes
def change_label_if_not_interested(label):
    if pd.isna(label) or any(label.startswith(prefix) for prefix in interested_label_prefixes):
        return label  # Return the label as is if it's NaN or starts with an interested prefix
    return pd.NA  # Change label to NaN if it doesn't start with an interested prefix


# Function to compare labels as sets with additional condition
def compare_label_sets(row, coder1_cols, coder2_cols):
    coder1_labels = set(row[coder1_cols].dropna())
    coder2_labels = set(row[coder2_cols].dropna())

    # 1) Find exact matches first
    exact_matches = coder1_labels.intersection(coder2_labels)

    # Remove exact matches from consideration for the next step
    coder1_labels -= exact_matches
    coder2_labels -= exact_matches

    # 2)
    # Special case: NM0 in coder 1 and NM in coder 2
    if 'NM0' in coder1_labels and 'NM' in coder2_labels:
        exact_matches.add('NM')
        coder1_labels.discard('NM0')
        coder2_labels.discard('NM')

    # 3)
    # Find matches where coder 1's label corresponds to coder 2's label with a number
    pattern_matches = set()
    labels_to_remove_from_coder1 = set()

    for label in coder1_labels:
        pattern = re.compile(rf'^{re.escape(label)}\d+$')
        matching_labels = {l for l in coder2_labels if pattern.match(l)}
        if matching_labels:
            pattern_matches.update(matching_labels)
            labels_to_remove_from_coder1.add(label)

    # Remove identified matches from coder 1's labels
    coder1_labels -= labels_to_remove_from_coder1
    # Combine exact matches and pattern matches
    all_matches = exact_matches.union(pattern_matches)

    # Remove identified matches from unique label sets
    coder1_unique = coder1_labels - pattern_matches

    coder2_unique = coder2_labels - pattern_matches

    return list(all_matches), list(coder1_unique), list(coder2_unique)

# Function to determine if a row needs review
def needs_review(row):
    # Check if either of unique labels for coders are non-empty
    # Check if either of unique labels for coders are non-empty
    if row['coder1_unique_labels'] or row['coder2_unique_labels']:
        return 'Review'

    # Pattern to check for AG, DAG, Q anywhere in the string
    pattern_contain = re.compile(r'(AG|DAG|Q)')
    if any(pattern_contain.search(label) for label in row['matched_labels']):
        return 'Review'

    # Pattern to check for DF, TT, DTT at the beginning of the string and not followed by a number
    pattern_start = re.compile(r'^(DF|TT|DTT)\b')
    if any(pattern_start.match(label) for label in row['matched_labels']):
        return 'Review'

    return ''


# read the game names from the txt file
game_names = []
with open('/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/Data/game_names.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        game_names.append(row[0])

print(game_names)

# for each game
for game_name in game_names:

    # find the corresponding files
    file_path_Coder1 = glob.glob(os.path.join(directory_Coder1, f'{game_name}.xlsx'))[0]
    file_path_Coder2 = glob.glob(os.path.join(directory_Coder2, f'{game_name}.xlsx'))[0]

    # read the files
    df_Coder1 = pd.read_excel(file_path_Coder1)
    df_Coder2 = pd.read_excel(file_path_Coder2)

    # check if the first four columns in two files are same
    if df_Coder1.iloc[:, 0:5].equals(df_Coder2.iloc[:, 0:5]):

        # keep the first four columns, combine two coders' tags
        # add tags columns from coder2 to coder1
        combined = pd.concat([df_Coder1, df_Coder2.iloc[:, 5:]], axis=1)
    elif df_Coder1.iloc[:, 0:3].equals(df_Coder2.iloc[:, 0:3]):
        # There are 7 files that need to be restructured
        # print(game_name)
        # these file missing the first column(index), add the first column
        df_Coder1.insert(0, 'indx', range(1, 1 + len(df_Coder1)))
        # keep the first four columns, combine two coders' tags
        # change the column order to ['indx', 'round', 'speaker', 'trans', 'game']
        # Assuming 'df' is your original DataFrame
        df_Coder1 = df_Coder1[['indx', 'round', 'speaker', 'trans', 'game','L1', 'L2', 'L3', 'L4', 'L5']]
        # add tags columns from coder2 to coder1
        combined = pd.concat([df_Coder1, df_Coder2.iloc[:, 4:]], axis=1)
    # change the column names
    combined.columns = ['indx', 'round', 'speaker', 'trans', 'game'] + label_columns

    # convert the label to same name

    # Apply the label conversion to each column
    for col in label_columns:
        combined[col] = combined[col].apply(convert_label)

    ###  check abnormal labels

    # Apply the check across the DataFrame
    for col in label_columns:
        if col in combined.columns:  # Check if the column exists
            # Find rows where the label does not start with any of the acceptable prefixes
            unacceptable_rows = combined[combined[col].apply(does_not_start_with_acceptable_prefix)]
            # Concatenate these rows to the DataFrame
            rows_with_unacceptable_prefixes = pd.concat([rows_with_unacceptable_prefixes, unacceptable_rows])

    ### cross check and save the data
    # we only interested in the tags that has object and include relational message
    # positive: AG,TT,NM,DF
    # negative: DAG,DTT,SPY,CH#
    # unsure label: Q(question)

    # Apply the change_label_if_not_interested function to each label column
    for col in label_columns:
        if col in combined.columns:  # Check if the column exists
            combined[col] = combined[col].apply(change_label_if_not_interested)

    ### cross check the label from two coders

    # Initialize a new DataFrame to store the results
    cross_checked = pd.DataFrame()

    # Apply the compare_label_sets function across the DataFrame
    cross_checked['matched_labels'], cross_checked['coder1_unique_labels'], cross_checked['coder2_unique_labels'] = zip(
        *combined.apply(compare_label_sets, args=(coder1_columns, coder2_columns), axis=1))


    # add a column to mark raws that need to be checked
    # Apply the needs_review function to each row to create a new column
    cross_checked['Review'] = cross_checked.apply(needs_review, axis=1)

    # include the original transcript
    cross_checked = pd.concat([df_Coder1.iloc[:, 0:5], cross_checked], axis=1)

    # add 3 empty columns to the df and save the data
    cross_checked['L1'] = ''
    cross_checked['L2'] = ''
    cross_checked['L3'] = ''
    cross_checked['L4'] = ''

    # save the data
    cross_checked.to_excel(f'{directory_save}/NeedReview/{game_name}.xlsx', index=False)


# show the labels of ag in the rows that need to be checked

# Filter rows where 'matched_labels' or 'coder1_unique_labels' contain 'AG'
# filtered_rows = cross_checked[
#     cross_checked['matched_labels'].apply(lambda labels: any('CH' in label for label in labels if isinstance(labels, list))) |
#     cross_checked['coder2_unique_labels'].astype(str).str.contains('CH')
# ]
