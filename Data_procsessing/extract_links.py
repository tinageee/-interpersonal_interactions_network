"""
Extract links from raw labels.

This script processes the raw labels from a CSV file, categorizes them based on sentiment,
and extracts 'from', 'to', and 'sentiment_score' for each link.


input:  the labels
output: the csv file that saved all the linked information

"""

import os
import csv
import glob
import pandas as pd
import ast
import re

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
            sentiment = 1 if any(label.startswith(pos_label) for pos_label in positive_labels) else -1 if any(label.startswith(neg_label) for neg_label in negative_labels) else 0

            # Create a dictionary for each link and append to the list
            link = {'from': row['speaker'], 'to': to_speaker, 'sentiment_score': sentiment, 'round': row['round'], 'game': row['game'],'indx': row['indx']}
            links =pd.concat([links, pd.DataFrame(link, index=[0])], ignore_index=True)

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



"""
extract the links from the raw labels
"""

import os
import csv
import glob
import pandas as pd
import ast
import re

Code_dir='/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/'
# read the raw labels
raw_labels = pd.read_csv(Code_dir+'Data/Private_Data/Reviewed/combined_labels_w_transcripts.csv')


# remove the rows with empty labels
raw_labels = raw_labels[raw_labels['raw_labels'].notna()]
# remove the transcripts column
raw_labels = raw_labels.drop(columns=['trans'])

# initialize a df to store the links
all_links = pd.DataFrame()
# categorize the labels
def categorize_and_update_row(row):
    """
    Update each row with categorized labels, 'from', 'to', and sentiment score.

    :param row: A pandas Series representing a row of the DataFrame.
    :return: The row with updated 'from', 'to' and sentiment score.
    """
    #initialize a df to store the links information from each row
    links = pd.DataFrame(columns=['to', 'sentiment_score'])
    row=raw_labels.iloc[5]
    positive_labels = {'AG', 'TT', 'NM', 'DF'}
    negative_labels = {'DAG', 'DTT', 'SPY', 'CH'}

    # Default sentiment and to/from values
    sentiment = 0
    from_speaker = row['speaker'] # Assuming 'speaker' is the 'from' value
    to_speaker = None

    # read all the labels
    # eg. raw_labels    {'TT3', 'TT2'}
    labels = row['raw_labels'].translate(str.maketrans('', '', '[]\'{} ')).split(',')

    for label in labels:
        # Check if label has a digit suffix and assign sentiment

        # Extract digits from label
        digits = ''.join(filter(str.isdigit, label))
        if not digits: #check if the label has a digit suffix
            print(row)
        else:

            # the digit represents the object of the statement
            to_speaker = int(digits)  # Convert to integer if digit is present

            if any(label.startswith(pos_label) for pos_label in positive_labels):
                print(label)
                sentiment = 1
            elif any(label.startswith(neg_label) for neg_label in negative_labels):
                sentiment = -1
            #add a row to record the link
            links = links.append({'to': to_speaker, 'sentiment_score': sentiment}, ignore_index=True)

    #remove the duplicates
    # if the same sentiment to the same person is recorded multiple times, only keep one
    # links = links.drop_duplicates()

    # add rest of the column to links then add the links to the all_links
    links['from'] = from_speaker
    links['round'] = row['round']
    links['game'] = row['game']
    all_links = all_links.append(links, ignore_index=True)

# Apply the function to each row in the DataFrame
raw_labels.apply(categorize_and_update_row, axis=1)

# save the links to a csv file
all_links.to_csv(Code_dir+'Data/all_links.csv', index=False)