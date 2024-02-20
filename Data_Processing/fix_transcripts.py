'''
This Python script processes a transcript DataFrame to identify and separate speaker segments using regular expressions,
handling variations in speaker notation and translating spelled-out numbers to digits.

 It then restructures the data into a new DataFrame where each row corresponds to a specific speaker's contribution,
  preserving the context of each dialogue.
Finally, the script saves the restructured transcript to an Excel file, providing a well-organized and accessible format for further analysis or review.

file fixed: 013HK,008USP,010HK(reviewed)
'''

import pandas as pd
import re

# Directory path



def process_transcript_row(row):
    '''
        process a row from a transcript, splitting it into multiple rows based on speaker changes.
    :param row: row of original transcript
    :return:   fixed rows
    '''
    # row=transcript.loc[141]
    # print(row['trans'])
    # Dictionary to translate spelled-out numbers to digits
    number_translation = {
        'One': '1', 'Two': '2', 'Three': '3', 'Four': '4', 'Five': '5',
        'Six': '6', 'Seven': '7', 'Eight': '8', 'Nine': '9', 'Ten': '10'
        # Add more if needed
    }
    # remove the odd marks

    # Regular expression pattern to identify speakers
    # pattern = r'\n\s*(Player [A-Za-z]+|Second Moderator|Moderator|Multiple Participants):\s*\t?'

    pattern = r'\s*(Player [A-Za-z]+|Second Moderator|Moderator|Multiple Participants):\t?'

    # Ensure the transcript text is a string
    transcript_text = str(row['trans'])
    transcript_text = transcript_text.replace("_x000D_ _x000D_", " ")

    # Split the content based on the pattern
    parts = re.split(pattern, transcript_text)

    # If no speakers are detected, return the original row
    if len(parts) <= 2:
        return pd.DataFrame([row])
    # keep all the columns in the original row
    new_rows = [row.to_dict()]
    # Process each part and create new rows
    for i in range(1, len(parts), 2):
        speaker = parts[i].strip()
        content = parts[i + 1].strip()

        # Determine the speaker number or set it to '0' for 'Moderator'
        if 'Moderator' in speaker or 'Multiple' in speaker:
            speaker_number = '0'
        else:
            # Extract the last word and translate it to a digit if it's a spelled-out number
            last_word = speaker.split(' ')[-1]
            speaker_number = number_translation.get(last_word, last_word)

        new_rows.append({'trans': content, 'round': row['round'], 'game': row['game'], 'speaker': speaker_number,
                         'Review': 'Review'})

    return pd.DataFrame(new_rows)


# read the transcript
# transcript_dir read from hiding file
transcript = pd.read_excel(transcript_dir)

# Process each row and collect new rows
all_new_rows = []
for index, row in transcript.iterrows():
    rdf = process_transcript_row(row)
    all_new_rows.append(rdf)

# Concatenate all the DataFrames in the list
new_transcript = pd.concat(all_new_rows, ignore_index=True)

# change the indx number match with index
new_transcript['indx'] = new_transcript.index
# Convert the 'speaker' column to string type
new_transcript['speaker'] = new_transcript['speaker'].fillna(-1).astype(int)

# Save the DataFrame to an Excel file
new_transcript.to_excel('010HK.xlsx', index=False)

print(f"Transcript saved")

# find duplicates,ignore the first column
# Ignore the first column for finding duplicates for combination of round and transcript
# Identify duplicates based on 'round', 'speaker', and 'transcript'
duplicates = new_transcript.duplicated(subset=['trans'], keep=False)

# Find the duplicate rows
a = new_transcript[duplicates]
