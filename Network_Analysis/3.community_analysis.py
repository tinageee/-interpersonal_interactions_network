'''
calculte the connection score for each game/network
adjust the connection score by the number of players
pair-wise comparison
visualize the connection scores
'''

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel


def calculate_connection(network):
    # Initialize counts for different types of connections within and between communities
    TT_Trust = 0  # Trust between truth-tellers
    DD_Trust = 0  # Trust between deceivers
    DT_Trust = 0  # Trust between truth-tellers and deceivers
    TD_Trust = 0  # Trust between deceivers and truth-tellers
    TT_Distrust = 0  # Distrust between truth-tellers
    DD_Distrust = 0  # Distrust between deceivers
    DT_Distrust = 0  # Distrust between truth-tellers and deceivers
    TD_Distrust = 0  # Distrust between deceivers and truth-tellers

    # Iterate through edges to count different types of connections
    for u, v, data in network.edges(data=True):
        role_u = network.nodes[u]['role']
        role_v = network.nodes[v]['role']

        if data['sign'] == 1:  # Trust
            if role_u == 'Villager' and role_v == 'Villager':
                TT_Trust += 1
            elif role_u == 'Spy' and role_v == 'Spy':
                DD_Trust += 1
            elif role_u == 'Spy' and role_v == 'Villager':
                DT_Trust += 1
            elif role_u == 'Villager' and role_v == 'Spy':
                TD_Trust += 1
        elif data['sign'] == -1:  # Distrust
            if role_u == 'Villager' and role_v == 'Villager':
                TT_Distrust += 1
            elif role_u == 'Spy' and role_v == 'Spy':
                DD_Distrust += 1
            elif role_u == 'Spy' and role_v == 'Villager':
                DT_Distrust += 1
            elif role_u == 'Villager' and role_v == 'Spy':
                TD_Distrust += 1
    return TT_Trust, DD_Trust, DT_Trust, TD_Trust, TT_Distrust, DD_Distrust, DT_Distrust, TD_Distrust


# read the networks in all the folders
Code_dir = '/Users/saiyingge/Coding Projects/PyCharmProjects/NetworkProject/'

# get the player(nodes) information
game_nodes = pd.read_csv(Code_dir + 'Data/all_nodes.csv')
game_info = pd.read_csv(Code_dir + 'Data/' + 'game_info.csv')

# get game names
games = game_info['game_name'].unique()

count = 0
# read the networks by game and calculate the connection score
for game in games:
    # game='018AZ'
    try:
        network = nx.read_graphml(Code_dir + 'Data/Networks/' + game + '.graphml')

        TT_Trust, DD_Trust, DT_Trust, TD_Trust, TT_Distrust, DD_Distrust, DT_Distrust, TD_Distrust = calculate_connection(
            network)
        # add the scores to the game_nodes divide by the number of spy or village

        game_info.loc[game_info['game_name'] == game, 'TT_Trust'] = TT_Trust
        game_info.loc[game_info['game_name'] == game, 'DD_Trust'] = DD_Trust
        game_info.loc[game_info['game_name'] == game, 'DT_Trust'] = DT_Trust
        game_info.loc[game_info['game_name'] == game, 'TD_Trust'] = TD_Trust

        game_info.loc[game_info['game_name'] == game, 'TT_Distrust'] = TT_Distrust
        game_info.loc[game_info['game_name'] == game, 'DD_Distrust'] = DD_Distrust
        game_info.loc[game_info['game_name'] == game, 'DT_Distrust'] = DT_Distrust
        game_info.loc[game_info['game_name'] == game, 'TD_Distrust'] = TD_Distrust

        count += 1
    except:
        print(f'Error processing {game}.graphml')
        continue

print(f'Finished processing {count} files')

# divide the connection scores by the number of players
game_info['TT_Trust'] = game_info['TT_Trust'] / game_info['Num0fVillager']
game_info['DD_Trust'] = game_info['DD_Trust'] / game_info['Num0fSpy']
game_info['DT_Trust'] = game_info['DT_Trust'] / game_info['Num0fVillager']
game_info['TD_Trust'] = game_info['TD_Trust'] / game_info['Num0fSpy']

game_info['TT_Distrust'] = game_info['TT_Distrust'] / game_info['Num0fVillager']
game_info['DD_Distrust'] = game_info['DD_Distrust'] / game_info['Num0fSpy']
game_info['DT_Distrust'] = game_info['DT_Distrust'] / game_info['Num0fVillager']
game_info['TD_Distrust'] = game_info['TD_Distrust'] / game_info['Num0fSpy']

# adjust by the round number
game_info['TT_Trust_adj'] = game_info['TT_Trust'] / game_info['max_round']
game_info['DD_Trust_adj'] = game_info['DD_Trust'] / game_info['max_round']
game_info['DT_Trust_adj'] = game_info['DT_Trust'] / game_info['max_round']
game_info['TD_Trust_adj'] = game_info['TD_Trust'] / game_info['max_round']

game_info['TT_Distrust_adj'] = game_info['TT_Distrust'] / game_info['max_round']
game_info['DD_Distrust_adj'] = game_info['DD_Distrust'] / game_info['max_round']
game_info['DT_Distrust_adj'] = game_info['DT_Distrust'] / game_info['max_round']
game_info['TD_Distrust_adj'] = game_info['TD_Distrust'] / game_info['max_round']

# pair-wise comparison
### Trust

# deceiver trust and TT vs trust D
# Perform the paired t-test between DD_Trust and DT_Trust
stat, p_value = ttest_rel(game_info['DD_Trust'], game_info['DT_Trust'])
print(f'Statistic: {stat}, P-value: {p_value}')
print("Significant results after Bonferroni correction:", p_value < 0.05 / 4)

P_DD_Trust = "{:.2e}".format(p_value)

# in-group comparison
# Perform the paired t-test between DD_Trust and TT_Trust
stat, p_value = ttest_rel(game_info['DD_Trust'], game_info['TT_Trust'])
print(f'Statistic: {stat}, P-value: {p_value}')
print("Significant results after Bonferroni correction:", p_value < 0.05 / 4)
P_in_group_Trust = "{:.2e}".format(p_value)
# # Perform the paired t-test between DT_Trust and TD_Trust
# stat, p_value = ttest_rel(game_info['DT_Trust'], game_info['TD_Trust'])
#
# print(f'Statistic: {stat}, P-value: {p_value}')

### Ditrust
# deceiver Distrust and TT vs Distrust D
# Perform the paired t-test between DD_Trust and DT_Trust
stat, p_value = ttest_rel(game_info['DD_Distrust'], game_info['DT_Distrust'])

print(f'Statistic: {stat}, P-value: {p_value}')
print("Significant results after Bonferroni correction:", p_value < 0.05 / 4)
P_DD_Distrust = "{:.2e}".format(p_value)
# in-group comparison
# Perform the paired t-test between DD_Trust and TT_Trust
stat, p_value = ttest_rel(game_info['DD_Distrust'], game_info['TT_Distrust'])
print(f'Statistic: {stat}, P-value: {p_value}')
print("Significant results after Bonferroni correction:", p_value < 0.05 / 4)
P_in_group_Distrust = "{:.2e}".format(p_value)
# visualize the connection scores


# Prepare the DataFrame for plotting
df_melted_Trust = pd.melt(game_info, value_vars=['DD_Trust', 'DT_Trust', 'TT_Trust'], var_name='Trust Type',
                          value_name='Score')
df_melted_Distrust = pd.melt(game_info, value_vars=['DD_Distrust', 'DT_Distrust', 'TT_Distrust'],
                             var_name='Distrust Type', value_name='Score')
# Custom palette
palette1 = {'DD_Trust': 'cyan', 'DT_Trust': 'magenta', 'TT_Trust': 'yellow'}
palette2 = {'DD_Distrust': 'cyan', 'DT_Distrust': 'magenta', 'TT_Distrust': 'yellow'}

labels = ['Deceiver Intra-group', 'Deceiver Inter-group', 'TT Intra-group']  # Define your custom labels

# ## box plot
# # Setting up the matplotlib subplot environment
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
#
# # Plotting on the first subplot
# sns.boxplot(x='Trust Type', y='Score', data=df_melted_Trust, ax=axes[0], palette=palette1)
# # change the x-axis label for each group
# # Specify the tick locations and set custom labels
# ticks = axes[0].get_xticks()  # Get current tick locations
#
# axes[0].set_xticks(ticks[:len(labels)])
# axes[0].set_xticklabels(labels)
# axes[0].set_title('Trust Connection')
# axes[0].set_xlabel('')
# axes[0].set_ylabel('Connection Score')
#
# # Plotting on the second subplot
# sns.boxplot(x='Distrust Type', y='Score', data=df_melted_Distrust, ax=axes[1], palette=palette2)
#
# axes[1].set_xticks(ticks[:len(labels)])
# axes[1].set_xticklabels(labels)
# axes[1].set_title('Distrust Connection')
# axes[1].set_xlabel('')
# # Optionally, hide the y-axis label for the second plot for clarity
# axes[1].set_ylabel('')
#
# plt.tight_layout()
# plt.show()
# plt.close()
#
# # violin plot
#
# # Setting up the matplotlib subplot environment
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
#
# # Plotting on the first subplot
# sns.violinplot(x='Trust Type', y='Score', data=df_melted_Trust, ax=axes[0], palette=palette1)
# # change the x-axis label for each group
# # Specify the tick locations and set custom labels
# ticks = axes[0].get_xticks()  # Get current tick locations
# labels = ['Deceiver In-group', 'Deceiver Out-group', 'TT In-group']  # Define your custom labels
#
# axes[0].set_xticks(ticks[:len(labels)])
# axes[0].set_xticklabels(labels)
# axes[0].set_title('Trust Connection')
# axes[0].set_xlabel('')
#
# axes[0].set_ylabel('Connection Score')
#
# # Plotting on the second subplot
# sns.violinplot(x='Distrust Type', y='Score', data=df_melted_Distrust, ax=axes[1], palette=palette2)
#
# axes[1].set_xticks(ticks[:len(labels)])
# axes[1].set_xticklabels(labels)
# axes[1].set_title('Distrust Connection')
# axes[1].set_xlabel('')
# # Optionally, hide the y-axis label for the second plot for clarity
# axes[1].set_ylabel('')
#
# plt.tight_layout()
# plt.show()
# plt.close()

#  violin and box plot with jittered dots

# Create figure and axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)

# Customization properties
medianprops = dict(
    linewidth=2,
    solid_capstyle="butt"
)
boxprops = dict(
    linewidth=2,
    color='black'
)
positions = [1, 2, 3]

# Colors for jittered dots, assuming a need for distinction
colors_group = ['red', 'blue', 'green']
RED_DARK = '#d62728'


# Function to add jittered dots with specific colors
def jittered_scatter(data, ax, color, jitter_strength=0.05, **scatter_kwargs):
    jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(data))
    ax.scatter(jitter + scatter_kwargs.pop('x_pos', 1), data, color=color, **scatter_kwargs)


# Plotting for Trust
trust_types = df_melted_Trust['Trust Type'].unique()
for i, trust_type in enumerate(trust_types, start=1):
    data = df_melted_Trust[df_melted_Trust['Trust Type'] == trust_type]['Score']
    color = colors_group[i - 1]
    # Violin plot
    parts = axes[0].violinplot(data, positions=[i], widths=0.5, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('none')
        pc.set_edgecolor('grey')
        pc.set_alpha(1)
    # Box plot
    axes[0].boxplot(data, positions=[i], widths=0.2, medianprops=medianprops, boxprops=boxprops, whiskerprops=boxprops,
                    showcaps=False, showfliers=False)
    # Jittered dots
    jittered_scatter(data, axes[0], color=color, x_pos=i, alpha=0.4, edgecolor='grey', linewidth=0.5, s=20)
    # Extracting y_data for 'DD_Trust', 'DT_Trust', 'TT_Trust'
    y_data = [game_info[column] for column in ['DD_Trust', 'DT_Trust', 'TT_Trust']]

# Add mean value labels
means = [np.mean(y) for y in y_data]
for i, mean in enumerate(means):
    axes[0].scatter(positions[i], mean, s=100, color=RED_DARK, zorder=3)
    axes[0].plot([positions[i], positions[i] + 0.25], [mean, mean], ls="dashdot", color="black", zorder=3)
    axes[0].text(positions[i] + 0.25, mean, f"$\hat{{\mu}}_{{mean}} = {round(mean, 2)}$", fontsize=13, va="center",
                 bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.15), zorder=10)
# multiple comparisons p-value annotations
# Draw a line and add the text
tick_len = 0.25
axes[0].plot([1, 1, 2, 2], [6.5 - tick_len, 6.5, 6.5, 6.5 - tick_len], c="black")
axes[0].plot([1, 1, 3, 3], [8 - tick_len, 8, 8, 8 - tick_len], c="black")
axes[0].text(1.5, 6.5 + 0.02, f"$p_{{diff}}$ = {P_DD_Trust}", fontsize=11, va="bottom", ha="center")
axes[0].text(2, 8 + 0.02, f"$p_{{diff}}$ = {P_in_group_Trust}", fontsize=11, va="bottom",
             ha="center")

axes[0].set_xticks(positions[:len(labels)])
axes[0].set_xticklabels(labels)
axes[0].set_title('Trust Connections')
axes[0].set_xlabel('')
axes[0].set_ylabel('Connection Score (adjusted by # of objects)')

# Plotting for Distrust
distrust_types = df_melted_Distrust['Distrust Type'].unique()
for i, distrust_type in enumerate(distrust_types, start=1):
    data = df_melted_Distrust[df_melted_Distrust['Distrust Type'] == distrust_type]['Score']
    color = colors_group[i - 1]
    # Violin plot
    parts = axes[1].violinplot(data, positions=[i], widths=0.5, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('none')
        pc.set_edgecolor('grey')
        pc.set_alpha(1)
    # Box plot
    axes[1].boxplot(data, positions=[i], widths=0.2, medianprops=medianprops, boxprops=boxprops, whiskerprops=boxprops,
                    showcaps=False, showfliers=False)
    # Jittered dots
    jittered_scatter(data, axes[1], color=color, x_pos=i, alpha=0.4, edgecolor='grey', linewidth=0.5, s=20)

# Extracting y_data for 'DD_Trust', 'DT_Trust', 'TT_Trust'
y_data = [game_info[column] for column in ['DD_Distrust', 'DT_Distrust', 'TT_Distrust']]

# Add mean value labels
means = [np.mean(y) for y in y_data]
for i, mean in enumerate(means):
    axes[1].scatter(positions[i], mean, s=100, color=RED_DARK, zorder=3)
    axes[1].plot([positions[i], positions[i] + 0.25], [mean, mean], ls="dashdot", color="black", zorder=3)
    axes[1].text(positions[i] + 0.25, mean, f"$\hat{{\mu}}_{{mean}} = {round(mean, 2)}$", fontsize=13, va="center",
                 bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.15), zorder=10)
# Draw a line and add the text
tick_len = 0.25
axes[1].plot([1, 1, 2, 2], [6.5 - tick_len, 6.5, 6.5, 6.5 - tick_len], c="black")
axes[1].plot([1, 1, 3, 3], [8 - tick_len, 8, 8, 8 - tick_len], c="black")
axes[1].text(1.5, 6.5 + 0.02, f"$p_{{diff}}$ = {P_DD_Distrust}", fontsize=11, va="bottom",
             ha="center")
axes[1].text(2, 8 + 0.02, f"$p_{{diff}}$ = {P_in_group_Distrust}", fontsize=11, va="bottom",
             ha="center")

axes[1].set_xticks(positions[:len(labels)])
axes[1].set_xticklabels(labels)
axes[1].set_title('Distrust Connections')
axes[1].set_xlabel('')
# Optionally, hide the y-axis label for the second plot for clarity
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

#save plot
plt.savefig(Code_dir + 'Data/Analysis_Results/' + 'Connection_Score.png')

#save the game_info to a csv file
game_info.to_csv(Code_dir + 'Data/Analysis_Results/' + 'Connection_Score.csv', index=False)


# game data statistics
# total number of games
print("total number of games:", len(game_info['game_name'].unique()))
# max round mean, min and max, std
print("Mean max round:", game_info['max_round'].mean())
print("Min max round:", game_info['max_round'].min())
print("Max max round:", game_info['max_round'].max())
print("Std max round:", game_info['max_round'].std())

# game result counts
game_info['game_result'].value_counts()

game_info['player_number'] = game_info['Num0fVillager'] + game_info['Num0fSpy']

# players
print("Mean number of players in each game:", game_info['player_number'].mean())
print("Min number of players in each game:", game_info['player_number'].min())
print("Max number of players in each game:", game_info['player_number'].max())
print("Std number of players in each game:", game_info['player_number'].std())

# plot seperately

import numpy as np
import matplotlib.pyplot as plt

# Assuming 'df_melted_Trust', 'df_melted_Distrust', and 'game_info' are defined
# Customization properties
medianprops = dict(linewidth=2, solid_capstyle="butt")
boxprops = dict(linewidth=2, color='black')
positions = [1, 2, 3]
colors_group = ['red', 'blue', 'green']  # Colors for jittered dots
RED_DARK = '#d62728'

def jittered_scatter(data, ax, color, jitter_strength=0.05, **scatter_kwargs):
    jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(data))
    ax.scatter(jitter + scatter_kwargs.pop('x_pos', 1), data, color=color, **scatter_kwargs)

# Trust Connections Plot
fig, ax = plt.subplots(figsize=(7, 6))
trust_types = df_melted_Trust['Trust Type'].unique()
for i, trust_type in enumerate(trust_types, start=1):
    data = df_melted_Trust[df_melted_Trust['Trust Type'] == trust_type]['Score']
    color = colors_group[i - 1]
    parts = ax.violinplot(data, positions=[i], widths=0.5, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('none')
        pc.set_edgecolor('grey')
        pc.set_alpha(1)
    ax.boxplot(data, positions=[i], widths=0.2, medianprops=medianprops, boxprops=boxprops, whiskerprops=boxprops,
               showcaps=False, showfliers=False)
    jittered_scatter(data, ax, color=color, x_pos=i, alpha=0.4, edgecolor='grey', linewidth=0.5, s=20)

# Add mean value labels for Trust
y_data = [game_info[column] for column in ['DD_Trust', 'DT_Trust', 'TT_Trust']]
means = [np.mean(y) for y in y_data]
for i, mean in enumerate(means):
    ax.scatter(positions[i], mean, s=100, color=RED_DARK, zorder=3)
    ax.plot([positions[i], positions[i] + 0.25], [mean, mean], ls="dashdot", color="black", zorder=3)
    ax.text(positions[i] + 0.25, mean, f"$\hat{{\mu}}_{{mean}} = {round(mean, 2)}$", fontsize=13, va="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.15), zorder=10)

# Customize Trust plot
ax.set_xticks(positions[:len(trust_types)])
ax.set_xticklabels(trust_types)
ax.set_title('Trust Connections')
ax.set_ylabel('Connection Score (adjusted by # of objects)')
plt.tight_layout()
plt.show()



# Distrust Connections Plot
fig, ax = plt.subplots(figsize=(7, 6))
distrust_types = df_melted_Distrust['Distrust Type'].unique()
for i, distrust_type in enumerate(distrust_types, start=1):
    data = df_melted_Distrust[df_melted_Distrust['Distrust Type'] == distrust_type]['Score']
    color = colors_group[i - 1]
    parts = ax.violinplot(data, positions=[i], widths=0.5, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('none')
        pc.set_edgecolor('grey')
        pc.set_alpha(1)
    ax.boxplot(data, positions=[i], widths=0.2, medianprops=medianprops, boxprops=boxprops, whiskerprops=boxprops,
               showcaps=False, showfliers=False)
    jittered_scatter(data, ax, color=color, x_pos=i, alpha=0.4, edgecolor='grey', linewidth=0.5, s=20)

# Add mean value labels for Distrust
y_data = [game_info[column] for column in ['DD_Distrust', 'DT_Distrust', 'TT_Distrust']]
means = [np.mean(y) for y in y_data]
for i, mean in enumerate(means):
    ax.scatter(positions[i], mean, s=100, color=RED_DARK, zorder=3)
    ax.plot([positions[i], positions[i] + 0.25], [mean, mean], ls="dashdot", color="black", zorder=3)
    ax.text(positions[i] + 0.25, mean, f"$\hat{{\mu}}_{{mean}} = {round(mean, 2)}$", fontsize=13, va="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.15), zorder=10)

# Customize Distrust plot
ax.set_xticks(positions[:len(distrust_types)])
ax.set_xticklabels(distrust_types)
ax.set_title('Distrust Connections')
ax.set_ylabel('Connection Score (adjusted by # of objects)')
plt.tight_layout()
plt.show()