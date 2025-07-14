import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

url_mm_csv = "https://www.texaslottery.com/export/sites/lottery/Games/Mega_Millions/Winning_Numbers/megamillions.csv"
url_pb_csv = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/powerball.csv"


# def getMM():
#     mm_df = pd.read_csv(url_mm_csv, header=None,
#                         names=['Game', 'Month', 'Day', 'Year', 'Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5',
#                                'Mega Ball', 'Megaplier'])
#     return mm_df
#
# def transMM():
#     mm_df = getMM()
#     mm_df['Date'] = pd.to_datetime(mm_df[['Month', 'Day', 'Year']])
#     return mm_df
#
#
# def MM_FL(data):
#     ax = plt.axes()
#     fl_map = sb.heatmap(data[['Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5', 'Mega Ball']], annot=True,
#                         cmap='coolwarm', vmin=-1, vmax=1)
#     return fl_map


def getPB():
    pb_df = pd.read_csv(url_pb_csv, header=None,
                        names=['Game', 'Month', 'Day', 'Year', 'Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5',
                               'Mega Ball', 'Megaplier'])
    return pb_df


def transPB():
    pb_df = getPB()
    pb_df['Date'] = pd.to_datetime(pb_df[['Month', 'Day', 'Year']])
    return pb_df


# get data and print head
if __name__ == "__main__":

    # # run for mega millions
    # mm_df = transMM()
    # print(mm_df.head())
    #
    # # get heatmap
    # ax = plt.axes()
    # chart = MM_FL(mm_df)
    # chart.title('Mega Millions FL Heatmap')
    # chart.show()

    # run for powerball

    pb_df = transPB()
    pd.set_option('display.max_columns', None)
    print(pb_df.head())

    print(pb_df[['Date','Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5', 'Mega Ball']].head())
    pb_df = pb_df[['Date','Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5', 'Mega Ball']]


    # # get heatmap
    # pb_chart = sns.heatmap(pb_df[['Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5', 'Mega Ball']], annot=True, fmt='.2f')
    # pb_chart.set_title('Powerball FL Heatmap')
    # plt.show()

    # Regular Balls heatmap
    regular_balls = pb_df.iloc[:, 1:6].values.flatten()
    freq_regular = pd.Series(regular_balls).value_counts().sort_index()

    # Adjusting for 1-based indexing with even tens
    heatmap_data_regular = np.zeros((10, 7))
    for idx, value in freq_regular.items():
        if 1 <= idx <= 70:  # Considering the additional spots for even tens
            i, j = divmod(idx, 7)
            # Placing even tens (10, 20, 30, etc.) at the end of each row
            j = j - 1 if j != 0 else 6
            heatmap_data_regular[i, j] = value
        else:
            print(f"Warning: Ball number {idx} is out of expected range!")

    heatmap_data_regular = heatmap_data_regular.astype(int)

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data_regular, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Frequency'},
                xticklabels=['1', '2-7', '8-14', '15-21', '22-28', '29-35', '36-70'],
                yticklabels=['1-7', '8-14', '15-21', '22-28', '29-35', '36-42', '43-49', '50-56', '57-63', '64-70'],
                fmt='d')
    plt.title('Regular Balls Frequency')
    plt.show()

    # Mega Ball heatmap with 1-based indexing
    freq_mega = pb_df['Mega Ball'].value_counts().sort_index()

    heatmap_data_mega = np.zeros(30)
    for idx, value in freq_mega.items():
        if 1 <= idx <= 30:
            heatmap_data_mega[idx] = value  # No need to subtract 1 for 1-based indexing
        else:
            print(f"Warning: Mega Ball number {idx} is out of expected range!")

    heatmap_data_mega = heatmap_data_mega.astype(int)  # Cast to integer

    heatmap_data_mega = heatmap_data_mega.reshape(6, 5)
    plt.figure(figsize=(10, 7))
    sns.heatmap(heatmap_data_mega, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Frequency'})
    plt.title('Mega Ball Frequency')
    plt.show()