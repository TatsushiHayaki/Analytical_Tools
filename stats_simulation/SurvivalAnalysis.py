import numpy 
import pandas as pd

def gen_dataset():
    '''うつ患者60人分のデータ'''
    columns = ['id', 'group', 'qol', 'event', 'day', 'predrug', 'duration']
    data = [
        [1, 'A', 15, 1, 50, 'NO', 1],
        [2, 'A', 13, 1, 200, 'NO', 3],
        [3, 'A', 11, 1, 250, 'NO', 2],
        [4, 'A', 11, 1, 300, 'NO', 4],
        [5, 'A', 10, 1, 350, 'NO', 2],
        [6, 'A', 9, 1, 400, 'NO', 2],
        [7, 'A', 8, 1, 450, 'NO', 4],
        [8, 'A', 8, 1, 550, 'NO', 2],
        [9, 'A', 6, 1, 600, 'NO', 5],
        [10, 'A', 6, 1, 100, 'NO', 7],
        [11, 'A', 4, 2, 250, 'NO', 4],
        [12, 'A', 3, 2, 500, 'NO', 6],
        [13, 'A', 3, 2, 750, 'NO', 3],
        [14, 'A', 3, 2, 650, 'NO', 7],
        [15, 'A', 1, 2, 1000, 'NO', 8],
        [16, 'A', 6, 1, 150, 'YES', 6],
        [17, 'A', 5, 1, 700, 'YES', 5],
        [18, 'A', 4, 2, 800, 'YES', 7],
        [19, 'A', 2, 2, 900, 'YES', 12],
        [20, 'A', 2, 2, 950, 'YES', 10],
        [21, 'B', 13, 1, 380, 'NO', 9],
        [22, 'B', 12, 1, 880, 'NO', 5],
        [23, 'B', 11, 1, 940, 'NO', 2],
        [24, 'B', 4, 2, 20, 'NO', 7],
        [25, 'B', 4, 2, 560, 'NO', 2],
        [26, 'B', 5, 1, 320, 'YES', 11],
        [27, 'B', 5, 1, 940, 'YES', 3],
        [28, 'B', 4, 2, 80, 'YES', 6],
        [29, 'B', 3, 2, 140, 'YES', 7],
        [30, 'B', 3, 2, 160, 'YES', 13],
        [31, 'B', 3, 2, 240, 'YES', 15],
        [32, 'B', 2, 2, 280, 'YES', 9],
        [33, 'B', 2, 2, 440, 'YES', 8],
        [34, 'B', 2, 2, 520, 'YES', 7],
        [35, 'B', 2, 2, 630, 'YES', 9],
        [36, 'B', 2, 2, 740, 'YES', 8],
        [37, 'B', 2, 2, 860, 'YES', 2],
        [38, 'B', 1, 2, 880, 'YES', 10],
        [39, 'B', 0, 2, 920, 'YES', 8],
        [40, 'B', 0, 2, 960, 'YES', 4],
        [41, 'C', 9, 1, 170, 'NO', 1],
        [42, 'C', 7, 1, 290, 'NO', 4],
        [43, 'C', 5, 1, 430, 'NO', 2],
        [44, 'C', 3, 2, 610, 'NO', 4],
        [45, 'C', 2, 2, 110, 'NO', 5],
        [46, 'C', 2, 2, 410, 'NO', 2],
        [47, 'C', 1, 2, 530, 'NO', 7],
        [48, 'C', 1, 2, 580, 'NO', 2],
        [49, 'C', 0, 2, 810, 'NO', 3],
        [50, 'C', 0, 2, 990, 'NO', 10],
        [51, 'C', 6, 1, 30, 'YES', 1],
        [52, 'C', 5, 1, 830, 'YES', 6],
        [53, 'C', 3, 2, 70, 'YES', 16],
        [54, 'C', 2, 2, 310, 'YES', 9],
        [55, 'C', 2, 2, 370, 'YES', 18],
        [56, 'C', 1, 2, 490, 'YES', 7],
        [57, 'C', 1, 2, 690, 'YES', 10],
        [58, 'C', 0, 2, 730, 'YES', 3],
        [59, 'C', 0, 2, 770, 'YES', 12],
        [60, 'C', 0, 2, 910, 'YES', 8],
    ]
    df = pd.DataFrame(data, columns=columns)
    return df

df = gen_dataset()
print(df.head())

df.columns
'''
'id', 
'group', 
'qol', 
'event', 改善の有無(1: 改善あり、2: 改善なし)
'day', 観察期間
'predrug', 前治療薬の有無
'duration'：罹病期間
'''

df.groupby(['group'])[['qol', 'event', 'predrug']].describe().T
df['event'] = abs(df['event']-2)

#%% カプランマイヤー曲線
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
kmf = KaplanMeierFitter()
for name, sub in df.groupby('group'):
    kmf.fit(sub['day'], sub['event'], label=f"Group {name}")
    kmf.plot_survival_function(ci_show=False)
plt.show()

