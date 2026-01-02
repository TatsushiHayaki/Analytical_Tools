""" 
Last update: Jan-2-2026
Owner: Tatsushi Hayaki
Status: Working in progress
""" 

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import os
import sys

pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)

sys_path = "/Users/tatsushihayaki/Documents/99_Work/Python/"
sys.path.insert(1, sys_path)
import sci_user_tools as my # type: ignore

def do_load(name):
    df = sns.load_dataset(name)
    df.columns = df.columns.str.lower()
    print(df.head(20))
    return df


def do_plot_median(df, byvar='country', var='life_expectancy', threshold_year=2000, reorder=True, outline='vertical'):

    def do_process(df):
        df_before = df[df['year'] < threshold_year]
        df_after = df[df['year'] >= threshold_year]

        df_before = df_before.groupby(byvar)[var].median().rename(f'before_{threshold_year}')
        df_after = df_after.groupby(byvar)[var].median().rename(f'after_{threshold_year}')
        medians = pd.concat([df_before, df_after], axis=1).reset_index()

        plot_df = medians.melt(
            id_vars=byvar,
            value_vars=[f'before_{threshold_year}', f'after_{threshold_year}'],
            var_name='year',
            value_name=f'median_{var}'
        )
        if reorder:
            plot_df = plot_df.sort_values(['year', f'median_{var}'], ascending=[False, True])
        return plot_df, medians

    plot_df, medians = do_process(df)
    is_horizontal = outline == 'horizontal'
    plt.figure(figsize=(12, 6))
    x = f'median_{var}' if is_horizontal else byvar
    y = byvar if is_horizontal else f'median_{var}'
    sns.scatterplot(
        data=plot_df,
        x=x,
        y=y,
        hue='year',
        palette={f'before_{threshold_year}': 'grey', f'after_{threshold_year}': 'blue'},
        s=70,
    )

    byvars = sorted(df[byvar].unique())
    for v in byvars:
        dots = medians.loc[medians[byvar] == v, [f'before_{threshold_year}', f'after_{threshold_year}']].values[0]
        if is_horizontal:
            plt.plot(dots, [v, v], color='lightgray', linewidth=1, zorder=0)
        else:
            plt.plot([v, v], dots, color='lightgray', linewidth=1, zorder=0)

    # plt.xlabel('xlabel')
    # plt.ylabel(ylabel')
    # plt.xticks(rotation=90, ha='right')
    # plt.title('title')
    # plt.legend(title='年度', labels=[f'{threshold_year}年以降', f'{threshold_year}年以前'])
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = do_load('healthexp')
    do_plot_median(df, outline='horizontal')
