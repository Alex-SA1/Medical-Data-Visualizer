import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('medical_examination.csv', sep=',', header='infer')

df['overweight'] = (
    (df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype('int8')


df['cholesterol'] = (df['cholesterol'] > 1).astype('int8')
df['gluc'] = (df['gluc'] > 1).astype('int8')


def draw_cat_plot():
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=[
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    df_cat = pd.DataFrame(df_cat.groupby(by=['cardio']).value_counts())
    df_cat.rename(columns={'count': 'total'}, inplace=True)

    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio", order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'], kind="bar",
                      data=df_cat).figure

    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(
        0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(9, 7))

    # 15
    sns.heatmap(corr, annot=True, mask=mask, vmax=0.24, linewidth=.5,
                center=False, square=True, fmt='.1f', cbar_kws={'shrink': 0.5})

    # 16
    fig.savefig('heatmap.png')
    return fig


draw_heat_map()
