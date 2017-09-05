import colorsys

import pandas as pd
from scipy.stats import beta
from matplotlib import pyplot as plt
import math
import numpy as np
import matplotlib.patches as mpatches


# Female-only colleges
excluded_colleges = ["Lucy Cavendish College", "Murray Edwards College", "Open offers"]


def rgb_tuble_to_html(color):
    eight_byte_color = [int(math.floor(c * 255.)) for c in color]
    return "#{:02x}{:02x}{:02x}".format(*eight_byte_color)


def load_bound_df(year):
    df = pd.read_csv("data/{}.csv".format(year))
    df = df[list(set(df.columns) - set(excluded_colleges))]
    df = df.set_index("-")
    df = df.T
    df["Total Applications"] = df["Direct applications"] + df["Open applications"]
    # df["Total Offers"] = df["Direct offers"] + df["Pool offers by other Colleges"]
    df["Total Rejections"] = df["Total Applications"] - df["Direct offers"]
    # Getting the 80% confidence interval using a Beta distribution
    year_intervals = [(college, beta.ppf([0.1, 0.9], row["Direct offers"] + 1, row["Total Rejections"] + 1))
                      for college, row in df.iterrows()]
    return pd.DataFrame([(c, l, u) for (c, (l, u)) in year_intervals], columns=["college", "lower", "upper"])


rank_dfs = []
for year in range(2013, 2018):
    df = pd.read_csv("data/{}.csv".format(year))
    df = df[list(set(df.columns) - set(excluded_colleges))]
    df = df.set_index("-")
    df = df.T
    df["Total Applications"] = df["Direct applications"] + df["Open applications"]
    # df["Total Offers"] = df["Direct offers"] + df["Pool offers by other Colleges"]
    df["Total Rejections"] = df["Total Applications"] - df["Direct offers"]
    df["prob"] = df["Direct offers"] / df["Total Applications"]
    df = df.sort_values("prob", ascending=False)
    df["rank"] = range(len(df.index))
    df = df[["rank"]].T
    df.index = [year]
    rank_dfs.append(df)

ranks = pd.concat(rank_dfs)
i = 0

for year in [2017, 2016, 2015]:
    year_intervals = load_bound_df(year)
    year_2017_overview_plot = (year_intervals.set_index("college").sort_values("lower").plot(kind="bar", title="{} Overview".format(year)))
    plt.show()

years_to_analyze = [2013, 2014, 2015, 2016, 2017]
interval_dfs = []
for year in years_to_analyze:
    intervals = load_bound_df(year)
    intervals["year"] = str(year)
    interval_dfs.append(intervals)

interval_df = pd.concat(interval_dfs)
ax = plt.axes()
# Picked these based on the overview of 2015,2016,2017
df = load_bound_df(2016)
df = df[df["upper"] - df["lower"] < 0.3]
selected_colleges = list(df.set_index("college").sort_values("lower").tail(10).index.values)
selected_colleges = set(selected_colleges) - {"Churchill College", "Christ's College", "Trinity College", "Corpus Christi College", "St Catharine's College"}
legend_patches = []
for college, color_index in zip(selected_colleges, np.linspace(0, 1., num=len(selected_colleges), endpoint=False)):
    college_vals = interval_df[interval_df["college"] == college].set_index("year")
    color = rgb_tuble_to_html(colorsys.hsv_to_rgb(color_index, 0.8, 0.8))
    college_vals["lower"].plot(style=":", ax=ax, color=color)
    college_vals["upper"].plot(style="-", ax=ax, color=color)
    legend_patches.append(mpatches.Patch(color=color, label=college))

plt.legend(handles=legend_patches)
plt.show()