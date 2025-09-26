# Analysis and visualization of accident data focusing on road conditions, weather and time-of-day (light conditions).
# This will:
# - clean and prepare the data
# - produce counts and pivot tables
# - create matplotlib visualizations (saved to /mnt/data/) and display them
# - run chi-square tests for association between conditions and accident severity
# Note: dataset has no geographic coordinates, so "hotspots" are inferred as high-frequency categories (junctions, lanes, road types).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import chi2_contingency
import os

# Load
df = pd.read_csv("C:/Users/nanda/OneDrive/Documents/Pratap's Tasks/4/cleaned.csv")

# Key columns we'll use
cols_of_interest = ['Road_surface_type', 'Weather_conditions', 'Light_conditions',
                    'Types_of_Junction', 'Lanes_or_Medians', 'Cause_of_accident',
                    'Accident_severity', 'Type_of_collision']

missing_before = df[cols_of_interest].isnull().sum()

# Drop rows missing any key field for the main analysis
df_clean = df.dropna(subset=cols_of_interest).copy()

# Convert Accident_severity to a consistent categorical ordering if possible
if df_clean['Accident_severity'].dtype == object:
    # keep as categorical by frequency order
    df_clean['Accident_severity'] = pd.Categorical(df_clean['Accident_severity'],
                                                   categories=sorted(df_clean['Accident_severity'].unique()),
                                                   ordered=False)

# 1) Overall counts
counts_weather = df_clean['Weather_conditions'].value_counts()
counts_road = df_clean['Road_surface_type'].value_counts()
counts_light = df_clean['Light_conditions'].value_counts()
counts_junction = df_clean['Types_of_Junction'].value_counts()
counts_lanes = df_clean['Lanes_or_Medians'].value_counts()
counts_cause = df_clean['Cause_of_accident'].value_counts().head(15)

# Create output directory
out_dir = '/mnt/data/accident_analysis_outputs'
os.makedirs(out_dir, exist_ok=True)

# Plot 1: Top weather conditions (bar)
plt.figure(figsize=(8,5))
ax = plt.gca()
counts_weather.plot(kind='bar', ax=ax)
ax.set_title('Accident count by Weather conditions')
ax.set_ylabel('Number of accidents')
ax.set_xlabel('Weather conditions')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'weather_counts.png'))
plt.show()

# Plot 2: Road surface types
plt.figure(figsize=(8,5))
ax = plt.gca()
counts_road.plot(kind='bar', ax=ax)
ax.set_title('Accident count by Road surface type')
ax.set_ylabel('Number of accidents')
ax.set_xlabel('Road surface type')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'road_surface_counts.png'))
plt.show()

# Plot 3: Light conditions (proxy for time of day)
plt.figure(figsize=(8,5))
ax = plt.gca()
counts_light.plot(kind='bar', ax=ax)
ax.set_title('Accident count by Light conditions (proxy for time of day)')
ax.set_ylabel('Number of accidents')
ax.set_xlabel('Light conditions')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'light_conditions_counts.png'))
plt.show()

# Plot 4: Top causes (horizontal bar)
plt.figure(figsize=(8,6))
ax = plt.gca()
counts_cause.sort_values().plot(kind='barh', ax=ax)
ax.set_title('Top contributing causes of accidents (top 15)')
ax.set_xlabel('Number of accidents')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'top_causes.png'))
plt.show()

# Plot 5: Stacked bar of Accident_severity by Weather
pivot_w_s = pd.crosstab(df_clean['Weather_conditions'], df_clean['Accident_severity'])
# normalize by row to see distribution
pivot_w_s_norm = pivot_w_s.div(pivot_w_s.sum(axis=1), axis=0).fillna(0)

plt.figure(figsize=(10,6))
pivot_w_s_norm.plot(kind='bar', stacked=True)
plt.title('Proportion of Accident severity by Weather condition (stacked, normalized)')
plt.ylabel('Proportion of accidents')
plt.xlabel('Weather conditions')
plt.legend(title='Accident severity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'severity_by_weather_stacked.png'))
plt.show()

# Plot 6: Heatmap-like visualization for Weather vs Road surface (counts)
pivot_heat = pd.crosstab(df_clean['Weather_conditions'], df_clean['Road_surface_type'])
plt.figure(figsize=(10,6))
plt.imshow(pivot_heat, aspect='auto', origin='lower')
plt.title('Accident counts: Weather vs Road surface type (heatmap)')
plt.xlabel('Road surface type (index order)')
plt.ylabel('Weather conditions (index order)')
plt.xticks(ticks=np.arange(len(pivot_heat.columns)), labels=pivot_heat.columns, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(pivot_heat.index)), labels=pivot_heat.index)
plt.colorbar(label='Accident counts')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'weather_vs_road_heatmap.png'))
plt.show()

# Statistical tests (chi-square) for association between categorical variables and severity
tests = {}
def run_chi2(var):
    table = pd.crosstab(df_clean[var], df_clean['Accident_severity'])
    chi2, p, dof, exp = chi2_contingency(table)
    return {'chi2': chi2, 'p_value': p, 'dof': dof, 'table_shape': table.shape}

for var in ['Weather_conditions', 'Road_surface_type', 'Light_conditions', 'Types_of_Junction']:
    tests[var] = run_chi2(var)

# Create summary dataframe for tests
tests_df = pd.DataFrame.from_dict(tests, orient='index')
tests_df = tests_df.reset_index().rename(columns={'index':'variable'})

# Save a CSV summary and the cleaned dataset sample
df_clean.to_csv(os.path.join(out_dir, 'cleaned_for_analysis.csv'), index=False)
tests_df.to_csv(os.path.join(out_dir, 'chi2_tests_summary.csv'), index=False)

# Display some useful outputs back to the user
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Counts: Weather (top 50)", counts_weather.reset_index().rename(columns={'index':'Weather','Weather_conditions':'count'}))
display_dataframe_to_user("Counts: Road Surface (top 50)", counts_road.reset_index().rename(columns={'index':'Road','Road_surface_type':'count'}))
display_dataframe_to_user("Chi-square test results (Weather/Road/Light/Types_of_Junction vs Accident_severity)", tests_df)

print("Saved plots and CSV summaries to:", out_dir)
os.listdir(out_dir)[:50]

