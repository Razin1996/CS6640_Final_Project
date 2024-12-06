import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load ground truth data, skip comment lines, and replace missing value markers with NaN
df_gt = pd.read_csv("BSF_CONF_BA_SourceID_1_QC_0_Year_2024.csv", sep=',', comment='#', na_values=-9999)
df_gt.columns = df_gt.columns.str.strip()  # Remove whitespace from column names

# Load ROI data
df_roi = pd.read_csv("water_pixels_BSF.csv", sep=',')
df_roi['Timestamp'] = pd.to_datetime(df_roi['Timestamp'])
df_roi.set_index('Timestamp', inplace=True)
df_roi = df_roi[~(df_roi == 0).any(axis=1)]
df_roi_reset = df_roi.reset_index()

# Parse 'DateTime' column to datetime
df_gt['LocalDateTime'] = pd.to_datetime(df_gt['LocalDateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')  # Adjust the format if needed

# Filter the ground truth DataFrame for the desired date range and time interval
start_date = '2024-03-22'
end_date = '2024-04-07'
df_gt_filtered = df_gt[(df_gt['LocalDateTime'] >= start_date) & (df_gt['LocalDateTime'] <= end_date)]

# Filter to get every hour starting from the 30 minute mark
hourly_data_gt = df_gt_filtered[df_gt_filtered['LocalDateTime'].dt.minute == 30]

columns_to_remove = ['UTCOffset', 'DateTimeUTC', 'BattVolt',
        'Discharge_cms', 'EXOTime', 'EXOVolt', 'ODO',
       'ODO_Local', 'ODO_Sat', 'RH_enc', 'Rain_Tot', 'SpCond','StageNaNCounter', 
        'StageOffset', 'TurbAvg', 'TurbBES', 'TurbMax','TurbMed', 'TurbMin', 'TurbVar', 
        'TurbWipe', 'WaterSurfaceElev','WaterTemp_EXO', 'WaterTemp_PT', 'WaterTemp_Turb', 
        'pH']
hourly_data_gt.drop(columns_to_remove, axis=1, inplace=True)

# Merge the ground truth and ROI data on the datetime, using nearest available match
merged_df = pd.merge_asof(hourly_data_gt.sort_values('LocalDateTime'),
                          df_roi_reset.sort_values('Timestamp'),
                          left_on='LocalDateTime',
                          right_on='Timestamp',
                          direction='nearest')
merged_df['average_Bank_Value'] = (merged_df['LeftBank'] + merged_df['RightBank_1']) / 2

# Define the window size for smoothing
window_size_pixel = 15
window_size_gt = 5

# Smooth the 'Left Bank' and 'Right Bank' data
merged_df['Smoothed_Left_Bank'] = merged_df['LeftBank'].rolling(window=window_size_pixel, min_periods=1, center=True).mean()
merged_df['Smoothed_Right_Bank_1'] = merged_df['RightBank_1'].rolling(window=window_size_pixel, min_periods=1, center=True).mean()
merged_df['Smoothed_Right_Bank_2'] = merged_df['RightBank_2'].rolling(window=window_size_pixel, min_periods=1, center=True).mean()
merged_df['Smoothed_Stage'] = merged_df['Stage'].rolling(window=window_size_gt, min_periods=1, center=True).mean()

# Day/Night
merged_df['Day/Night'] = np.where((merged_df['LocalDateTime'].dt.hour >= 6) & (merged_df['LocalDateTime'].dt.hour < 18), 'Day', 'Night')
merged_df = merged_df[merged_df['Day/Night'] != 'Night']
plt.rcParams['font.size'] = 14
# Plot the smoothed data
fig, ax1 = plt.subplots(figsize=(16, 7))


# Groundtruth water stage value
color = 'tab:green'
ax1.set_xlabel('LocalDateTime', fontsize=18)
ax1.set_ylabel('Groundtruth Water Stage Value (cm)', fontsize=18)
lns1 = ax1.plot(merged_df['LocalDateTime'], merged_df['Smoothed_Stage'], color=color, linestyle='-.', label='Groundtruth Value')
ax1.tick_params(axis='y', labelsize=18)
ax1.tick_params(axis='x', labelsize=18, rotation=20)

# Instantiating a second axes that shares the same x-axis for water pixels
ax2 = ax1.twinx()
color1 = 'tab:blue'
color2 = 'tab:red'
ax2.set_ylabel('Water Pixels', fontsize=18)
#lns2 = ax2.plot(merged_df['Timestamp'], merged_df['Smoothed_Left_Bank'], color=color1, label='Smoothed Left Bank')
lns3 = ax2.plot(merged_df['LocalDateTime'], merged_df['Smoothed_Right_Bank_1'], color=color2, label='Smoothed Stream Bank')
ax2.tick_params(axis='y', labelsize=18)

# Combining the legends from both axes
lns = lns1 + lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left')

# Adjust layout to prevent clipping of ylabel
fig.tight_layout()
fig.savefig('banksVSgt_BSF.png', dpi=300)
plt.show()

########### Plotting each bank with the ground truth as a subplot with two y-axes
fig, axs = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

# Smoothed Left Bank vs Ground Truth
ax1 = axs[0].twinx()
axs[0].plot(merged_df['LocalDateTime'], merged_df['Smoothed_Left_Bank'], color='blue', label='Smoothed Left Bank')
ax1.plot(merged_df['LocalDateTime'], merged_df['Smoothed_Stage'], color='orange', label='Ground Truth', linestyle='--')
axs[0].set_ylabel('No. of Water Pixels')
ax1.set_ylabel('Water Stage (cm)')
lines, labels = axs[0].get_legend_handles_labels()
lines2, labels2 = ax1.get_legend_handles_labels()
axs[0].legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0, 1), ncol=1)
axs[0].set_title('Smoothed Left Bank vs Ground Truth')

# Smoothed Right Bank 1 vs Ground Truth
ax2 = axs[1].twinx()
axs[1].plot(merged_df['LocalDateTime'], merged_df['Smoothed_Right_Bank_1'], color='crimson', label='Smoothed Right Bank 1')
ax2.plot(merged_df['LocalDateTime'], merged_df['Smoothed_Stage'], color='orange', label='Ground Truth', linestyle='--')
axs[1].set_ylabel('No. of Water Pixels')
ax2.set_ylabel('Water Stage (cm)')
lines, labels = axs[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axs[1].legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0, 1), ncol=1)
axs[1].set_title('Smoothed Right Bank 1 vs Ground Truth')

# Setting the x-label only on the last subplot
axs[1].set_xlabel('LocalDateTime')
axs[1].tick_params(axis='x', labelsize=18, rotation=20)

# Adjust layout to prevent overlap
plt.tight_layout()

plt.savefig("individual_bankVSgt_BSF.png", dpi=300)
# Show the plot
plt.show()