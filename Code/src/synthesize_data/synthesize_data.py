import os
import pandas as pd
import numpy as np
from hashlib import sha256
from datetime import timedelta
from tqdm import tqdm


# # Waypoint
# # Set folders
# input_folder = '../../synthesize_data/waypoint'
# output_folder = '../../synthesize_data/waypoint'
#
# # Set parameters
# lat_std = 0.000005            # Latitude noise
# elev_std = 3                  # Elevation noise (ft)
# speed_std = 5                 # Speed noise (mph)
#
# np.random.seed(42)  # Ensure reproducibility
#
# # List all CSV files in the input folder
# csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
#
# # Process each file
# for filename in tqdm(csv_files, desc="Processing Waypoint Data"):
#     input_path = os.path.join(input_folder, filename)
#     df = pd.read_csv(input_path)
#
#     # Calculate longitude noise adjusted for latitude
#     lon_std = lat_std / np.cos(np.deg2rad(df['latitude'].mean()))
#
#     # Add Gaussian noise to coordinates, elevation, and speed
#     df['latitude'] = df['latitude'] + np.random.normal(0, lat_std, len(df))
#     df['longitude'] = df['longitude'] + np.random.normal(0, lon_std, len(df))
#
#     mask_elev = df['elevation_ft'].notnull() & (df['elevation_ft'] != 0)
#     noise_elev = np.random.normal(0, elev_std, mask_elev.sum())
#     df.loc[mask_elev, 'elevation_ft'] = np.round(np.maximum(0, df.loc[mask_elev, 'elevation_ft'] + noise_elev), 0)
#
#     mask_speed = df['speed_mph'].notnull() & (df['speed_mph'] != 0)
#     noise_speed = np.random.normal(0, speed_std, mask_speed.sum())
#     df.loc[mask_speed, 'speed_mph'] = np.round(np.maximum(0, df.loc[mask_speed, 'speed_mph'] + noise_speed), 0)
#
#     # Randomize journey_id using SHA-256 hash (same length)
#     unique_ids = df['journey_id'].unique()
#     randomized_map = {
#         jid: sha256((jid + str(np.random.randint(0, 1e9))).encode()).hexdigest()[:len(jid)]
#         for jid in unique_ids
#     }
#     df['journey_id'] = df['journey_id'].map(randomized_map)
#
#     # Perturb capture_time per journey while preserving relative order
#     def perturb_time(group):
#         noise = int(np.random.normal(0, 5))  # seconds
#         group['capture_time'] = group['capture_time'] + noise
#         return group
#
#     grouped = df.groupby('journey_id', sort=False)
#     df_list = [perturb_time(group) for _, group in grouped]
#     df_final = pd.concat(df_list, ignore_index=True)
#
#     # Keep original column order and round numeric fields
#     df_final = df_final[df.columns.tolist()]
#     df_final['latitude'] = df_final['latitude'].round(6)
#     df_final['longitude'] = df_final['longitude'].round(6)
#     # df_final = df_final.dropna(subset=['elevation_ft'])
#     # df_final['elevation_ft'] = df_final['elevation_ft'].round(0).astype(int)
#     # df_final = df_final.dropna(subset=['speed_mph'])
#     # df_final['speed_mph'] = df_final['speed_mph'].round(0).astype(int)
#
#     # Save processed output
#     output_path = os.path.join(output_folder, filename.replace('.csv', '_synthesized.csv'))
#     df_final.to_csv(output_path, index=False)
#
# print("All waypoint data have been processed and saved.")


# Trip Path
# Load headers
header_file = '../../synthesize_data/trip path/TripBulkReportTrajectoriesHeaders.csv'
with open(header_file, 'r') as f:
    header = f.read().strip().split(',')

# === Load original data (no header in file) ===
input_file = '../../synthesize_data/trip path/trajs.csv'
df = pd.read_csv(input_file, header=None, low_memory=False)
df.columns = header

# Parse datetime columns
df['CrossingStartDateUtc'] = pd.to_datetime(df['CrossingStartDateUtc'], utc=True)
df['CrossingEndDateUtc'] = pd.to_datetime(df['CrossingEndDateUtc'], utc=True)

# Set noise parameters
time_std = 3       # Time noise ±3 seconds
speed_std = 1.0    # Speed noise ±1 km/h

np.random.seed(42)  # Set random seed for reproducibility

# Randomize identifiers using SHA-256 hash (truncate to original length)
for col in ['TripId', 'DeviceId', 'ProviderId']:
    unique_ids = df[col].astype(str).unique()
    randomized_map = {
        uid: sha256((uid + str(np.random.randint(0, 1e9))).encode()).hexdigest()[:len(uid)]
        for uid in unique_ids
    }
    df[col] = df[col].astype(str).map(randomized_map)

# Add noise to speed
mask_cs = df['CrossingSpeedKph'].notnull() & (df['CrossingSpeedKph'] != 0)
noise_cs = np.random.normal(0, speed_std, mask_cs.sum())
df.loc[mask_cs, 'CrossingSpeedKph'] = np.maximum(0, df.loc[mask_cs, 'CrossingSpeedKph'] + noise_cs)

# Add the same time noise to start and end timestamps per row
def perturb_timestamps(row):
    delta = timedelta(seconds=np.random.normal(0, time_std))
    start = (row['CrossingStartDateUtc'] + delta).replace(tzinfo=None)
    end = (row['CrossingEndDateUtc'] + delta).replace(tzinfo=None)
    row['CrossingStartDateUtc'] = start.isoformat(timespec='milliseconds') + 'Z'
    row['CrossingEndDateUtc'] = end.isoformat(timespec='milliseconds') + 'Z'
    return row

tqdm.pandas(desc="Processing Trip Path Data")
df = df.progress_apply(perturb_timestamps, axis=1)

# Save synthesized output
output_file = '../../synthesize_data/trip path/trajs_synthesized.csv'
df.to_csv(output_file, index=False)
print(f"All trip path data have been processed and saved")


# # Probe OD Data
# # Load original data
# input_file = '../../synthesize_data/probe_od/od.csv'
# df = pd.read_csv(input_file)
#
# # Set noise parameters
# volume_std_pct = 0.02   # 2% Gaussian noise for volume columns
# time_std_sec = 5        # 5 seconds Gaussian noise for travel time
#
# np.random.seed(42)  # For reproducibility
#
# # Add noise to volume-related columns
# volume_columns = [
#     'Average Daily O-D Traffic (StL Volume)',
#     'Average Daily Origin Zone Traffic (StL Volume)',
#     'Average Daily Destination Zone Traffic (StL Volume)'
# ]
#
# for col in volume_columns:
#     std = df[col] * volume_std_pct
#     noise = np.random.normal(0, std)
#     df[col] = (df[col] + noise).round(0).astype('Int64')
#
# # Add noise to average travel time
# def perturb_travel_time(x):
#     if isinstance(x, str) and x.strip().upper() == 'N/A':
#         return 'N/A'
#     try:
#         val = float(x)
#         val_noised = val + np.random.normal(0, time_std_sec)
#         return round(val_noised, 1)
#     except:
#         return 'N/A'
#
#
# # Save synthesized output
# output_file = '../../synthesize_data/probe_od/od_synthesized.csv'
# df.to_csv(output_file, index=False, na_rep='N/A')
# print(f"All probe od data have been processed and saved")


# TMC Data
# Load original data
input_file = '../../synthesize_data/tmc_speed/Readings.csv'
df = pd.read_csv(input_file)

# Set noise parameters
speed_std = 5           # Speed noise: ±2.0 mph
travel_time_std = 1     # 5 seconds Gaussian noise for travel time

np.random.seed(42)  # For reproducibility

# Apply Gaussian noise to 'speed'
mask_speed = df['speed'].notna() & (df['speed'] != 0)
df.loc[mask_speed, 'speed'] = (df.loc[mask_speed, 'speed'] +
                               np.random.normal(0, speed_std, mask_speed.sum())).round(0)

# Apply relative Gaussian noise to 'travel_time_seconds'
mask_tt = df['travel_time_seconds'].notnull() & (df['travel_time_seconds'] != 0)
noise_tt = np.random.normal(0, travel_time_std, mask_tt.sum())
df.loc[mask_tt, 'travel_time_seconds'] = np.round(np.maximum(0, df.loc[mask_tt, 'travel_time_seconds'] + noise_tt), 2)

# Save synthesized output
output_file = '../../synthesize_data/tmc_speed/Readings_synthesized.csv'
df.to_csv(output_file, index=False)
print(f"All TMC data have been processed and saved")


# # Sensor Data
# # Load original data
# input_file = '../../synthesize_data/sensor/lane_readings.csv'
# df = pd.read_csv(input_file)
#
# # Convert measurement_start to datetime
# df['local_time'] = pd.to_datetime(
#     df['measurement_start'],
#     format='%Y-%m-%d %H:%M:%S%z',
#     errors='coerce')
#
# df['local_time'] = df['local_time'].dt.tz_localize(None)
#
# # Create 15-minute time bins
# df['time_bin'] = df['local_time'].dt.floor('15min')
#
# # Shuffle speed and volume within each group
# new_speed = df['speed'].copy()
# new_volume = df['volume'].copy()
# new_occupancy = df['occupancy'].copy()
#
# shuffled_groups = 0
# for _, group in df.groupby(['zone_id', 'lane_id', 'time_bin']):
#     idx = group.index
#     if len(idx) > 1:
#         shuffled = group.sample(frac=1)
#         new_speed.loc[idx] = shuffled['speed'].values
#         new_volume.loc[idx] = shuffled['volume'].values
#         new_occupancy.loc[idx] = shuffled['occupancy'].values
#         shuffled_groups += 1
#
# print(f"Total groups: {df.groupby(['zone_id','lane_id','time_bin']).ngroups}, Shuffled: {shuffled_groups}")
#
# df['speed'] = new_speed
# df['volume'] = new_volume
# df['occupancy'] = new_occupancy
#
# # Drop time_bin column or keep it for debug
# df.drop(columns=['local_time', 'time_bin'], inplace=True)
#
# # Save output
# output_file = '../../synthesize_data/sensor/lane_readings_synthesized.csv'
# df.to_csv(output_file, index=False)
# print(f"All sensor data have been processed and saved")
