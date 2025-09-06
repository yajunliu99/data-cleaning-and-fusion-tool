import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from pykalman import KalmanFilter
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


class ODMatrixAnalyzerTP:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initialize with the database path, output folder, and optional stop event."""
        self.database_path = database_path
        self.output_folder = output_folder

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event

    def load_data(self):
        """
        Load node, link, and trip path data from the SQLite database.

        Reads:
         - node table (node information)
         - link table (road link data)
         - trajs table (trip path data) extracting TripId, SegmentId, CrossingStartDateLocal, CrossingEndDateLocal
         - SegmentId_to_link table to match SegmentId to link_id
        """
        conn = sqlite3.connect(self.database_path)
        node_df = pd.read_sql("SELECT node_id, is_boundary FROM node", conn)
        link_df = pd.read_sql("SELECT link_id, from_node_id, to_node_id FROM link", conn)
        trip_df = pd.read_sql(
            "SELECT TripId, SegmentId, CrossingStartDateLocal, CrossingEndDateLocal FROM trajs", conn)
        segment_link_df = pd.read_sql("SELECT SegmentId, link_id FROM SegmentId_to_link", conn)
        trip_df = pd.merge(trip_df, segment_link_df, on="SegmentId", how="left")
        conn.close()
        return node_df, link_df, trip_df

    def process_od_links(self, node_df, link_df):
        """
        Identify origin and destination links based on the node's is_boundary values.

        - is_boundary = 1 or 2 -> Origin (matches with from_node_id)
        - is_boundary = -1 or 2 -> Destination (matches with to_node_id)
        """
        origins = node_df[node_df["is_boundary"].isin([1, 2])]
        destinations = node_df[node_df["is_boundary"].isin([-1, 2])]
        origin_links = link_df[link_df["from_node_id"].isin(origins["node_id"])]
        origin_links = origin_links[["from_node_id", "link_id"]].rename(columns={"from_node_id": "node_id"})
        destination_links = link_df[link_df["to_node_id"].isin(destinations["node_id"])]
        destination_links = destination_links[["to_node_id", "link_id"]].rename(columns={"to_node_id": "node_id"})
        return origin_links, destination_links, link_df

    def compute_od_matrix(self, trip_df, origin_links, destination_links, link_df):
        """
        Compute the OD matrix by aggregating data over the entire day and splitting by is_weekend flag.

        Steps:
         - Filter trip data for origin and destination links.
         - Merge origin and destination records using TripId and ensure the origin time is earlier.
         - Extract the date and is_weekend flag from the origin timestamp.
         - For each day, count the unique TripIds per OD pair and then average these daily counts over days.
         - Map the link_ids to node_ids.
         - Compute the origin proportion (avg_volume divided by total volume from the origin zone).
        """
        # Filter trip data for origin and destination links
        matched_origins = trip_df[trip_df["link_id"].isin(origin_links["link_id"])]
        matched_destinations = trip_df[trip_df["link_id"].isin(destination_links["link_id"])]
        valid_trips = set(matched_origins["TripId"]) & set(matched_destinations["TripId"])
        matched_origins = matched_origins[matched_origins["TripId"].isin(valid_trips)]
        matched_destinations = matched_destinations[matched_destinations["TripId"].isin(valid_trips)]

        # Merge origin and destination records using TripId
        od_pairs = pd.merge(matched_origins, matched_destinations, on="TripId", suffixes=("_origin", "_dest"))

        # Ensure the origin's CrossingStartDateLocal is earlier than the destination's CrossingEndDateLocal
        od_pairs = od_pairs[pd.to_datetime(od_pairs["CrossingStartDateLocal_origin"]) <
                            pd.to_datetime(od_pairs["CrossingStartDateLocal_dest"])]

        # Extract date and is_weekend flag from the origin timestamp
        od_pairs["date"] = pd.to_datetime(od_pairs["CrossingStartDateLocal_origin"]).dt.date
        od_pairs["is_weekend"] = pd.to_datetime(od_pairs["CrossingStartDateLocal_origin"]).dt.weekday >= 5

        # Compute daily volume per OD pair by grouping by is_weekend, date, and link ids
        daily_volumes = (
            od_pairs.groupby(["is_weekend", "date", "link_id_origin", "link_id_dest"])["TripId"]
            .nunique()
            .reset_index(name="daily_volume")
        )

        # Average daily volume over days for each is_weekend flag and OD pair
        od_matrix_avg = (
            daily_volumes.groupby(["is_weekend", "link_id_origin", "link_id_dest"])["daily_volume"]
            .mean()
            .reset_index()
            .rename(columns={"daily_volume": "avg_volume"})
        )
        # Round the average volume to integer values
        od_matrix_avg["avg_volume"] = od_matrix_avg["avg_volume"].round(0).astype(int)

        # Map link ids to node ids
        link_node_map = link_df.set_index("link_id")[["from_node_id", "to_node_id"]]
        od_matrix_avg["o_zone_id"] = od_matrix_avg["link_id_origin"].map(link_node_map["from_node_id"])
        od_matrix_avg["d_zone_id"] = od_matrix_avg["link_id_dest"].map(link_node_map["to_node_id"])

        # Compute total average volume per origin zone for each is_weekend flag to calculate origin proportion
        origin_totals = (
            od_matrix_avg.groupby(["is_weekend", "o_zone_id"])["avg_volume"]
            .sum()
            .reset_index(name="total_volume")
        )
        od_matrix_avg = pd.merge(od_matrix_avg, origin_totals, on=["is_weekend", "o_zone_id"])
        od_matrix_avg["o_proportion"] = (od_matrix_avg["avg_volume"] / od_matrix_avg["total_volume"]).round(3)

        return od_matrix_avg

    def plot_od_matrix(self, od_matrix_avg):
        """
        Generate and save a heatmap for each is_weekend flag.

        For each is_weekend value (True for weekend, False for weekday) where data exists,
        a heatmap is produced showing the average volume (avg_volume) per OD pair.
        The data values are displayed as integers.
        """
        if self.stop_event and self.stop_event.is_set():
            print("Stopping before plotting OD matrix")
            return

        print("Plotting OD Matrix Heatmaps...")
        for is_weekend in od_matrix_avg["is_weekend"].unique():
            day_label = "Weekend" if is_weekend else "Weekday"
            subset = od_matrix_avg[od_matrix_avg["is_weekend"] == is_weekend]
            if subset.empty:
                continue
            pivot_table = subset.pivot(index="o_zone_id", columns="d_zone_id", values="avg_volume").fillna(0)
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, cmap="YlOrRd", linewidths=0.5, annot=True, annot_kws={"size": 12}, fmt='.0f',
                        cbar=True)
            plt.xlabel("Destinations", fontsize=18)
            plt.ylabel("Origins", fontsize=18)
            plt.title(f"OD Matrix Heatmap Trip Path {day_label}", fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            # plt.show()

            output_dir = os.path.join(self.output_folder, "trend_analysis_cbi", "od_matrix_tp")
            os.makedirs(output_dir, exist_ok=True)

            # Save figure
            filename = f"od_matrix_heatmap_tp_{'weekend' if is_weekend else 'weekday'}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()

    def save_to_csv(self, od_matrix):
        """Save the aggregated OD matrix to a CSV file."""
        od_matrix = od_matrix.drop(columns=["link_id_origin", "link_id_dest"])
        od_matrix["is_weekend"] = od_matrix["is_weekend"].astype(int)
        od_matrix = od_matrix[["o_zone_id", "d_zone_id", "is_weekend", "avg_volume", "total_volume", "o_proportion"]]

        # Define output directory
        output_dir = os.path.join(self.output_folder, "trend_analysis_cbi", "od_matrix_tp")
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, "od_volume_tp.csv")
        od_matrix.to_csv(filepath, encoding="utf-8-sig", index=False)

    def run(self):
        """Run the OD matrix analysis pipeline for Trip Path data aggregated by is_weekend."""
        if self.stop_event and self.stop_event.is_set():
            return
        node_df, link_df, trip_df = self.load_data()

        if self.stop_event and self.stop_event.is_set():
            return
        origin_links, destination_links, link_df = self.process_od_links(node_df, link_df)

        if self.stop_event and self.stop_event.is_set():
            return
        od_matrix = self.compute_od_matrix(trip_df, origin_links, destination_links, link_df)

        if self.stop_event and self.stop_event.is_set():
            return
        self.save_to_csv(od_matrix)

        if self.stop_event and self.stop_event.is_set():
            return
        self.plot_od_matrix(od_matrix)
        print("OD Matrix Heatmaps saved.")


class ODTravelTimeAnalyzerTP:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initialize with the database path, output folder, and optional stop event."""
        self.database_path = database_path
        self.output_folder = output_folder
        self.output_file = os.path.join(output_folder, "od_travel_time.csv")
        self.stop_event = stop_event

    def load_data(self):
        """
        Load node, link, and trip path data from the SQLite database.

        Similar to ODMatrixAnalyzerTP.load_data(), but used for travel time calculations.
        """
        conn = sqlite3.connect(self.database_path)

        # Read node data
        node_df = pd.read_sql("SELECT node_id, is_boundary FROM node", conn)

        # Read link data
        link_df = pd.read_sql("SELECT link_id, from_node_id, to_node_id FROM link", conn)

        # Read trip path data and merge with SegmentId_to_link for link_id
        trip_df = pd.read_sql(
            "SELECT TripId, SegmentId, CrossingStartDateLocal, CrossingEndDateLocal FROM trajs", conn)
        segment_link_df = pd.read_sql(
            "SELECT SegmentId, link_id FROM SegmentId_to_link", conn)
        trip_df = pd.merge(trip_df, segment_link_df, on="SegmentId", how="left")

        conn.close()
        return node_df, link_df, trip_df

    def process_od_links(self, node_df, link_df):
        """
        Identify origin and destination links based on the node's is_boundary values.

        This processing is the same as in ODMatrixAnalyzerTP.
        """
        origins = node_df[node_df["is_boundary"].isin([1, 2])]
        destinations = node_df[node_df["is_boundary"].isin([-1, 2])]

        origin_links = link_df[link_df["from_node_id"].isin(origins["node_id"])]
        origin_links = origin_links[["from_node_id", "link_id"]].rename(columns={"from_node_id": "node_id"})

        destination_links = link_df[link_df["to_node_id"].isin(destinations["node_id"])]
        destination_links = destination_links[["to_node_id", "link_id"]].rename(columns={"to_node_id": "node_id"})

        return origin_links, destination_links, link_df

    def compute_od_travel_time(self, trip_df, origin_links, destination_links, link_df):
        """
        Compute OD travel time for each TripId.

        - Filters records that appear at both origin and destination links.
        - Computes travel time as the difference (in minutes) between the destination's CrossingEndDateLocal
          and the origin's CrossingStartDateLocal.
        - Maps link_id to corresponding node_id for OD grouping.
        """
        # Convert CrossingStartDateLocal and CrossingEndDateLocal to datetime
        trip_df["CrossingStartDateLocal"] = pd.to_datetime(trip_df["CrossingStartDateLocal"])
        trip_df["CrossingEndDateLocal"] = pd.to_datetime(trip_df["CrossingEndDateLocal"])

        # Filter data for origin and destination links
        matched_origins = trip_df[trip_df["link_id"].isin(origin_links["link_id"])]
        matched_destinations = trip_df[trip_df["link_id"].isin(destination_links["link_id"])]

        valid_trips = set(matched_origins["TripId"]) & set(matched_destinations["TripId"])
        matched_origins = matched_origins[matched_origins["TripId"].isin(valid_trips)]
        matched_destinations = matched_destinations[matched_destinations["TripId"].isin(valid_trips)]

        # Merge origin and destination records using TripId, adding suffixes
        od_pairs = pd.merge(matched_origins, matched_destinations, on="TripId", suffixes=("_origin", "_dest"))

        # Ensure the origin's CrossingStartDateLocal is before the destination's CrossingEndDateLocal
        od_pairs = od_pairs[pd.to_datetime(od_pairs["CrossingStartDateLocal_origin"]) <
                            pd.to_datetime(od_pairs["CrossingEndDateLocal_dest"])]

        # Compute travel time in minutes
        od_pairs["travel_time"] = (od_pairs["CrossingEndDateLocal_dest"] - od_pairs["CrossingStartDateLocal_origin"]).dt.total_seconds() / 60

        # Map link_id to node_id (origin: from_node_id, destination: to_node_id)
        link_node_map = link_df.set_index("link_id")[["from_node_id", "to_node_id"]]
        od_pairs["o_zone_id"] = od_pairs["link_id_origin"].map(link_node_map["from_node_id"])
        od_pairs["d_zone_id"] = od_pairs["link_id_dest"].map(link_node_map["to_node_id"])

        # Use the origin CrossingStartDateLocal as the time axis for plotting
        od_pairs["start_time"] = od_pairs["CrossingStartDateLocal_origin"]

        # Select the output columns
        od_travel_times = od_pairs[["TripId", "o_zone_id", "d_zone_id", "travel_time", "start_time"]]

        return od_travel_times

    def plot_od_travel_times(self, od_travel_times):
        """
        Plot aggregated OD travel times as bar charts for selected OD pairs.

        All travel time data (from all dates) is aggregated into one day by mapping the
        start_time to a dummy date (1900-01-01) so that only the time-of-day remains.
        The x-axis will show time-of-day only.
        """
        print("Plotting Travel Time Profile...")

        # Select specific OD pairs
        od_pairs = [(36, 28), (27, 37), (35, 25), (25, 35)]
        filtered_od_travel_times = od_travel_times[
            od_travel_times.apply(lambda row: (row["o_zone_id"], row["d_zone_id"]) in od_pairs, axis=1)
        ].copy()

        colors = ['#4472C5', '#E90E01', '#1C841D', '#9A6600']
        od_labels = {
            (36, 28): 'Northbound',
            (27, 37): 'Southbound',
            (35, 25): 'Express Lane Northbound',
            (25, 35): 'Express Lane Southbound'
        }

        for (o_zone, d_zone), color in zip(od_labels.keys(), colors):
            if self.stop_event and self.stop_event.is_set():
                print(f"Stopping at OD pair ({o_zone}, {d_zone})")
                return

            subset = filtered_od_travel_times[
                (filtered_od_travel_times["o_zone_id"] == o_zone) &
                (filtered_od_travel_times["d_zone_id"] == d_zone)
                ].copy()
            if subset.empty:
                print(f"Warning: No valid data found for OD pair ({o_zone}, {d_zone}). Skipping...")
                continue

            # Map all start_time values to a dummy date (1900-01-01) so that only the time-of-day remains.
            subset.loc[:, "dummy_time"] = subset["start_time"].apply(lambda x: x.replace(year=1900, month=1, day=1).to_pydatetime())
            # Sort by the dummy time
            subset = subset.sort_values("dummy_time")

            # Extract start times and travel times for plotting
            start_times = subset["dummy_time"]
            travel_times = subset["travel_time"]

            # Compute outlier ratio using the original travel times
            kf = KalmanFilter(
                initial_state_mean=travel_times.iloc[0],
                transition_matrices=[1],
                observation_matrices=[1],
                transition_covariance=np.array([[0.5]]),
                observation_covariance=np.array([[0.01]])
            )
            state_means, state_covariances = kf.filter(travel_times.ffill().bfill())
            smoothed_travel_times = state_means.flatten()
            state_std = np.sqrt(state_covariances.flatten())
            upper_bound_smooth = smoothed_travel_times + 3 * state_std
            lower_bound_smooth = smoothed_travel_times - 3 * state_std
            out_of_bounds = (travel_times < lower_bound_smooth) | (travel_times > upper_bound_smooth)
            outlier_ratio = np.sum(out_of_bounds) / len(travel_times) * 100
            print(f"OD pair ({o_zone}, {d_zone}): {outlier_ratio:.2f}% of data are outside the confidence interval.")

            plt.figure(figsize=(12, 8))
            bar_width = 0.0005

            # Plot original travel times
            plt.bar(start_times, travel_times, color=color, width=bar_width, label=od_labels[(o_zone, d_zone)])

            plt.xlabel('Time of Day', fontsize=18)
            plt.ylabel('Average Travel Time (min)', fontsize=18)
            plt.title(f'Corridor End-to-End Travel Time Trip Path: {od_labels[(o_zone, d_zone)]}', fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(fontsize=14)
            plt.grid(axis='y', linestyle='--')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            # Set x-axis and y-axis limits.
            # start_time = subset["dummy_time"].min()  # Earliest timestamp
            # end_time = subset["dummy_time"].max()  # Latest timestamp
            start_time = datetime(1900, 1, 1, 0, 0, 12)
            end_time = datetime(1900, 1, 1, 23, 59, 53)
            plt.xlim(start_time, end_time)
            plt.ylim(0, 30)

            plt.tight_layout()
            # plt.show()

            output_dir = os.path.join(self.output_folder, "trend_analysis_cbi", "od_travel_time_tp")
            os.makedirs(output_dir, exist_ok=True)

            # Save figure
            filename = f"corridor_travel_time_tp: {od_labels[(o_zone, d_zone)].lower().replace(' ', '_')}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()

    def run(self):
        """Run the OD travel time analysis and visualization for Trip Path data."""
        if self.stop_event and self.stop_event.is_set():
            return
        node_df, link_df, trip_df = self.load_data()

        if self.stop_event and self.stop_event.is_set():
            return
        origin_links, destination_links, link_df = self.process_od_links(node_df, link_df)

        if self.stop_event and self.stop_event.is_set():
            return
        od_travel_times = self.compute_od_travel_time(trip_df, origin_links, destination_links, link_df)

        if self.stop_event and self.stop_event.is_set():
            return
        self.plot_od_travel_times(od_travel_times)

        print("Travel Time Profile saved")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")
    DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "output", "database", "unified_database.db")

    od_analyzer_tp = ODMatrixAnalyzerTP(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    od_analyzer_tp.run()

    odtt_analyzer_tp = ODTravelTimeAnalyzerTP(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    odtt_analyzer_tp.run()