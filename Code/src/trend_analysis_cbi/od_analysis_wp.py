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


class ODMatrixAnalyzerWP:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initialize the database path and stop event."""
        self.database_path = database_path
        self.output_folder = output_folder

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event

    def load_data(self):
        """Load node, link, and map_matching data from the SQLite database."""
        conn = sqlite3.connect(self.database_path)

        # Read node data
        node_df = pd.read_sql("SELECT node_id, is_boundary FROM node", conn)

        # Read link data
        link_df = pd.read_sql("SELECT link_id, from_node_id, to_node_id FROM link", conn)

        # Read map_matching data
        map_matching_df = pd.read_sql("SELECT agent_id, link_id, seq, time FROM map_matching", conn)

        conn.close()
        return node_df, link_df, map_matching_df

    def process_od_links(self, node_df, link_df):
        """
        Identify origin and destination links based on node is_boundary values.

        - is_boundary = 1 or 2 → Origin (match with from_node_id)
        - is_boundary = -1 or 2 → Destination (match with to_node_id)
        """
        origins = node_df[node_df["is_boundary"].isin([1, 2])]
        destinations = node_df[node_df["is_boundary"].isin([-1, 2])]

        # Get origin-related link_ids where from_node_id matches the node_id
        origin_links = link_df[link_df["from_node_id"].isin(origins["node_id"])]
        origin_links = origin_links[["from_node_id", "link_id"]].rename(columns={"from_node_id": "node_id"})

        # Get destination-related link_ids where to_node_id matches the node_id
        destination_links = link_df[link_df["to_node_id"].isin(destinations["node_id"])]
        destination_links = destination_links[["to_node_id", "link_id"]].rename(columns={"to_node_id": "node_id"})

        return origin_links, destination_links, link_df

    def compute_od_matrix(self, map_matching_df, origin_links, destination_links, link_df):
        """
        Compute the OD matrix using unique `agent_id` counts per OD pair, aggregated over the entire day
        and split by the is_weekend flag.

        Steps:
         - Filter map_matching data for origin and destination links.
         - Keep only agent_ids that have records in both origins and destinations.
         - Merge origin and destination records using agent_id and ensure the origin appears earlier based on sequence.
         - Extract the date and is_weekend flag from the origin timestamp.
         - For each day, count unique agent_ids per OD pair and then average these daily counts over days.
         - Map the link_ids to node_ids.
         - Compute the origin proportion (avg_volume divided by total volume from the origin zone).
        """
        # Filter map_matching data for origin and destination link_ids
        matched_origins = map_matching_df[map_matching_df["link_id"].isin(origin_links["link_id"])]
        matched_destinations = map_matching_df[map_matching_df["link_id"].isin(destination_links["link_id"])]

        # Retain only agent_ids that appear in both origins and destinations
        valid_agents = set(matched_origins["agent_id"]) & set(matched_destinations["agent_id"])
        matched_origins = matched_origins[matched_origins["agent_id"].isin(valid_agents)]
        matched_destinations = matched_destinations[matched_destinations["agent_id"].isin(valid_agents)]

        # Merge origin and destination records by agent_id with suffixes for origin and destination
        od_pairs = pd.merge(matched_origins, matched_destinations, on="agent_id", suffixes=("_origin", "_dest"))

        # Ensure that the origin appears before the destination based on the sequence field
        od_pairs = od_pairs[od_pairs["seq_origin"] < od_pairs["seq_dest"]]

        # Assuming map_matching_df has a timestamp field similar to CrossingStartDateLocal for origin
        # Extract date and is_weekend flag from the origin timestamp
        od_pairs["date"] = pd.to_datetime(od_pairs["time_origin"]).dt.date
        od_pairs["is_weekend"] = pd.to_datetime(od_pairs["time_origin"]).dt.weekday >= 5

        # Group by is_weekend, date, and OD pair (link_id_origin, link_id_dest) and count unique agent_ids per day
        daily_volumes = (
            od_pairs.groupby(["is_weekend", "date", "link_id_origin", "link_id_dest"])["agent_id"]
            .nunique()
            .reset_index(name="daily_volume")
        )

        # Average daily volume over days for each is_weekend flag and OD pair, then round to an integer
        od_matrix_avg = (
            daily_volumes.groupby(["is_weekend", "link_id_origin", "link_id_dest"])["daily_volume"]
            .mean()
            .reset_index()
            .rename(columns={"daily_volume": "avg_volume"})
        )
        od_matrix_avg["avg_volume"] = od_matrix_avg["avg_volume"].round(0).astype(int)

        # Map link_ids to node_ids using link_df
        link_node_map = link_df.set_index("link_id")[["from_node_id", "to_node_id"]]
        od_matrix_avg["o_zone_id"] = od_matrix_avg["link_id_origin"].map(link_node_map["from_node_id"])
        od_matrix_avg["d_zone_id"] = od_matrix_avg["link_id_dest"].map(link_node_map["to_node_id"])

        # Calculate total average volume per origin zone for each is_weekend flag for origin proportion calculation
        origin_totals = (
            od_matrix_avg.groupby(["is_weekend", "o_zone_id"])["avg_volume"]
            .sum()
            .reset_index(name="total_volume")
        )
        od_matrix_avg = pd.merge(od_matrix_avg, origin_totals, on=["is_weekend", "o_zone_id"])
        od_matrix_avg["o_proportion"] = (od_matrix_avg["avg_volume"] / od_matrix_avg["total_volume"]).round(3)

        return od_matrix_avg

    def plot_od_matrix(self, od_matrix_avg):
        """Visualize the OD matrix as a heatmap."""
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

            output_dir = os.path.join(self.output_folder, "trend_analysis_cbi", "od_matrix_wp")
            os.makedirs(output_dir, exist_ok=True)
            filename = f"od_matrix_heatmap_wp_{'weekend' if is_weekend else 'weekday'}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()

    def save_to_csv(self, od_matrix):
        """Save the OD matrix to a CSV file."""
        od_matrix = od_matrix.drop(columns=["link_id_origin", "link_id_dest"])
        od_matrix["is_weekend"] = od_matrix["is_weekend"].astype(int)
        od_matrix = od_matrix[["o_zone_id", "d_zone_id", "is_weekend", "avg_volume", "total_volume", "o_proportion"]]

        # Define output directory
        output_dir = os.path.join(self.output_folder, "trend_analysis_cbi", "od_matrix_wp")
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, "od_volume_wp.csv")
        od_matrix.to_csv(filepath, encoding="utf-8-sig", index=False)
        # print(f"OD Matrix saved to: {self.output_file}")

    def run(self):
        """Run the OD matrix analysis pipeline."""
        # Load data from the database
        if self.stop_event and self.stop_event.is_set():
            return
        node_df, link_df, map_matching_df = self.load_data()

        # Process OD links
        if self.stop_event and self.stop_event.is_set():
            return
        origin_links, destination_links, link_df = self.process_od_links(node_df, link_df)

        # Compute OD matrix
        if self.stop_event and self.stop_event.is_set():
            return
        od_matrix_avg = self.compute_od_matrix(map_matching_df, origin_links, destination_links, link_df)

        # Save OD matrix to CSV
        if self.stop_event and self.stop_event.is_set():
            return
        self.save_to_csv(od_matrix_avg)

        # # Print OD matrix result
        # print(od_matrix_avg)

        # Visualize OD matrix
        if self.stop_event and self.stop_event.is_set():
            return
        self.plot_od_matrix(od_matrix_avg)

        print("OD Matrix Heatmap saved.")


class ODTravelTimeAnalyzerWP:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initialize database path and output folder for saving results."""
        self.database_path = database_path
        self.output_folder = output_folder
        self.output_file = os.path.join(output_folder, "od_travel_time.csv")
        self.stop_event = stop_event

    def load_data(self):
        """Load node, link, and map_matching data from the SQLite database."""
        conn = sqlite3.connect(self.database_path)

        # Read node data
        node_df = pd.read_sql("SELECT node_id, is_boundary FROM node", conn)

        # Read link data
        link_df = pd.read_sql("SELECT link_id, from_node_id, to_node_id FROM link", conn)

        # Read map_matching data
        map_matching_df = pd.read_sql("SELECT agent_id, link_id, seq, time FROM map_matching", conn)

        conn.close()
        return node_df, link_df, map_matching_df

    def process_od_links(self, node_df, link_df):
        """
        Identify origin and destination links based on node is_boundary values.

        - is_boundary = 1 or 2 → Origin (match with from_node_id)
        - is_boundary = -1 or 2 → Destination (match with to_node_id)
        """
        origins = node_df[node_df["is_boundary"].isin([1, 2])]
        destinations = node_df[node_df["is_boundary"].isin([-1, 2])]

        # Get origin-related link_ids where from_node_id matches the node_id
        origin_links = link_df[link_df["from_node_id"].isin(origins["node_id"])]
        origin_links = origin_links[["from_node_id", "link_id"]].rename(columns={"from_node_id": "node_id"})

        # Get destination-related link_ids where to_node_id matches the node_id
        destination_links = link_df[link_df["to_node_id"].isin(destinations["node_id"])]
        destination_links = destination_links[["to_node_id", "link_id"]].rename(columns={"to_node_id": "node_id"})

        return origin_links, destination_links, link_df

    def compute_od_travel_time(self, map_matching_df, origin_links, destination_links, link_df):
        """
        Compute OD travel time for each `agent_id`.

        - Ensure each `agent_id` contains at least one origin and one destination link.
        - Compute travel time as `time_dest - time_origin` for each `agent_id`.
        - Replace `link_id_origin` and `link_id_dest` with corresponding `node_id` (o_zone_id, d_zone_id).
        """
        # Convert `time` column to datetime if it's stored as a string
        map_matching_df["time"] = pd.to_datetime(map_matching_df["time"])

        # Filter map_matching data for origin and destination link_ids
        matched_origins = map_matching_df[map_matching_df["link_id"].isin(origin_links["link_id"])]
        matched_destinations = map_matching_df[map_matching_df["link_id"].isin(destination_links["link_id"])]

        # Keep only `agent_id`s that exist in both matched origins and destinations
        valid_agents = set(matched_origins["agent_id"]) & set(matched_destinations["agent_id"])
        matched_origins = matched_origins[matched_origins["agent_id"].isin(valid_agents)]
        matched_destinations = matched_destinations[matched_destinations["agent_id"].isin(valid_agents)]

        # Merge origin and destination pairs by `agent_id` to form valid OD pairs
        od_pairs = pd.merge(matched_origins, matched_destinations, on="agent_id", suffixes=("_origin", "_dest"))

        # Ensure that the origin appears earlier in time (sequence condition)
        od_pairs = od_pairs[od_pairs["seq_origin"] < od_pairs["seq_dest"]]

        # Convert `time_origin` and `time_dest` to datetime if not already
        od_pairs["time_origin"] = pd.to_datetime(od_pairs["time_origin"])
        od_pairs["time_dest"] = pd.to_datetime(od_pairs["time_dest"])

        # Compute travel time in minutes
        od_pairs["travel_time"] = (od_pairs["time_dest"] - od_pairs["time_origin"]).dt.total_seconds() / 60

        # Map `link_id_origin` and `link_id_dest` to corresponding `node_id`
        link_node_map = link_df.set_index("link_id")[["from_node_id", "to_node_id"]]

        # Get `o_zone_id` (origin node) from `from_node_id`
        od_pairs["o_zone_id"] = od_pairs["link_id_origin"].map(link_node_map["from_node_id"])

        # Get `d_zone_id` (destination node) from `to_node_id`
        od_pairs["d_zone_id"] = od_pairs["link_id_dest"].map(link_node_map["to_node_id"])

        # Add start time column for plotting
        od_pairs["start_time"] = od_pairs["time_origin"]

        # Select final output columns
        od_travel_times = od_pairs[["agent_id", "o_zone_id", "d_zone_id", "travel_time", "start_time"]]

        return od_travel_times

    def plot_od_travel_times(self, od_travel_times):
        """
        Plot OD travel times as separate bar charts for each OD pair.

        - Each OD pair gets its own subplot.
        - Travel times greater than predefined thresholds are filtered out.
        - 3 Sigma filtering is applied within a rolling window.
        - Smoothed trend lines are generated using Exponential Moving Average (EMA) and Savitzky-Golay filtering.
        """
        print("Plotting Travel Time Profile...")

        od_pairs = [(36, 28), (27, 37), (35, 25), (25, 35)]
        # od_pairs = [(36, 28)]

        # Filter for selected OD pairs
        filtered_od_travel_times = od_travel_times[
            od_travel_times.apply(lambda row: (row["o_zone_id"], row["d_zone_id"]) in od_pairs, axis=1)
        ].copy()

        # # Apply travel time filtering based on predefined thresholds
        # filtered_od_travel_times = filtered_od_travel_times[
        #     ~(
        #             filtered_od_travel_times.apply(
        #                 lambda row: (row["o_zone_id"], row["d_zone_id"]) in [(110, 69), (68, 113)] and row[
        #                     "travel_time"] > 30,
        #                 axis=1
        #             ) |
        #             filtered_od_travel_times.apply(
        #                 lambda row: (row["o_zone_id"], row["d_zone_id"]) in [(102, 59), (59, 102)] and row[
        #                     "travel_time"] > 10,
        #                 axis=1
        #             )
        #     )
        # ]

        colors = ['#4472C5', '#E90E01', '#1C841D', '#9A6600']  # Colors for different OD pairs

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
                continue  # Skip this OD pair if no data is available

            # Sort data for time-series plotting
            subset.loc[:, "dummy_time"] = subset["start_time"].apply(lambda x: x.replace(year=1900, month=1, day=1).to_pydatetime())
            # subset = subset.sort_values("start_time")
            subset = subset.sort_values("dummy_time")


            # Extract start times and travel times for plotting
            # start_times = subset["start_time"]
            # travel_times = subset["travel_time"]
            start_times = subset["dummy_time"]
            travel_times = subset["travel_time"]

            kf = KalmanFilter(
                initial_state_mean=travel_times.iloc[0],
                transition_matrices=[1],  # Constant velocity model
                observation_matrices=[1],  # Direct observation
                transition_covariance=np.array([[0.5]]),  # Process noise (Higher = Smoother)
                observation_covariance=np.array([[0.01]])  # Measurement noise (Lower = Trust estimates more)
            )

            # Run the Kalman filter
            state_means, state_covariances = kf.filter(travel_times.ffill().bfill())

            # Convert Kalman state estimates to 1D array
            smoothed_travel_times = state_means.flatten()

            state_std = np.sqrt(state_covariances.flatten())  # Convert variance to standard deviation

            upper_bound_smooth = smoothed_travel_times + 3 * state_std
            lower_bound_smooth = smoothed_travel_times - 3 * state_std

            out_of_bounds = (travel_times < lower_bound_smooth) | (travel_times > upper_bound_smooth)
            outlier_ratio = np.sum(out_of_bounds) / len(travel_times) * 100
            print(
                f"OD pair ({o_zone}, {d_zone}): {outlier_ratio:.2f}% of data are outside the confidence interval.")

            # sigma = 60  # Higher value makes trend smoother
            # smoothed_travel_times = gaussian_filter1d(smoothed_travel_times, sigma=sigma)
            # upper_bound_smooth = gaussian_filter1d(upper_bound_smooth, sigma=sigma)
            # lower_bound_smooth = gaussian_filter1d(lower_bound_smooth, sigma=sigma)
            #
            # poly_order = 4  # Polynomial degree for smoothness
            # window_size = 51  # Must be an odd number
            #
            # smoothed_travel_times = savgol_filter(smoothed_travel_times, window_size, poly_order)
            # upper_bound_smooth = savgol_filter(upper_bound_smooth, window_size, poly_order)
            # lower_bound_smooth = savgol_filter(lower_bound_smooth, window_size, poly_order)

            plt.figure(figsize=(12, 8))
            bar_width = 0.0005

            # Plot original travel times
            plt.bar(start_times, travel_times, color=color, width=bar_width, label=od_labels[(o_zone, d_zone)])

            # # Plot smoothed travel time (trend)
            # plt.plot(start_times, smoothed_travel_times, color='black', linewidth=2.5,
            #          label="Travel Time Trend")
            #
            # # Plot ±3 sigma range as shaded area
            # plt.fill_between(start_times, lower_bound_smooth, upper_bound_smooth, color='#627B7B',
            #                  label="Confidence Interval")

            # Formatting
            plt.xlabel('Time', fontsize=18)
            plt.ylabel('Travel Time (min)', fontsize=18)
            plt.title(f'Corridor End-to-End Travel Time Waypoint: {od_labels[(o_zone, d_zone)]}', fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(fontsize=14)
            plt.grid(axis='y', linestyle='--')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            # Set x-axis time range
            # start_time = filtered_od_travel_times["start_time"].min()  # Earliest timestamp
            # end_time = filtered_od_travel_times["start_time"].max()  # Latest timestamp
            # start_time = subset["dummy_time"].min()  # Earliest timestamp
            # end_time = subset["dummy_time"].max()  # Latest timestamp
            start_time = datetime(1900, 1, 1, 0, 0, 12)
            end_time = datetime(1900, 1, 1, 23, 59, 53)
            plt.xlim(start_time, end_time)
            plt.ylim(0, 30)

            plt.tight_layout()
            # plt.show()

            output_dir = os.path.join(self.output_folder, "trend_analysis_cbi", "od_travel_time_wp")
            os.makedirs(output_dir, exist_ok=True)

            # Save figure
            filename = f"corridor_travel_time_wp_{od_labels[(o_zone, d_zone)].lower().replace(' ', '_')}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()  # Close figure to free memory

    def run(self):
        """Runs the OD travel time analysis and visualization."""
        # Load data from the database
        if self.stop_event and self.stop_event.is_set():
            return
        node_df, link_df, map_matching_df = self.load_data()

        # Process OD links
        if self.stop_event and self.stop_event.is_set():
            return
        origin_links, destination_links, link_df = self.process_od_links(node_df, link_df)

        if self.stop_event and self.stop_event.is_set():
            return
        od_travel_times = self.compute_od_travel_time(map_matching_df, origin_links, destination_links, link_df)

        if self.stop_event and self.stop_event.is_set():
            return
        self.plot_od_travel_times(od_travel_times)

        # print(od_travel_times)

        print("Travel Time Profile saved")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")
    DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "output", "database", "unified_database.db")

    od_analyzer_wp = ODMatrixAnalyzerWP(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    od_analyzer_wp.run()

    odtt_analyzer_wp = ODTravelTimeAnalyzerWP(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    odtt_analyzer_wp.run()