import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class TimeDependentODAggregatorWP:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initialize the database path."""
        self.database_path = database_path
        self.output_folder = output_folder

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event

        self.conn = sqlite3.connect(database_path)
        self.penetration_rate_df = None  # Placeholder for penetration rate data

    def get_lane_readings(self):
        """Retrieve and aggregate lane volume data by 15-minute intervals, then compute time-of-day averages."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_lane_readings")
            return pd.DataFrame()

        query = """
        SELECT zone_id, volume, local_time FROM lane_readings
        """
        df_lane = pd.read_sql(query, self.conn)
        df_lane["local_time"] = pd.to_datetime(df_lane["local_time"])
        df_lane["time_bin"] = df_lane["local_time"].dt.floor("15min")
        df_lane["time_of_day"] = df_lane["time_bin"].dt.time
        df_lane["weekday"] = df_lane["local_time"].dt.weekday
        df_lane["is_weekend"] = df_lane["weekday"].apply(lambda x: 1 if x >= 5 else 0)

        # Aggregate total volume per 15-minute interval
        df_lane = df_lane.groupby(["time_bin", "is_weekend"])["volume"].sum().reset_index()

        # Compute daily time-of-day averages
        df_lane["time_of_day"] = df_lane["time_bin"].dt.time
        df_avg_lane = df_lane.groupby(["time_of_day", "is_weekend"])["volume"].mean().reset_index()

        return df_avg_lane

    def get_map_matching_volumes(self):
        """Retrieve and aggregate unique agent counts by 15-minute intervals, then compute time-of-day averages."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_map_matching_volumes")
            return pd.DataFrame()

        query = """
        SELECT agent_id, time FROM map_matching
        """
        df_map = pd.read_sql(query, self.conn)
        df_map["time"] = pd.to_datetime(df_map["time"])
        df_map["time_bin"] = df_map["time"].dt.floor("15min")
        df_map["time_of_day"] = df_map["time_bin"].dt.time
        df_map["weekday"] = df_map["time"].dt.weekday
        df_map["is_weekend"] = df_map["weekday"].apply(lambda x: 1 if x >= 5 else 0)

        # Aggregate unique agent_id counts per 15-minute interval
        df_map = df_map.groupby(["time_bin", "is_weekend"])["agent_id"].nunique().reset_index()

        # Compute daily time-of-day averages
        df_map["time_of_day"] = df_map["time_bin"].dt.time
        df_avg_map = df_map.groupby(["time_of_day", "is_weekend"])["agent_id"].mean().reset_index()

        return df_avg_map

    def compute_penetration_rate(self):
        """Compute penetration rates and return the merged dataset."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping compute_penetration_rate")
            return

        df_lane = self.get_lane_readings()
        df_map = self.get_map_matching_volumes()

        df_lane = df_lane.rename(columns={"volume": "lane_volume"})
        df_map = df_map.rename(columns={"agent_id": "map_matching_volume"})

        # Merge datasets
        df_merged = df_lane.merge(df_map, on=["time_of_day", "is_weekend"], how="left")

        # Compute penetration rates
        df_merged["map_penetration_rate"] = df_merged["map_matching_volume"] / df_merged["lane_volume"]

        self.penetration_rate_df = df_merged  # Store penetration rates for OD matrix usage

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
        """Identify origin and destination links based on is_boundary values."""
        origins = node_df[node_df["is_boundary"].isin([1, 2])]
        destinations = node_df[node_df["is_boundary"].isin([-1, 2])]

        # Get origin-related links
        origin_links = link_df[link_df["from_node_id"].isin(origins["node_id"])]
        origin_links = origin_links[["from_node_id", "link_id"]].rename(columns={"from_node_id": "node_id"})

        # Get destination-related links
        destination_links = link_df[link_df["to_node_id"].isin(destinations["node_id"])]
        destination_links = destination_links[["to_node_id", "link_id"]].rename(columns={"to_node_id": "node_id"})

        return origin_links, destination_links, link_df

    def compute_time_dependent_od_matrices(self, map_matching_df, origin_links, destination_links, link_df):
        """
        Compute time-dependent OD matrices, adjusting volumes using penetration rate.
        """
        if self.stop_event and self.stop_event.is_set():
            print("Stopping compute_time_dependent_od_matrices")
            return {}

        # Convert time column to datetime format
        map_matching_df["time"] = pd.to_datetime(map_matching_df["time"])
        map_matching_df["time_bin"] = map_matching_df["time"].dt.floor("15min")  # 15-minute bins
        map_matching_df["time_of_day"] = map_matching_df["time_bin"].dt.time.astype(str)  # Extract time as string
        map_matching_df["time_dependent_bin"] = map_matching_df["time"].dt.floor("h")  # Hourly bins

        # Add is_weekend flag based on the time (using the original timestamp)
        map_matching_df["is_weekend"] = map_matching_df["time"].dt.weekday >= 5

        # Filter for relevant origin and destination links
        matched_origins = map_matching_df[map_matching_df["link_id"].isin(origin_links["link_id"])]
        matched_destinations = map_matching_df[map_matching_df["link_id"].isin(destination_links["link_id"])]

        # Keep only agent IDs that appear in both origin and destination data
        valid_agents = set(matched_origins["agent_id"]) & set(matched_destinations["agent_id"])
        matched_origins = matched_origins[matched_origins["agent_id"].isin(valid_agents)]
        matched_destinations = matched_destinations[matched_destinations["agent_id"].isin(valid_agents)]

        # Ensure time_bin is datetime format
        matched_origins["time_bin"] = pd.to_datetime(matched_origins["time_bin"])
        matched_destinations["time_bin"] = pd.to_datetime(matched_destinations["time_bin"])

        # Merge OD pairs by agent_id and time_bin
        od_pairs = pd.merge(matched_origins, matched_destinations, on=["agent_id", "time_bin"],
                            suffixes=("_origin", "_dest"))

        # Ensure sequence order (origin before destination)
        od_pairs = od_pairs[od_pairs["seq_origin"] < od_pairs["seq_dest"]]

        # Use the time_bin from the merge to (re)compute is_weekend (assumed same for origin)
        od_pairs["is_weekend"] = od_pairs["time_bin"].dt.weekday >= 5

        # Ensure time_dependent_bin exists in od_pairs
        od_pairs["time_dependent_bin"] = od_pairs["time_bin"].dt.floor("h")

        # Ensure time_bin is included in the groupby
        od_matrix = od_pairs.groupby(["time_dependent_bin", "time_bin", "link_id_origin", "link_id_dest", "is_weekend"])[
            "agent_id"].nunique().reset_index()
        od_matrix.rename(columns={"agent_id": "volume"}, inplace=True)

        # Map link IDs to node IDs
        link_node_map = link_df.set_index("link_id")[["from_node_id", "to_node_id"]]
        od_matrix["o_zone_id"] = od_matrix["link_id_origin"].map(link_node_map["from_node_id"])
        od_matrix["d_zone_id"] = od_matrix["link_id_dest"].map(link_node_map["to_node_id"])

        # Convert time_of_day to string for merging
        self.penetration_rate_df["time_of_day"] = self.penetration_rate_df["time_of_day"].astype(str)

        # Ensure od_matrix has time_of_day before merging
        od_matrix["time_of_day"] = od_matrix["time_bin"].dt.time.astype(str)

        # Merge with penetration rate using the corrected time format
        od_matrix = od_matrix.merge(self.penetration_rate_df[["time_of_day", "is_weekend", "map_penetration_rate"]],
                                    on=["time_of_day", "is_weekend"], how="left")

        # Adjust OD volume by penetration rate
        od_matrix["adjusted_volume"] = od_matrix["volume"] / od_matrix["map_penetration_rate"]

        all_origins = sorted(origin_links["node_id"].unique())
        all_destinations = sorted(destination_links["node_id"].unique())
        full_index = pd.MultiIndex.from_product([all_origins, all_destinations], names=["o_zone_id", "d_zone_id"])

        # Aggregate OD matrix for each time-dependent bin
        time_dependent_od_matrices = {}
        for is_weekend in od_matrix["is_weekend"].unique():
            od_subset = od_matrix[od_matrix["is_weekend"] == is_weekend]

            for time_bin, group in od_subset.groupby("time_dependent_bin"):
                od_matrix_time_dependent = group.groupby(["o_zone_id", "d_zone_id"])["adjusted_volume"].sum().reset_index()
                od_matrix_time_dependent.set_index(["o_zone_id", "d_zone_id"], inplace=True)
                od_matrix_time_dependent = od_matrix_time_dependent.reindex(full_index, fill_value=0)
                od_matrix_time_dependent = od_matrix_time_dependent.sort_index(level="o_zone_id")
                time_dependent_od_matrices[(time_bin, is_weekend)] = od_matrix_time_dependent

        return time_dependent_od_matrices

    def plot_time_dependent_od_matrices(self, time_dependent_od_matrices):
        """
        Generate and save heatmap visualizations for each time-dependent OD matrix.
        """
        if self.stop_event and self.stop_event.is_set():
            print("Stopping plot_time_dependent_od_matrices")
            return

        output_dir = os.path.join(self.output_folder, "odme_simulation", "od_matrix_wp")
        os.makedirs(output_dir, exist_ok=True)

        sample_matrix = next(iter(time_dependent_od_matrices.values()))
        all_origins = sorted(sample_matrix.index.get_level_values("o_zone_id").unique())
        all_destinations = sorted(sample_matrix.index.get_level_values("d_zone_id").unique())

        for (time_bin, is_weekend), od_matrix in time_dependent_od_matrices.items():
            day_label = "Weekend" if is_weekend else "Weekday"

            # Convert OD matrix to pivot table for heatmap
            pivot_table = od_matrix.reset_index().pivot(index="o_zone_id", columns="d_zone_id",
                                                        values="adjusted_volume")
            pivot_table = pivot_table.reindex(index=all_origins, columns=all_destinations, fill_value=0)

            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, cmap="YlOrRd", linewidths=0.5, annot=True, fmt='.0f', cbar=True)
            plt.xlabel("Destinations", fontsize=18)
            plt.ylabel("Origins", fontsize=18)
            plt.title(f"Time-Dependent OD Matrix Heatmap Waypoint ({time_bin.strftime('%H:%M')}) {day_label}", fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            # Save the heatmap image
            plt.savefig(os.path.join(output_dir, f"OD_Matrix_Heatmap_{time_bin.strftime('%H')}_{day_label.lower()}_wp.png"), dpi=300)
            plt.close()

    def save_to_csv(self, od_matrix_dict):
        """Save the OD matrix to separate CSV files for each time-dependent bin."""
        output_dir = os.path.join(self.output_folder, "odme_simulation", "od_matrix_wp")
        os.makedirs(output_dir, exist_ok=True)

        for (time_bin, is_weekend), od_matrix in od_matrix_dict.items():
            day_label = "Weekend" if is_weekend else "Weekday"
            od_matrix_sorted = od_matrix.sort_index(level="o_zone_id")
            filename = f"fused_od_volume_{time_bin.strftime('%H')}_{day_label.lower()}_wp.csv"
            output_path = os.path.join(output_dir, filename)
            od_matrix_sorted.to_csv(output_path, encoding="utf-8-sig")

    def run(self):
        """Run the full pipeline: Compute penetration rates, then compute OD matrix."""
        if self.stop_event and self.stop_event.is_set():
            return
        self.compute_penetration_rate()

        # Load OD-related data
        if self.stop_event and self.stop_event.is_set():
            return
        node_df, link_df, map_matching_df = self.load_data()
        origin_links, destination_links, link_df = self.process_od_links(node_df, link_df)

        # Now run OD matrix computation
        if self.stop_event and self.stop_event.is_set():
            return
        print("Estimating Time Dependent OD Matrix...")
        time_dependent_od_matrix = self.compute_time_dependent_od_matrices(map_matching_df, origin_links, destination_links,
                                           link_df)

        # Save and visualize the OD matrix
        if self.stop_event and self.stop_event.is_set():
            return
        self.save_to_csv(time_dependent_od_matrix)

        if self.stop_event and self.stop_event.is_set():
            return
        self.plot_time_dependent_od_matrices(time_dependent_od_matrix)

        print("Time Dependent ODME Completed.")
        self.conn.close()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")
    DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "output", "database", "unified_database.db")

    time_dependent_od_aggregator_wp = TimeDependentODAggregatorWP(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    time_dependent_od_aggregator_wp.run()