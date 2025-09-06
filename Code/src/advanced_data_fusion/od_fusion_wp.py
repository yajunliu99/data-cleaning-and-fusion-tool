import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ODFusionAggregatorWP:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initialize the database path."""
        self.database_path = database_path
        self.output_folder = output_folder

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event

        self.conn = sqlite3.connect(database_path)
        self.penetration_rate_df = None

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

    def compute_od_matrix(self, map_matching_df, origin_links, destination_links, link_df):
        """
        Compute OD matrix in 15-minute intervals, adjusting volumes using penetration rate.
        """
        map_matching_df["time"] = pd.to_datetime(map_matching_df["time"])
        map_matching_df["time_bin"] = map_matching_df["time"].dt.floor("15min")
        map_matching_df["time_of_day"] = map_matching_df["time_bin"].dt.time

        # Add is_weekend flag based on the time (using the original timestamp)
        map_matching_df["is_weekend"] = map_matching_df["time"].dt.weekday >= 5

        # Filter for relevant OD links
        matched_origins = map_matching_df[map_matching_df["link_id"].isin(origin_links["link_id"])]
        matched_destinations = map_matching_df[map_matching_df["link_id"].isin(destination_links["link_id"])]

        # Keep only agent IDs present in both origins and destinations
        valid_agents = set(matched_origins["agent_id"]) & set(matched_destinations["agent_id"])
        matched_origins = matched_origins[matched_origins["agent_id"].isin(valid_agents)]
        matched_destinations = matched_destinations[matched_destinations["agent_id"].isin(valid_agents)]

        # Merge OD pairs by agent_id and ensure seq_origin < seq_dest
        od_pairs = pd.merge(matched_origins, matched_destinations, on=["agent_id", "time_bin"], suffixes=("_origin", "_dest"))
        od_pairs = od_pairs[od_pairs["seq_origin"] < od_pairs["seq_dest"]]

        # Use the time_bin from the merge to (re)compute is_weekend (assumed same for origin)
        od_pairs["is_weekend"] = od_pairs["time_bin"].dt.weekday >= 5

        # Compute OD matrix by 15-minute intervals
        od_matrix = od_pairs.groupby(["time_bin", "link_id_origin", "link_id_dest", "is_weekend"])["agent_id"].nunique().reset_index()
        od_matrix.rename(columns={"agent_id": "volume"}, inplace=True)

        # Map link IDs to node IDs
        link_node_map = link_df.set_index("link_id")[["from_node_id", "to_node_id"]]
        od_matrix["o_zone_id"] = od_matrix["link_id_origin"].map(link_node_map["from_node_id"])
        od_matrix["d_zone_id"] = od_matrix["link_id_dest"].map(link_node_map["to_node_id"])

        # Merge with penetration rate
        self.penetration_rate_df["time_of_day"] = self.penetration_rate_df["time_of_day"].astype(str)
        od_matrix["time_of_day"] = od_matrix["time_bin"].dt.time.astype(str)
        od_matrix = od_matrix.merge(self.penetration_rate_df[["time_of_day", "is_weekend", "map_penetration_rate"]], on=["time_of_day", "is_weekend"], how="left")

        # Adjust OD volume by penetration rate
        od_matrix["adjusted_volume"] = od_matrix["volume"] / od_matrix["map_penetration_rate"]

        # Aggregate adjusted volume over all time bins
        od_matrix_final = od_matrix.groupby(["is_weekend","o_zone_id", "d_zone_id"])["adjusted_volume"].sum().reset_index()

        # Set the index as (o_zone_id, d_zone_id)
        od_matrix_final.set_index(["o_zone_id", "d_zone_id"], inplace=True)

        return od_matrix_final

    def plot_od_matrix(self, od_matrix_avg):
        """Visualize the adjusted OD matrix as a heatmap."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping before plotting OD matrix")
            return

        od_matrix_avg = od_matrix_avg.reset_index()

        for is_weekend in od_matrix_avg["is_weekend"].unique():
            day_label = "Weekend" if is_weekend else "Weekday"
            subset = od_matrix_avg[od_matrix_avg["is_weekend"] == is_weekend]
            if subset.empty:
                continue

            # Create a pivot table with origins as rows and destinations as columns
            pivot_table = subset.pivot(index="o_zone_id", columns="d_zone_id", values="adjusted_volume").fillna(0)
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, cmap="YlOrRd", linewidths=0.5, annot=True, fmt='.0f', cbar=True)
            plt.xlabel("Destinations", fontsize=18)
            plt.ylabel("Origins", fontsize=18)
            plt.title(f"Fused OD Matrix Heatmap Waypoint {day_label}", fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            output_dir = os.path.join(self.output_folder, "advanced_data_fusion", "od_matrix_wp")
            os.makedirs(output_dir, exist_ok=True)
            filename = f"fused_od_matrix_heatmap_{'weekend' if is_weekend else 'weekday'}_wp.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()

    def save_to_csv(self, od_matrix):
        """Save the OD matrix to a CSV file."""
        output_dir = os.path.join(self.output_folder, "advanced_data_fusion", "od_matrix_wp")
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, "fused_od_volume_wp.csv")
        od_matrix.to_csv(filepath, encoding="utf-8-sig")

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
        print("Fusing OD Matrix...")
        od_matrix = self.compute_od_matrix(map_matching_df, origin_links, destination_links,
                                           link_df)  # Pass required arguments

        # Save and visualize the OD matrix
        if self.stop_event and self.stop_event.is_set():
            return
        self.save_to_csv(od_matrix)

        if self.stop_event and self.stop_event.is_set():
            return
        self.plot_od_matrix(od_matrix)

        print("OD Matrix Fusion Completed.")
        self.conn.close()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")
    DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "output", "database", "unified_database.db")

    od_aggregator_wp = ODFusionAggregatorWP(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    od_aggregator_wp.run()