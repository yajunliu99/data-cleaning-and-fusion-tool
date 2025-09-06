import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ProbeDataAnalyzer:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initialize the database path."""
        self.database_path = database_path
        self.output_folder = output_folder

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event

        self.conn = sqlite3.connect(database_path)
        self.penetration_rate_df = None  # Placeholder for penetration rate data

    def load_data(self):
        """Load node, link, and map_matching data from the SQLite database."""
        conn = sqlite3.connect(self.database_path)

        # Read node data
        node_df = pd.read_sql("SELECT node_id, is_boundary FROM node", conn)

        # Read link data
        link_df = pd.read_sql("SELECT link_id, from_node_id, to_node_id FROM link", conn)

        # Read zone_name_to_link data
        zone_name_to_link_df = pd.read_sql("SELECT Zone_Name, link_id FROM zone_name_to_link", conn)

        conn.close()
        return node_df, link_df, zone_name_to_link_df

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

        return origin_links, destination_links

    def extract_od_volume(self):
        """Extract OD volumes for All Days (0) and 8:00am - 9:00am (33-36)."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping extract_od_volume")
            return

        with sqlite3.connect(self.database_path) as conn:
            od_df = pd.read_sql(
                'SELECT Origin_Zone_Name, Destination_Zone_Name, Day_Type, Day_Part, "Average_Daily_O_D_Traffic_(StL_Volume)" FROM od',
                conn)

        od_df["Day_Type"] = od_df["Day_Type"].astype(str).str.extract(r"^(\d+)").astype(int)
        od_df["Day_Part"] = od_df["Day_Part"].astype(str).str.extract(r"^(\d+)").astype(int)

        # Filter OD data for days and the time range
        od_df = od_df[(od_df["Day_Type"] == 0) & (od_df["Day_Part"].isin([33, 34, 35, 36]))]

        node_df, link_df, zone_name_to_link_df = self.load_data()
        origin_links, destination_links = self.process_od_links(node_df, link_df)

        zone_name_to_link_df["Zone_Name"] = zone_name_to_link_df["Zone_Name"].str.strip().str.lower()
        od_df["Origin_Zone_Name"] = od_df["Origin_Zone_Name"].str.strip().str.lower()
        od_df["Destination_Zone_Name"] = od_df["Destination_Zone_Name"].str.strip().str.lower()

        origin_zone_df = zone_name_to_link_df.rename(columns={"Zone_Name": "Origin_Zone_Name", "link_id": "o_link_id"})
        destination_zone_df = zone_name_to_link_df.rename(
            columns={"Zone_Name": "Destination_Zone_Name", "link_id": "d_link_id"})

        od_df = od_df.merge(origin_zone_df, on="Origin_Zone_Name", how="left")
        od_df = od_df.merge(destination_zone_df, on="Destination_Zone_Name", how="left")

        # Map link_id to node_id using link_df (mapping to from_node_id and to_node_id)
        link_node_map = link_df.set_index("link_id")[["from_node_id", "to_node_id"]]

        od_df["o_zone_id"] = od_df["o_link_id"].map(link_node_map["from_node_id"])
        od_df["d_zone_id"] = od_df["d_link_id"].map(link_node_map["to_node_id"])

        od_df = od_df[od_df["o_zone_id"].isin(origin_links["node_id"])]
        od_df = od_df[od_df["d_zone_id"].isin(destination_links["node_id"])]

        od_df = od_df.dropna(subset=["o_zone_id", "d_zone_id"])

        all_origins = sorted(origin_links["node_id"].unique())
        all_destinations = sorted(destination_links["node_id"].unique())
        full_index = pd.MultiIndex.from_product([all_origins, all_destinations], names=["o_zone_id", "d_zone_id"])

        od_matrix = od_df.groupby(["o_zone_id", "d_zone_id"])[
            "Average_Daily_O_D_Traffic_(StL_Volume)"].sum().reset_index()
        od_matrix.rename(columns={"Average_Daily_O_D_Traffic_(StL_Volume)": "volume"}, inplace=True)

        od_matrix.set_index(["o_zone_id", "d_zone_id"], inplace=True)
        od_matrix = od_matrix.reindex(full_index, fill_value=0)
        od_matrix = od_matrix.reset_index()

        # Save to CSV
        output_dir = os.path.join(self.output_folder, "odme_simulation", "probe_od_volume")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "probe_od_volume_8_9am.csv")
        od_matrix.to_csv(output_path, encoding="utf-8-sig", index=False)
        print(f"OD volume data saved to: {output_path}")

    def extract_link_volume(self):
        """
        Extract traffic volume for each Zone_Name from the 'volume' table,
        """
        if self.stop_event and self.stop_event.is_set():
            print("Stopping extract_link_volume")
            return

        # Connect to the database and retrieve volume data
        with sqlite3.connect(self.database_path) as conn:
            volume_df = pd.read_sql(
                'SELECT Zone_Name, Day_Type, Day_Part, "Average_Daily_Segment_Traffic_(StL_Volume)" FROM volume',
                conn)

        volume_df["Day_Type"] = volume_df["Day_Type"].astype(str).str.extract(r"^(\d+)").astype(int)
        volume_df["Day_Part"] = volume_df["Day_Part"].astype(str).str.extract(r"^(\d+)").astype(int)

        volume_df = volume_df[(volume_df["Day_Type"] == 0) & (volume_df["Day_Part"].isin([9]))]

        _, _, zone_name_to_link_df = self.load_data()

        zone_name_to_link_df["Zone_Name"] = zone_name_to_link_df["Zone_Name"].str.strip().str.lower()
        volume_df["Zone_Name"] = volume_df["Zone_Name"].str.strip().str.lower()

        volume_df = volume_df.merge(zone_name_to_link_df, on="Zone_Name", how="left")

        all_links = zone_name_to_link_df["link_id"].unique()

        zone_volume_matrix = volume_df.groupby("link_id")["Average_Daily_Segment_Traffic_(StL_Volume)"].sum().reset_index()
        # zone_volume_matrix = volume_df.groupby(["Zone_Name", "link_id"])["Average_Daily_Segment_Traffic_(StL_Volume)"].sum().reset_index()
        zone_volume_matrix.rename(columns={"Average_Daily_Segment_Traffic_(StL_Volume)": "volume"}, inplace=True)

        full_index = pd.DataFrame({"link_id": all_links})
        zone_volume_matrix = full_index.merge(zone_volume_matrix, on="link_id", how="left")
        # zone_volume_matrix = full_index.merge(zone_volume_matrix, on="link_id", how="left").fillna(0)

        output_dir = os.path.join(self.output_folder, "odme_simulation", "probe_link_volume")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "probe_link_volume_8_9am.csv")
        zone_volume_matrix.to_csv(output_path, encoding="utf-8-sig", index=False)

        print(f"Zone volume data saved to: {output_path}")

    def run(self):
        """Run the full pipeline: Compute penetration rates, then compute OD matrix."""
        if self.stop_event and self.stop_event.is_set():
            return
        self.extract_od_volume()

        if self.stop_event and self.stop_event.is_set():
            return
        self.extract_link_volume()

        print("Probe OD Completed.")
        self.conn.close()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")
    DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "output", "database", "unified_database.db")

    probe_data_analyzer = ProbeDataAnalyzer(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    probe_data_analyzer.run()