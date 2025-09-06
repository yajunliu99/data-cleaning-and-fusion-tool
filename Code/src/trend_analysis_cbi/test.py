import os
import sqlite3
import pandas as pd


class ODGroup:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initialize database connection."""
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"ERROR: Database not found at {database_path}")

        self.database_path = database_path
        self.output_folder = output_folder

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event

        self.conn = sqlite3.connect(database_path)
        self.valid_link_ids = set()

    def get_tmc_groups(self):
        """Retrieves TMC codes, road_order, and groups them into continuous road segments with the same direction."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_tmc_groups")
            return pd.DataFrame()

        query = """
        SELECT tmc, road, direction, road_order FROM TMC_Identification 
        WHERE road_order != 1 ORDER BY direction, road_order"""
        df_tmc = pd.read_sql(query, self.conn)

        if df_tmc.empty:
            raise ValueError("No valid TMC codes found with road_order != 1.")

        # Convert road_order to integer
        df_tmc["road_order"] = df_tmc["road_order"].astype(int)

        # Assign a group ID considering both road_order and direction
        df_tmc["group_id"] = df_tmc.groupby("direction")["road_order"].diff().ne(1).cumsum()

        return df_tmc

    def get_grouped_links(self, df_tmc):
        """Finds link_ids corresponding to each TMC group in the correct order."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_grouped_links")
            return pd.DataFrame()

        query = """
        SELECT tmc, link_ids FROM tmc_to_link"""
        df_tmc_to_link = pd.read_sql(query, self.conn)

        if df_tmc_to_link.empty:
            raise ValueError("No mapping data found in tmc_to_link table.")

        # Explode link_ids into separate rows
        df_tmc_to_link["link_ids"] = df_tmc_to_link["link_ids"].str.split(",")
        df_tmc_to_link = df_tmc_to_link.explode("link_ids").rename(columns={"link_ids": "link_id"})

        # Merge TMC group information with link data
        df_grouped_links = df_tmc.merge(df_tmc_to_link, on="tmc", how="left").dropna()

        # Ensure correct ordering based on road_order
        df_grouped_links = df_grouped_links.sort_values(["group_id", "road_order"]).drop_duplicates(
            ["group_id", "link_id"])

        # Group by group_id and aggregate links
        grouped_links = df_grouped_links.groupby("group_id")["link_id"].apply(list).reset_index()

        # Save to CSV
        output_dir = os.path.join(self.output_folder, "basic_data_fusion")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "grouped_links.csv")
        grouped_links.to_csv(output_file, index=False, encoding="utf-8-sig")

        print("Grouped Links Extraction Completed.")
        return grouped_links

    def get_group_od_pairs(self, grouped_links_df):
        """Computes OD pairs (from_node_id, to_node_id) for each group of ordered links."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_group_od_pairs")
            return pd.DataFrame()

        # Read link_id to from_node_id and to_node_id mapping
        query = "SELECT link_id, from_node_id, to_node_id FROM link"
        df_link = pd.read_sql(query, self.conn).astype({"link_id": str})

        # Ensure link_id in grouped_links_df is string for proper matching
        grouped_links_df["link_id"] = grouped_links_df["link_id"].apply(lambda lst: [str(l) for l in lst])

        od_pairs = []
        for _, row in grouped_links_df.iterrows():
            group_id = row["group_id"]
            link_list = row["link_id"]

            if not link_list:
                continue

            first_link = link_list[0]
            last_link = link_list[-1]

            # Lookup node IDs from the link table
            from_node = df_link.loc[df_link["link_id"] == first_link, "from_node_id"].values
            to_node = df_link.loc[df_link["link_id"] == last_link, "to_node_id"].values

            if from_node.size == 0 or to_node.size == 0:
                print(f"Missing node info for group {group_id}")
                continue

            od_pairs.append({
                "group_id": group_id,
                "from_node_id": from_node[0],
                "to_node_id": to_node[0]
            })

        df_od_pairs = pd.DataFrame(od_pairs)

        # Save to CSV
        # output_dir = os.path.join(self.output_folder, "basic_data_fusion")
        # os.makedirs(output_dir, exist_ok=True)
        # output_file = os.path.join(output_dir, "group_od_pairs.csv")
        # df_od_pairs.to_csv(output_file, index=False, encoding="utf-8-sig")

        print("OD Pair Extraction Completed.")
        return df_od_pairs

    def run(self):
        """Execute the data processing pipeline."""
        if self.stop_event and self.stop_event.is_set():
            return
        df_tmc_groups = self.get_tmc_groups()

        if self.stop_event and self.stop_event.is_set():
            return
        grouped_links = self.get_grouped_links(df_tmc_groups)

        if self.stop_event and self.stop_event.is_set():
            return
        df_od_pairs = self.get_group_od_pairs(grouped_links)

        print(df_od_pairs)

        self.conn.close()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")
    DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "output", "database", "unified_database.db")

    od_group = ODGroup(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    od_group.run()