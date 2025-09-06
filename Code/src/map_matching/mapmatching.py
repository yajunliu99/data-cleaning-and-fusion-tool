import os
import sqlite3
import platform
import sys
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Point, LineString
from shapely.wkt import loads
from gotrackit.map.Net import Net
from gotrackit.MapMatch import MapMatch
from src.basic_data_cleaning.unified_database import CSVToSQLiteProcessor


class MapMatchingProcessor:
    def __init__(self, input_folder, output_folder, stop_event=None, progress_callback=None, progress_output=None):
        """Initializes paths for input and output folders."""
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.database_path = os.path.join(output_folder, "database", "unified_database.db")

        # Paths for network files
        self.node_csv_path = os.path.join(input_folder, "network", "node.csv")
        self.link_csv_path = os.path.join(input_folder, "network", "link.csv")
        self.node_shp_path = os.path.join(input_folder, "network", "shp", "node.shp")
        self.link_shp_path = os.path.join(input_folder, "network", "shp", "link.shp")

        os.makedirs(os.path.dirname(self.node_shp_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.link_shp_path), exist_ok=True)

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event
        self.progress_callback = progress_callback
        # self.progress_output = progress_output
        self.progress_output = progress_output if progress_output else sys.stdout

    def process_network(self):
        """Converts node/link CSV files into Shapefiles for map matching."""
        print("Processing network data...")

        # Process Node Data
        nodes_df = pd.read_csv(self.node_csv_path)

        # Create GeoDataFrame for nodes
        nodes_gdf = gpd.GeoDataFrame(
            nodes_df,
            geometry=[Point(xy) for xy in zip(nodes_df["x_coord"], nodes_df["y_coord"])],
            crs="EPSG:4326"
        )

        # Save nodes to a Shapefile
        nodes_gdf.to_file(self.node_shp_path, driver="ESRI Shapefile")

        # Process Link Data
        links_df = pd.read_csv(self.link_csv_path)

        # Rename columns to match expected field names
        links_df = links_df.rename(columns={
            "from_node_id": "from_node",
            "to_node_id": "to_node",
            "directed": "dir"
        })

        # Create a dictionary mapping node_id to (longitude, latitude)
        node_dict = nodes_df.set_index("node_id")[["x_coord", "y_coord"]].to_dict("index")

        # Function to create a LineString from from_node and to_node
        def create_linestring(row):
            from_node = node_dict.get(row["from_node"])
            to_node = node_dict.get(row["to_node"])
            if from_node and to_node:
                return LineString([from_node, to_node])
            return None

        # Generate geometry if 'geometry' column is missing
        if "geometry" in links_df.columns:
            try:
                links_df["geometry"] = links_df["geometry"].apply(loads)
            except:
                links_df["geometry"] = links_df.apply(create_linestring, axis=1)
        else:
            links_df["geometry"] = links_df.apply(create_linestring, axis=1)

        # Create GeoDataFrame for links
        links_gdf = gpd.GeoDataFrame(
            links_df.dropna(subset=["geometry"]),
            geometry="geometry",
            crs="EPSG:4326"
        )

        # Save links to a Shapefile
        links_gdf.to_file(self.link_shp_path, driver="ESRI Shapefile")

        print("Network processing completed")

    def load_gps_data(self):
        """Loads GPS data from the SQLite database."""
        print("Loading Waypoint Data from SQLite database...")
        conn = sqlite3.connect(self.database_path)
        gps_df = pd.read_sql(
            "SELECT journey_id AS agent_id, latitude AS lat, longitude AS lng, local_time AS time FROM waypoint",
            conn
        )
        conn.close()
        print("Waypoint Data loaded successfully")
        return gps_df

    def update_progress(self, message, progress=None):
        """Updates the GUI progress bar if a callback is provided."""
        if self.progress_callback:
            self.progress_callback(message, progress)

    def perform_map_matching(self):
        """Runs the map matching process using the processed network and GPS data."""
        print("Performing Map Matching...")

        # Read network shapefiles
        link = gpd.read_file(self.link_shp_path)
        node = gpd.read_file(self.node_shp_path)

        # Initialize the network for map matching
        my_net = Net(link_gdf=link, node_gdf=node, not_conn_cost=2400)
        my_net.init_net()

        if self.stop_event and self.stop_event.is_set():
            print("Stopping before loading waypoint data...")
            return

        # Load GPS data
        gps_df = self.load_gps_data()

        # Count the total number of agents (for progress tracking)
        total_agents = gps_df["agent_id"].nunique()

        print(f"Total agents to process: {total_agents}")

        # Initialize MapMatching with parameters
        mpm = MapMatch(
            net=my_net, flag_name='agent',
            time_format='%Y-%m-%d %H:%M:%S',
            gps_buffer=12, top_k=10,
            dense_gps=False,
            use_heading_inf=True, omitted_l=6.0,
            del_dwell=True, dwell_l_length=5.0, dwell_n=2,
            export_html=False, export_geo_res=False, use_gps_source=False,
            gps_radius=15.0, export_all_agents=False,
            out_fldr=os.path.join(self.output_folder, "mapmatching")
        )

        match_results = []

        with tqdm(
                total=total_agents,
                desc="Processing agents",
                file=self.progress_output,
                dynamic_ncols=True,
                ascii=True,
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:
            for _, (agent_id, agent_data) in enumerate(gps_df.groupby("agent_id")):
                if self.stop_event and self.stop_event.is_set():
                    self.progress_output.write("Stopping Map Matching process...")
                    return

                match_res, _, _ = mpm.execute(gps_df=agent_data)

                match_results.append(match_res)

                pbar.update(1)

        self.progress_output.write("Map Matching Completed")

        # Save results
        final_match_res = pd.concat(match_results, ignore_index=True)

        output_mapmatching_dir = os.path.join(self.output_folder, "mapmatching")
        os.makedirs(output_mapmatching_dir, exist_ok=True)

        output_mapmatching_path = os.path.join(self.output_folder, "mapmatching", "mapmatching.csv")
        final_match_res.to_csv(output_mapmatching_path, encoding='utf_8_sig', index=False)

    def insert_map_matching_results(self):
        """Reads mapmatching.csv and inserts it into the SQLite database."""
        output_mapmatching_path = os.path.join(self.output_folder, "mapmatching", "mapmatching.csv")

        # Read the CSV file
        df = pd.read_csv(output_mapmatching_path)

        # Infer column types using CSVToSQLiteProcessor
        column_types = CSVToSQLiteProcessor._infer_column_types(self, df)

        # Connect to SQLite
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        table_name = "map_matching"
        columns_sql = ", ".join([f"{col} {col_type}" for col, col_type in column_types.items()])
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql})"
        cursor.execute(create_table_query)

        # Insert data
        df.to_sql(table_name, conn, if_exists="replace", index=False)

        # Commit and close connection
        conn.commit()
        conn.close()

    def run(self):
        """Runs the full pipeline for processing network data and map matching."""
        self.process_network()
        self.perform_map_matching()
        self.insert_map_matching_results()


class MapMatchingAnalyzer:
    def __init__(self, database_path):
        """Initializes database path."""
        self.database_path = database_path
        self.conn = sqlite3.connect(database_path)


    # def analyze_matching_coverage(self):
    #     """Analyzes the percentage of matched journeys and data coverage."""
    #     # Load data
    #     waypoint_df = pd.read_sql("SELECT journey_id FROM waypoint", self.conn)
    #     map_matching_df = pd.read_sql("SELECT agent_id FROM map_matching", self.conn)
    #
    #     # Compute percentages
    #     total_journeys = waypoint_df["journey_id"].nunique()
    #     matched_agents = map_matching_df["agent_id"].nunique()
    #
    #     total_waypoint_rows = len(waypoint_df)
    #     total_mapmatching_rows = len(map_matching_df)
    #
    #     journey_match_percentage = (matched_agents / total_journeys) * 100 if total_journeys > 0 else 0
    #     data_match_percentage = (total_mapmatching_rows / total_waypoint_rows) * 100 if total_waypoint_rows > 0 else 0
    #
    #     print(f"{matched_agents}/{total_journeys} ({journey_match_percentage:.2f}%) agents matched")
    #     print(f"{total_mapmatching_rows}/{total_waypoint_rows} ({data_match_percentage:.2f}%) waypoint matched")

    def analyze_matching_coverage(self):
        """Analyzes the percentage of matched journeys and data coverage."""
        # Load data
        waypoint_df = pd.read_sql("SELECT journey_id FROM waypoint", self.conn)
        map_matching_df = pd.read_sql("SELECT DISTINCT agent_id FROM map_matching", self.conn)

        # Find matched journey_ids from waypoint using agent_id
        matched_journeys = pd.read_sql(
            "SELECT DISTINCT journey_id FROM waypoint WHERE journey_id IN (SELECT agent_id FROM map_matching)",
            self.conn
        )

        # Total number of journey_ids in waypoint
        total_journeys = waypoint_df["journey_id"].nunique()

        # Number of matched journey_ids
        matched_journey_count = matched_journeys["journey_id"].nunique()

        # Compute total data points in waypoint for matched journeys
        matched_waypoint_count = pd.read_sql(
            f"SELECT COUNT(*) AS count FROM waypoint WHERE journey_id IN (SELECT agent_id FROM map_matching)",
            self.conn
        )["count"].iloc[0]

        # Total map_matching rows
        total_mapmatching_rows = pd.read_sql(
            "SELECT COUNT(*) AS count FROM map_matching",
            self.conn
        )["count"].iloc[0]

        # Compute percentages
        journey_match_percentage = (matched_journey_count / total_journeys) * 100 if total_journeys > 0 else 0
        data_match_percentage = (total_mapmatching_rows / matched_waypoint_count) * 100 if matched_waypoint_count > 0 else 0

        print(f"{matched_journey_count}/{total_journeys} ({journey_match_percentage:.2f}%) journeys matched")
        print(f"{total_mapmatching_rows}/{matched_waypoint_count} ({data_match_percentage:.2f}%) waypoint matched")

    def run(self):
        """Runs the full pipeline for processing network data and map matching."""
        self.analyze_matching_coverage()

        self.conn.close()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_INPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "input")
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")
    DEFAULT_DATABASE_PATH = os.path.join(DEFAULT_OUTPUT_FOLDER, "database", "unified_database.db")


    mapmatching_processor = MapMatchingProcessor(DEFAULT_INPUT_FOLDER, DEFAULT_OUTPUT_FOLDER)
    mapmatching_processor.run()

    # mapmatching_analyzer = MapMatchingAnalyzer(DEFAULT_DATABASE_PATH)
    # mapmatching_analyzer.run()