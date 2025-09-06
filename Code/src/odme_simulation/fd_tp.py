import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score

class fundamental_diagram_model():

    def __init__(self, observed_flow, observed_density, observed_speed):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed

    def S3(self, beta):
        vf, kc, foc = beta
        estimated_speed = vf / np.power(1 + np.power((self.observed_density / kc), foc), 2 / foc)
        f_obj = np.mean(np.power(estimated_speed - self.observed_speed, 2))
        return f_obj


class first_order_derivative():

    def __init__(self, observed_flow, observed_density, observed_speed):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed

    def S3(self, beta):
        vf, kc, foc = beta
        intermediate_variable = np.power(self.observed_density / kc, foc)
        first_order_derivative_1 = 2 * np.mean(
            (vf / np.power(1 + intermediate_variable, 2 / foc) - self.observed_speed) / np.power(
                1 + intermediate_variable, 2 / foc))
        first_order_derivative_2 = 2 * np.mean((vf / np.power(1 + intermediate_variable,
                                                              2 / foc) - self.observed_speed) * 2 * vf * intermediate_variable / kc / np.power(
            1 + intermediate_variable, (foc + 2) / foc))
        first_order_derivative_3 = 2 * np.mean(
            (vf / np.power(1 + intermediate_variable, 2 / foc) - self.observed_speed) * 2 * vf * (
                        (1 + intermediate_variable) * np.log(
                    1 + intermediate_variable) - foc * intermediate_variable * np.log(
                    intermediate_variable)) / np.power(foc, 2) / np.power(1 + intermediate_variable, (foc + 2) / foc))
        first_order_derivative = np.asarray(
            [first_order_derivative_1, first_order_derivative_2, first_order_derivative_3])
        return first_order_derivative


class estimated_value():

    def __init__(self, observed_flow, observed_density, observed_speed):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed

    def S3(self, beta):
        vf, kc, foc = beta
        estimated_speed = vf / np.power(1 + np.power((self.observed_density / kc), foc), 2 / foc)
        estimated_flow = self.observed_density * estimated_speed
        return estimated_speed, estimated_flow


class theoretical_value():

    def __init__(self, density, speed):
        self.density = density
        self.speed = speed

    def S3(self, beta):
        vf, kc, foc = beta
        theoretical_speed = vf / np.power(1 + np.power((self.density / kc), foc), 2 / foc)
        theoretical_flow = self.density * theoretical_speed
        return theoretical_speed, theoretical_flow


class Adam_optimization():

    def __init__(self, objective, first_order_derivative, bounds, x0):
        self.n_iter = 2000
        self.alpha = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.objective = objective
        self.first_order_derivative = first_order_derivative
        self.bounds = bounds
        self.x0 = x0

    def adam(self):
        # keep track of solutions and scores
        solutions = list()
        scores = list()
        # generate an initial point
        x = list(self.x0)
        score = self.objective(x)
        # initialize first and second moments
        m = [0.0 for _ in range(self.bounds.shape[0])]
        v = [0.0 for _ in range(self.bounds.shape[0])]
        # run the gradient descent updates
        for t in range(1, self.n_iter):
            # calculate gradient g(t)
            g = self.first_order_derivative(x)
            # build a solution one variable at a time
            for i in range(self.bounds.shape[0]):
                # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g[i]
                # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g[i] ** 2
                # mhat(t) = m(t) / (1 - beta1(t))
                mhat = m[i] / (1.0 - self.beta1 ** (t + 1))
                # vhat(t) = v(t) / (1 - beta2(t))
                vhat = v[i] / (1.0 - self.beta2 ** (t + 1))
                # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
                x[i] = x[i] - self.alpha * mhat / (sqrt(vhat) + self.eps)
            # evaluate candidate point
            score = self.objective(x)
            # keep track of solutions and scores
            solutions.append(x.copy())
            scores.append(score)
            # report progress
        # print('Solution: %s, \nOptimal function value: %.5f' %(solutions[np.argmin(scores)], min(scores)))
        return solutions, scores

    def plot_iteration_process_adam(self, solutions):
        # sample input range uniformly at 0.1 increments
        xaxis = np.arange(self.bounds[0, 0], self.bounds[0, 1], 0.1)
        yaxis = np.arange(self.bounds[1, 0], self.bounds[1, 1], 0.1)
        x, y = np.meshgrid(xaxis, yaxis)
        results = self.objective(x, y)
        solutions = np.asarray(solutions)
        fig, ax = plt.subplots(figsize=(10, 6))
        cs = ax.contourf(x, y, results, levels=50, cmap='jet')
        ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
        ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        plt.tick_params(labelsize=14)
        plt.plot(solutions[:, 0], solutions[:, 1], '.-', color='k')
        plt.colorbar(cs)
        plt.title('Iteration process')
        # fig.savefig('../Figures/Case 1/Iteration process.png', dpi=300, bbox_inches='tight')


class plot_calibration_results():

    def __init__(self, observed_flow, observed_density, observed_speed, calibrated_paras, output_folder):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed
        self.calibrated_paras_S3 = calibrated_paras["S3"]  # Calibrated from fundamental diagram model, vf, kc, foc
        self.k = np.linspace(0.000001, 140, 70)
        self.v = np.linspace(0.000001, 0, 70)
        self.theoretical_value = theoretical_value(self.k, self.v)
        self.theoretical_speed_S3, self.theoretical_flow_S3 = self.theoretical_value.S3(self.calibrated_paras_S3)
        self.output_folder = output_folder

    def plot_qk(self):
        fig = plt.figure(figsize=(12, 8))
        plt.scatter(self.observed_density, self.observed_flow, s=5, marker='o', color='#E90E01', label='Observation')
        plt.plot(self.k, self.theoretical_flow_S3, color='#4472C5', linestyle='-', linewidth=4, label="S3")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Density (veh/km)', fontsize=18)
        plt.ylabel('Flow (veh/h)', fontsize=18)
        plt.xlim((0, 150))
        plt.ylim((0, 2000))
        plt.legend(loc='upper right', fontsize=14)
        plt.title('Flow vs. density Trip Path', fontsize=18)
        save_path = os.path.join(self.output_folder, "flow_vs_density_tp.png")
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)

    def plot_vk(self):
        fig = plt.figure(figsize=(12, 8))
        plt.scatter(self.observed_density, self.observed_speed, s=5, marker='o', color='#E90E01', label='Observation')
        plt.plot(self.k, self.theoretical_speed_S3, color='#4472C5', linestyle='-', linewidth=4, label="S3")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Density (veh/km)', fontsize=18)
        plt.ylabel('Speed (km/h)', fontsize=18)
        plt.xlim((5, 150))
        plt.ylim((0, 100))
        plt.legend(loc='upper right', fontsize=14)
        plt.title('Speed vs. density Trip Path', fontsize=18)
        save_path = os.path.join(self.output_folder, "speed_vs_density_tp.png")
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)

    def plot_vq(self):
        fig = plt.figure(figsize=(12, 8))
        plt.scatter(self.observed_flow, self.observed_speed, s=5, marker='o', color='#E90E01', label='Observation')
        plt.plot(self.theoretical_flow_S3, self.theoretical_speed_S3, color='#4472C5', linestyle='-', linewidth=4, label="S3")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Flow (veh/h)', fontsize=18)
        plt.ylabel('Speed (km/h)', fontsize=18)
        plt.xlim((0, 2000))
        plt.ylim((0, 100))
        plt.legend(loc='upper right', fontsize=14)
        plt.title('Speed vs. flow Trip Path', fontsize=18)
        save_path = os.path.join(self.output_folder, "speed_vs_flow_tp.png")
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)


class getMetrics():

    def __init__(self, observed_flow, observed_density, observed_speed):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed
        self.estimated_value = estimated_value(observed_flow, observed_density, observed_speed)

    def S3_RMSE_Overall(self, paras):
        estimated_speed, estimated_flow = self.estimated_value.S3(paras)
        rmse_speed = mean_squared_error(self.observed_speed, estimated_speed, squared=False)
        rmse_flow = mean_squared_error(self.observed_flow, estimated_flow, squared=False)
        r2_speed = r2_score(self.observed_speed, estimated_speed)
        r2_flow = r2_score(self.observed_flow, estimated_flow)
        return rmse_speed, rmse_flow, r2_speed, r2_flow

    def S3_RMSE_Small_Range(self, paras, interval=10):
        estimated_speed, estimated_flow = self.estimated_value.S3(paras)
        rmse_speed_small_range = []
        rmse_flow_small_range = []
        for i in range(0, 10):
            temp_index = np.where((self.observed_density >= 10 * i) & (self.observed_density < 10 * (i + 1)))
            observed_speed_i = self.observed_speed[temp_index]
            estimated_speed_i = estimated_speed[temp_index]
            observed_flow_i = self.observed_flow[temp_index]
            estimated_flow_i = estimated_flow[temp_index]
            rmse_speed_small_range.append(mean_squared_error(observed_speed_i, estimated_speed_i, squared=False))
            rmse_flow_small_range.append(mean_squared_error(observed_flow_i, estimated_flow_i, squared=False))
        observed_speed_last = self.observed_speed[np.where((self.observed_density >= 100))]
        estimated_speed_last = estimated_speed[np.where((self.observed_density >= 100))]
        observed_flow_last = self.observed_flow[np.where((self.observed_density >= 100))]
        estimated_flow_last = estimated_flow[np.where((self.observed_density >= 100))]
        rmse_speed_small_range.append(mean_squared_error(observed_speed_last, estimated_speed_last, squared=False))
        rmse_flow_small_range.append(mean_squared_error(observed_flow_last, estimated_flow_last, squared=False))
        return rmse_speed_small_range, rmse_flow_small_range


class calibrate():

    def __init__(self, flow, density, speed):
        self.flow = flow
        self.density = density
        self.speed = speed
        self.init_model_dict()

    def init_model_dict(self):
        self.model = fundamental_diagram_model(self.flow, self.density, self.speed)
        self.first_order_derivative = first_order_derivative(self.flow, self.density, self.speed)
        self.model_dict = {"S3": self.model.S3,
                           }
        self.derivative = {"S3": self.first_order_derivative.S3,
                           }
        self.bounds = {"S3": np.asarray([[70, 80], [20, 60], [1, 8]]),  # vf, kc, foc
                       }
        self.x0 = {"S3": np.asarray([75, 35, 3.6]),
                   }

    def getSolution(self, model_str):
        # calibration
        objective = self.model_dict[model_str]
        derivative = self.derivative[model_str]
        bounds = self.bounds[model_str]
        x0 = self.x0[model_str]
        Adam = Adam_optimization(objective, derivative, bounds, x0)
        solutions, scores = Adam.adam()
        parameters = solutions[np.argmin(scores)]
        # obj = min(scores)
        return parameters


class DataAggregator:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initialize the database connection and output folder."""
        self.database_path = database_path
        self.output_folder = output_folder
        self.conn = sqlite3.connect(database_path)

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event

    def get_valid_link_ids(self):
        """Retrieve valid link_ids that should be included in calculations."""
        query = """
        SELECT tmc, link_ids FROM tmc_to_link
        WHERE tmc NOT IN (SELECT tmc FROM TMC_Identification WHERE road_order = 1)
        """
        df_links = pd.read_sql(query, self.conn)

        # Convert link_ids from a comma-separated string to a list
        valid_links = set()
        for link_list in df_links["link_ids"]:
            valid_links.update(link_list.split(','))

        return valid_links

    def get_lane_readings(self):
        """Retrieve and aggregate lane volume data by 15-minute intervals, then compute time-of-day averages."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_lane_readings")
            return pd.DataFrame()

        query = """
        SELECT zone_id, volume, local_time FROM lane_readings
        WHERE zone_id IN (223305, 197584, 197063, 196740, 196495, 197309, 197088)
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
        df_lane["time_of_day"] = df_lane["time_bin"].dt.time  # Ensure only time remains
        df_avg_lane = df_lane.groupby(["time_of_day", "is_weekend"])["volume"].mean().reset_index()

        return df_avg_lane

    def get_traj_volumes(self):
        """Retrieve and aggregate unique TripId counts by 15-minute intervals, then compute time-of-day averages."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_traj_volumes")
            return pd.DataFrame()

        query = """
        SELECT TripId, CrossingStartDateLocal FROM trajs
        """
        df_traj = pd.read_sql(query, self.conn)
        df_traj["CrossingStartDateLocal"] = pd.to_datetime(df_traj["CrossingStartDateLocal"])
        df_traj["time_bin"] = df_traj["CrossingStartDateLocal"].dt.floor("15min")
        df_traj["time_of_day"] = df_traj["time_bin"].dt.time
        df_traj["weekday"] = df_traj["CrossingStartDateLocal"].dt.weekday
        df_traj["is_weekend"] = df_traj["weekday"].apply(lambda x: 1 if x >= 5 else 0)

        # Aggregate unique TripId counts per 15-minute interval
        df_traj = df_traj.groupby(["time_bin", "is_weekend"])["TripId"].nunique().reset_index()

        # Compute daily time-of-day averages
        df_traj["time_of_day"] = df_traj["time_bin"].dt.time
        df_avg_traj = df_traj.groupby(["time_of_day", "is_weekend"])["TripId"].mean().reset_index()

        return df_avg_traj

    def compute_penetration_rate(self, df_lane, df_traj):
        """Compute penetration rates and return the merged dataset."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping compute_penetration_rate")
            return

        df_lane = self.get_lane_readings()
        df_traj = self.get_traj_volumes()

        df_lane = df_lane.rename(columns={"volume": "lane_volume"})
        df_traj = df_traj.rename(columns={"TripId": "traj_volume"})

        # Merge datasets
        df_merged = df_lane.merge(df_traj, on=["time_of_day", "is_weekend"], how="left")

        # Compute penetration rates
        df_merged["traj_penetration_rate"] = df_merged["traj_volume"] / df_merged["lane_volume"]

        self.penetration_rate_df = df_merged

        return df_merged

    def get_trajs_speed(self):
        """Retrieve trajs data and compute 15-minute average speed per SegmentId, excluding invalid link_id."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_segment_readings")
            return pd.DataFrame()

        query = """
        SELECT SegmentId, CrossingStartDateLocal, CrossingSpeedMph 
        FROM trajs
        """
        df_trajs = pd.read_sql(query, self.conn)

        if df_trajs.empty:
            raise ValueError("No trajs data found in trajs table.")

        # df_trajs["SegmentId"] = df_trajs["SegmentId"].astype(str)

        segment_link_df = pd.read_sql("SELECT SegmentId, link_id FROM SegmentId_to_link", self.conn)
        segment_link_df["SegmentId"] = segment_link_df["SegmentId"].astype(str)

        df_trajs = pd.merge(df_trajs, segment_link_df, on="SegmentId", how="left")
        df_trajs["link_id"] = df_trajs["link_id"].astype(str)

        valid_links = self.get_valid_link_ids()

        df_trajs = df_trajs[df_trajs["link_id"].isin(valid_links)]

        df_trajs["CrossingStartDateLocal"] = pd.to_datetime(df_trajs["CrossingStartDateLocal"])
        df_trajs["CrossingSpeedMph"] = pd.to_numeric(df_trajs["CrossingSpeedMph"], errors="coerce")
        df_trajs["time_bin"] = df_trajs["CrossingStartDateLocal"].dt.floor("15min")

        df_avg_speed = df_trajs.groupby(["link_id", "time_bin"])["CrossingSpeedMph"].mean().reset_index()

        df_avg_speed.rename(columns={"CrossingSpeedMph": "speed"}, inplace=True)

        # print("Found {} valid links.".format(len(valid_links)))
        df_avg_speed = df_avg_speed[df_avg_speed["link_id"].astype(str).isin(valid_links)]

        return df_avg_speed

    def get_link_volume(self):
        """Retrieve and compute the hourly adjusted traffic volume (flow) per link_id."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_link_volume")
            return pd.DataFrame()

        query = "SELECT TripId, SegmentId, CrossingStartDateLocal FROM trajs"
        df_volume = pd.read_sql(query, self.conn)

        segment_link_df = pd.read_sql("SELECT SegmentId, link_id FROM SegmentId_to_link", self.conn)
        segment_link_df["SegmentId"] = segment_link_df["SegmentId"].astype(str)

        df_volume = pd.merge(df_volume, segment_link_df, on="SegmentId", how="left")
        df_volume["link_id"] = df_volume["link_id"].astype(str)

        # Convert time column to datetime format
        df_volume["time"] = pd.to_datetime(df_volume["CrossingStartDateLocal"])
        df_volume["time_bin"] = df_volume["time"].dt.floor("15min")  # 15-minute bins
        df_volume["time_of_day"] = df_volume["time_bin"].dt.time.astype(str)  # Extract time as string

        # Aggregate unique agent_id counts per 15-minute interval
        df_link_volume = df_volume.groupby(["link_id", "time_bin"])["TripId"].nunique().reset_index()
        df_link_volume.rename(columns={"TripId": "volume"}, inplace=True)

        df_lane = self.get_lane_readings()
        df_traj = self.get_traj_volumes()
        self.compute_penetration_rate(df_lane, df_traj)

        # Convert penetration rate time_of_day to string for merging
        self.penetration_rate_df["time_of_day"] = self.penetration_rate_df["time_of_day"].astype(str)

        # Ensure df_link_volume has time_of_day before merging
        df_link_volume["time_of_day"] = df_link_volume["time_bin"].dt.time.astype(str)

        # Merge with penetration rate
        df_link_volume = df_link_volume.merge(self.penetration_rate_df[["time_of_day", "traj_penetration_rate"]], on="time_of_day", how="left")

        # Adjust volume using penetration rate
        df_link_volume["adjusted_volume"] = df_link_volume["volume"] / df_link_volume["traj_penetration_rate"]

        link_lanes_df = pd.read_sql("SELECT link_id, lanes FROM link", self.conn)
        link_lanes_df["link_id"] = link_lanes_df["link_id"].astype(str)

        # Merge lane count into df_link_volume
        df_link_volume["link_id"] = df_link_volume["link_id"].astype(str)

        df_link_volume = df_link_volume.merge(link_lanes_df, on="link_id", how="left")

        # Convert 15-minute adjusted volume to hourly flow
        df_link_volume["flow"] = df_link_volume["adjusted_volume"] * 4  # Convert to vehicles/hour
        df_link_volume["flow"] = df_link_volume["flow"] / df_link_volume["lanes"]

        valid_links = self.get_valid_link_ids()
        df_link_volume = df_link_volume[df_link_volume["link_id"].astype(str).isin(valid_links)]

        return df_link_volume

    # def compute_density(self):
    #     """Compute density using the road length: Density = Volume / Segment Length (converted to miles)."""
    #     df_speed = self.get_map_matching_readings()
    #     df_volume = self.get_link_volume()
    #     df_link = pd.read_sql("SELECT link_id, length FROM link", self.conn)
    #
    #     # Convert length from meters to miles
    #     df_link["length_miles"] = df_link["length"] / 1609.34
    #
    #     # Merge speed and volume data
    #     df_fd = df_speed.merge(df_volume, on=["link_id", "time_bin"], how="inner")
    #     df_fd = df_fd.merge(df_link[["link_id", "length_miles"]], on="link_id", how="left")
    #
    #     # Compute density: vehicles per mile
    #     df_fd["density"] = df_fd["adjusted_volume"] / df_fd["length_miles"]
    #
    #     # Remove invalid data where density <= 0 or length_miles is NaN
    #     df_fd = df_fd[df_fd["density"] > 0].dropna(subset=["length_miles"])
    #
    #     return df_fd

    def compute_density(self):
        """Compute density using Flow / Speed method: Density = Flow / Speed."""
        df_speed = self.get_trajs_speed()
        df_volume = self.get_link_volume()

        # Merge speed and volume data
        df_speed["link_id"] = df_speed["link_id"].astype(str)
        df_volume["link_id"] = df_volume["link_id"].astype(str)

        df_fd = df_speed.merge(df_volume, on=["link_id", "time_bin"], how="inner")

        if self.stop_event and self.stop_event.is_set():
            print("Stopping compute_density")
            return pd.DataFrame()

        # Compute density: vehicles per mile
        df_fd["density"] = df_fd["flow"] / df_fd["speed"]

        # Remove invalid data where density <= 0 or speed is NaN/zero
        df_fd = df_fd[(df_fd["density"] > 0) & df_fd["speed"].notna() & (df_fd["speed"] > 0)]

        return df_fd


class FDCalibratorTP:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initialize the Fundamental Diagram Calibrator."""
        self.database_path = database_path
        self.output_folder = output_folder
        self.output_path = os.path.join(self.output_folder, "odme_simulation", "fd_tp")

        # Ensure necessary directories exist
        os.makedirs(self.output_path, exist_ok=True)

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event

    def compute_fundamental_diagram_data(self):
        """ Compute link-level speed, volume, and density."""
        aggregator = DataAggregator(self.database_path, self.output_folder)
        df_fd = aggregator.compute_density()

        df_fd["time_of_day"] = df_fd["time_bin"].dt.time
        df_fd = df_fd[["link_id", "time_of_day", "speed", "flow", "density"]]

        df_fd.to_csv(os.path.join(self.output_path, "fundamental_diagram_data_tp.csv"), index=False)

        return df_fd

    def calibrate_fundamental_diagram(self, df_fd):
        """Calibrate the fundamental diagram model using S3."""
        observed_flow = df_fd["flow"].values
        observed_density = df_fd["density"].values
        observed_speed = df_fd["speed"].values

        solver = calibrate(observed_flow, observed_density, observed_speed)
        result = {"S3": solver.getSolution("S3"),
                  }
        return result, observed_flow, observed_density, observed_speed

    def generate_plots(self, observed_flow, observed_density, observed_speed, result):
        """Generate and save plots for the fundamental diagram model."""
        plot_results = plot_calibration_results(observed_flow, observed_density, observed_speed, result,
                                                self.output_path)
        plot_results.plot_qk()
        plot_results.plot_vk()
        plot_results.plot_vq()

    def compute_metrics(self, observed_flow, observed_density, observed_speed, result):
        """Compute RMSE and R^2 metrics for the calibration."""
        metrics = getMetrics(observed_flow, observed_density, observed_speed)
        S3_RMSE_SPEED_Overall, S3_RMSE_FLOW_Overall, S3_R2_SPEED_Overall, S3_R2_FLOW_Overall = metrics.S3_RMSE_Overall(
            result["S3"])
        S3_RMSE_Small_Range = metrics.S3_RMSE_Small_Range(result["S3"])

    def run(self):
        """Execute the fundamental diagram calibration process."""
        if self.stop_event and self.stop_event.is_set():
            return
        print("Processing data...")
        df_fd = self.compute_fundamental_diagram_data()

        if self.stop_event and self.stop_event.is_set():
            return
        print("Calibrating FD...")
        result, observed_flow, observed_density, observed_speed = self.calibrate_fundamental_diagram(df_fd)

        if self.stop_event and self.stop_event.is_set():
            return
        print(f"Calibrated Parameters: Free-Flow Speed (vf): {result['S3'][0]:.2f}, Critical Density (kc): {result['S3'][1]:.2f}, Curvature Parameter (foc): {result['S3'][2]:.2f}")

        print("Plotting FD...")
        self.generate_plots(observed_flow, observed_density, observed_speed, result)

        # print("Computing calibration metrics...")
        # self.compute_metrics(observed_flow, observed_density, observed_speed, result)

        print("FD Calibration Completed.")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")
    DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "output", "database", "unified_database.db")

    fd_calibrator_tp = FDCalibratorTP(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    fd_calibrator_tp.run()