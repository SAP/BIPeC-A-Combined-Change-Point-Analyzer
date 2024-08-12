import argparse
import pandas as pd
import bayesian_detector
import ruptures as rpt
from plot import plot_fig
from log import get_logger


class DetectRegression:
    """
    Detects changes in time series data using Bayesian change point detection and Pelt models.
    """

    def __init__(self, pen, log_odds_threshold, window_size, chunk_size):
        """
        Initializes the DetectRegression class with detection parameters.
        
        Args:
            pen (float): Penalty value for the Pelt model.
            log_odds_threshold (float): Log odds threshold for Bayesian detection.
            window_size (int): Sliding window size for segmenting the time series data.
            chunk_size (int): Number of points in each data segment for analysis.
        """
        self.pen = pen
        self.log_odds_threshold = log_odds_threshold
        self.window_size = window_size
        self.chunk_size = chunk_size

    def main(self, timeseries_data, column_name, plot):
        """
        Processes time series data to detect change points and optionally plots the results.
        
        Args:
            timeseries_data (str): Path to the CSV file containing the time series data.
            column_name (str) : CSV column which you want to run BIPeC.
            plot (bool): Flag to determine whether to plot the results.
        
        Returns:
            list: List of detected change points in the time series data.
        """
        
        df = pd.read_csv(timeseries_data)
        def merge_close_values(input_list, threshold):
            """
            Merges values that are close to each other within a specified threshold.
            
            Args:
                input_list (list of int): List of indices of change points.
                threshold (int): Threshold for merging close values.
                
            Returns:
                list: Merged list of change point indices.
            """
            if not input_list or len(input_list) == 1:
                return input_list

            input_list.sort()
            merged_values = [input_list[0]]

            for current_value in input_list[1:]:
                if current_value - merged_values[-1] <= threshold:
                    merged_values[-1] = current_value
                else:
                    merged_values.append(current_value)

            return merged_values

        threshold = 5
        df['normalized'] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
        timeseries_result = df['normalized']

        total_size = len(df)
        steps = (total_size - self.chunk_size) // self.window_size + 1
        last_index_processed = 0
        cp_list = []

        for step in range(steps):
            start_index = step * self.window_size
            end_index = start_index + self.chunk_size
            if start_index >= total_size:
                break
            if end_index > total_size:
                end_index = total_size
            current_window = timeseries_result[start_index:end_index]
            last_index_processed = end_index

            reshaped_window = current_window.values
            detector = bayesian_detector.ChangePointDetector([reshaped_window], threshold=self.log_odds_threshold)
            pos = detector.detect()
            if pos:
                algo = rpt.Pelt(model="rbf").fit(reshaped_window)
                pos = algo.predict(pen=self.pen)[:-1]
                get_logger().info('Pelt identified new change points at: ' + str(pos))

            absolute_indexes = [start_index + relative_index for relative_index in pos]
            cp_list.extend(absolute_indexes)

        if last_index_processed < total_size:
            final_start_index = max(total_size - self.chunk_size, 0)
            final_window = timeseries_result[final_start_index:]
            reshaped_window = final_window.values
            detector = bayesian_detector.ChangePointDetector([reshaped_window], threshold=self.log_odds_threshold)
            pos = detector.detect()
            if pos:
                algo = rpt.Pelt(model="rbf").fit(reshaped_window)
                pos = algo.predict(pen=self.pen)[:-1]
                get_logger().info('Pelt identified new change points at: ' + str(pos))

            absolute_indexes = [final_start_index + relative_index for relative_index in pos]
            cp_list.extend(absolute_indexes)

        unique_list = list(set(cp_list))
        change_point_list = merge_close_values(unique_list, threshold)

        # plot function
        if len(change_point_list) > 0 and plot:
            
            get_logger().info("Start ploting")
            plot_fig(timeseries_result, change_point_list, "./plot.jpg")

        get_logger().info(f"Final Change Points: {change_point_list}")
        return change_point_list


def main():
    parser = argparse.ArgumentParser(description='Detect Regression in Time Series Data.')
    parser.add_argument('--data', type=str, required=True, help='Path to the data file.')
    parser.add_argument('--column_name', type=str, help='Name of the column to extract.')
    parser.add_argument('--pen', type=float, default=10, help='Penalty value for the Pelt model.')
    parser.add_argument('--window_size', type=int, default=80, help='Number of data points in each sliding window for analysis.')
    parser.add_argument('--chunk_size', type=int, default=400, help='Number of data points in each chunk for processing time series data.')
    parser.add_argument('--log_odds_threshold', type=float, default=-5, help='Log odds threshold for Bayesian detection.')
    parser.add_argument('--plot', type=bool, default=False, help='Enable plotting of the results.')

    args = parser.parse_args()
    analyzer = DetectRegression(pen=args.pen, log_odds_threshold=args.log_odds_threshold, window_size=args.window_size, chunk_size=args.chunk_size)
    analyzer.main(timeseries_data=args.data, plot=args.plot, column_name=args.column_name)


if __name__ == '__main__':
    main()
