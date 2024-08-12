import copy
import numpy as np
import math
import warnings
import logging
from scipy.special import gammaln, logsumexp


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set up logging to console
logging.basicConfig(level=logging.INFO)  # Set to WARNING to suppress debug information
logger = logging.getLogger('Bayesian_detector')

class LogNormalDistribution:
    """
    LogNormalDistribution class to calculate the mean and variance of log-transformed data.
    """
    @classmethod
    def calculate_mean_and_variance(cls, data: np.ndarray) -> tuple[float, float]:
        """
        Calculate the mean and variance of log-transformed data.
        
        Parameters:
            data (np.ndarray): The data to be transformed.
        
        Returns:
            tuple[float, float]: The mean and variance of the log-transformed data.
        """
        # Replace or remove zero values to avoid log(0) resulting in nan
        log_transformed_data = np.log(np.where(data == 0, np.finfo(float).eps, data))
        mean_log = log_transformed_data.sum() / len(data)
        variance_log = ((log_transformed_data - mean_log) ** 2).sum() / len(data)
        return mean_log, variance_log

class ChangePointDetector:
    """
    Change point detector using Bayesian statistics.
    """
    def __init__(self, time_series_list, threshold=0):
        """
        Initialize the ChangePointDetector object.
        
        Parameters:
            time_series_list (list of np.ndarray): List of time series data.
            threshold (float): The threshold for log odds to identify significant change points.
        """
        self.time_series_list = [np.asarray(ts) for ts in copy.deepcopy(time_series_list)]
        self.n_series = len(self.time_series_list)
        self.series_lengths = [len(ts) for ts in self.time_series_list]
        self.threshold = threshold
        self.distribution = LogNormalDistribution()
        self.precomputed_log_gamma_values = self._precompute_log_gamma_values(max(self.series_lengths))

    def _precompute_log_gamma_values(self, max_length: int) -> list:
        """
        Generate a table of precomputed log-gamma values for use in Bayesian calculations.
        
        Parameters:
            max_length (int): The maximum length of the time series data.
        
        Returns:
            list: A list of precomputed log-gamma values.
        """
        log_gamma_values = [-99, -99, -99]
        for i in range(3, max_length + 1):
            log_gamma_values.append(gammaln(0.5 * i - 1))
        return log_gamma_values

    def _calculate_bayes_factor(self, segment: np.ndarray) -> tuple:
        """
        Calculate the Bayes factor for a segment of the time series data.
        
        Parameters:
            segment (np.ndarray): The segment of the time series data.
        
        Returns:
            tuple: A tuple containing the normalized probability, the index of the highest probability change point, and the log odds ratio.
        """
        # Ensure the segment is long enough
        n = len(segment)
        if n < 10:
            logger.debug('Timeseries length is less than 10 points')
            return None

        # Calculate mean and variance for the entire segment
        mean_log, variance_log = self.distribution.calculate_mean_and_variance(segment)
        if variance_log <= 0:
            logger.debug('Variance is non-positive, returning None')
            return None

        # Precompute common factors
        log_gamma_value = self.precomputed_log_gamma_values[n]
        denom = (1.5 * np.log(np.pi) + (-n / 2.0 + 0.5) * np.log(n * variance_log) + log_gamma_value)
        
        # Initialize weights
        weights = [0, 0, 0]
        
        for i in range(3, n - 2):
            # Divide segment into two parts
            segment_a = segment[:i]
            segment_b = segment[i:]
            
            # Calculate mean and variance for each part
            mean_a_log, var_a_log = self.distribution.calculate_mean_and_variance(segment_a)
            mean_b_log, var_b_log = self.distribution.calculate_mean_and_variance(segment_b)
            
            # Calculate the components of the weight
            weight_a = ((-0.5 * len(segment_a) + 0.5) * np.log(len(segment_a)) + 
                        (-0.5 * len(segment_a) + 1) * np.log(var_a_log) + 
                        self.precomputed_log_gamma_values[len(segment_a)])
            
            weight_b = ((-0.5 * len(segment_b) + 0.5) * np.log(len(segment_b)) + 
                        (-0.5 * len(segment_b) + 1) * np.log(var_b_log) + 
                        self.precomputed_log_gamma_values[len(segment_b)])
            
            combined_variance = var_a_log + var_b_log
            mean_product = mean_a_log ** 2 * mean_b_log ** 2
            combined_variance_mean_term = np.log(combined_variance) + np.log(mean_product)
            
            weights.append((weight_a + weight_b) - combined_variance_mean_term)
        
        # Add padding
        weights.extend([0, 0])
        weights = np.array(weights)

        # Compute the final log odds ratio
        num = 2.5 * np.log(2.0) + np.log(abs(mean_log)) + weights.mean()
        log_odds_ratio = num - denom
        
        # Adjust weights for normalization
        weights[:3] = weights[-2:] = -np.inf
        norm = logsumexp(weights[3:-2])
        normalized_probability = np.exp(weights[3:-3] - norm)

        # Check if log odds ratio meets the threshold
        if math.isnan(log_odds_ratio) or log_odds_ratio < self.threshold:
            return normalized_probability, weights.argmax(), log_odds_ratio, None

        max_probability_index = weights.argmax()
        return normalized_probability, max_probability_index, log_odds_ratio

    def _detect_change_points(self, time_series: np.ndarray) -> list:
        """
        Detect change points in a given time series using recursive splitting.
        
        Parameters:
            time_series (np.ndarray): The time series data.
        
        Returns:
            list: A list of detected change points.
        """
        def recursive_split(segment, start, end):
            result = self._calculate_bayes_factor(segment[start:end])
            if result is None or result[-1] is None:
                return []

            log_odds_ratio, split_point = result[2], start + result[1]
            if log_odds_ratio < self.threshold:
                return []

            left_splits = recursive_split(segment, start, split_point)
            right_splits = recursive_split(segment, split_point, end)
            return left_splits + [split_point] + right_splits

        return recursive_split(time_series, 0, len(time_series))

    def detect(self) -> list:
        """
        Detect change points in all provided time series data.
        
        Returns:
            list: A list of all detected change points across all time series.
        """
        all_change_points = []
        for time_series in self.time_series_list:
            change_points = self._detect_change_points(time_series)
            change_points = [int(cp) for cp in change_points]  # Convert numpy int64 to native int
            all_change_points.extend(change_points)  # Add all change points to a single list
            logger.info(f'Bayesian detected change points: {all_change_points}')

        return all_change_points

