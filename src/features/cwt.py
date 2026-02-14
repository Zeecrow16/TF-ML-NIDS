from typing import Dict, List

import numpy as np
import pandas as pd
import pywt


class CalculateCwt:
    """
    Class to compute CWT features (mean magnitude and mean phase) for UNSW-NB15 dataset.
    """

    def __init__(self, wavelet: str = "cmor3.5-1", scales: np.ndarray = None):
        self.wavelet = wavelet
        self.scales = scales if scales is not None else np.arange(1, 5)

    def calculate_all_categories(
        self, df: pd.DataFrame, signal_categories: Dict[str, List[str]]
    ):
        """
        Apply CWT to all feature groups in dataframe and add mean/phase columns.
        """
        df_cwt = df.copy()
        for cat, cols in signal_categories.items():
            mean_mag, mean_phase = self.calculate_cwt_category(df_cwt, cols)
            df_cwt[f"{cat}_mean"] = mean_mag
            df_cwt[f"{cat}_phase"] = mean_phase
        return df_cwt
    
    def calculate_cwt_category(self, df: pd.DataFrame, columns: List[str]):
        """
        Compute mean magnitude & phase for a feature
        """
        signals = df[columns].values
        results = np.array(
            [self.cwt_mean_features(row, self.scales, self.wavelet) for row in signals]
        )
        mean_mag = results[:, 0]
        mean_phase = results[:, 1]
        return mean_mag, mean_phase

    @staticmethod
    def cwt_mean_features(signal: np.ndarray, scales: np.ndarray, wavelet: str):
        """
        Compute CWT for a single signal row and extract mean magnitude & phase.
        """
        coeffs, _ = pywt.cwt(signal, scales=scales, wavelet=wavelet)
        magnitude = np.abs(coeffs)
        phase = np.angle(coeffs)
        return magnitude.mean(), phase.mean()

    def save_supervised_csv(self, df: pd.DataFrame, path: str):
        """
        Add binary label and save CSV for supervised learning.
        """
        df_copy = df.copy()
        df_copy["Label"] = (df_copy["Attack"] != "Benign").astype(int)
        df_copy.to_csv(path, index=False)
