from typing import Dict

import numpy as np
import pandas as pd


class ProcessDataset:
    """
    Class to handle loading, cleaning and sampling the UNSW-NB15 dataset.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.df_clean = None
        self.df_sampled = None

        # self.packet_size = [
        #     "NUM_PKTS_UP_TO_128_BYTES",
        #     "NUM_PKTS_128_TO_256_BYTES",
        #     "NUM_PKTS_256_TO_512_BYTES",
        #     "NUM_PKTS_512_TO_1024_BYTES",
        #     "NUM_PKTS_1024_TO_1514_BYTES",
        # ]

        self.src_to_dst_timing = [
            "SRC_TO_DST_IAT_MIN",
            "SRC_TO_DST_IAT_AVG",
            "SRC_TO_DST_IAT_MAX",
            "SRC_TO_DST_IAT_STDDEV",
        ]

        self.dst_to_src_timing = [
            "DST_TO_SRC_IAT_MIN",
            "DST_TO_SRC_IAT_AVG",
            "DST_TO_SRC_IAT_MAX",
            "DST_TO_SRC_IAT_STDDEV",
        ]

        # self.byte_signal = ["IN_BYTES", "OUT_BYTES"]

        self.throughput_signal = [
            "SRC_TO_DST_AVG_THROUGHPUT",
            "DST_TO_SRC_AVG_THROUGHPUT",
        ]

        self.cwt_columns = (
            # self.packet_size
            self.src_to_dst_timing
            + self.dst_to_src_timing
            # + self.byte_signal
            + self.throughput_signal
        )

    def load_data(self):
        """
        Load data from CSV
        """
        self.df = pd.read_csv(self.csv_path)

    def clean_data(self):
        """Remove NaN's and infinite values from CWT columns"""
        self.df_clean = (
            self.df[self.cwt_columns + ["Attack"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        self.df_clean["Attack"] = self.df_clean["Attack"].astype(str)

    def sample_data(
        self,
        total_samples: int = 100_000,
        benign_ratio: float = 0.85,
        min_attack_samples: int = 500,
        random_state: int = 42,
    ) -> None:
        """
        Sample representative benign and attack traffic.
        """

        benign_target = int(total_samples * benign_ratio)
        attack_target = total_samples - benign_target

        benign_df = self.df_clean[self.df_clean["Attack"] == "Benign"]
        attack_df = self.df_clean[self.df_clean["Attack"] != "Benign"]

        benign_sample = benign_df.sample(
            n=min(len(benign_df), benign_target), random_state=random_state
        )

        attack_counts = attack_df["Attack"].value_counts(normalize=True)
        attack_samples = []

        for attack, ratio in attack_counts.items():
            n = max(int(ratio * attack_target), min_attack_samples)
            subset = attack_df[attack_df["Attack"] == attack]

            attack_samples.append(
                subset.sample(n=min(len(subset), n), random_state=random_state)
            )

        attack_sample = pd.concat(attack_samples)

        self.df_sampled = (
            pd.concat([benign_sample, attack_sample])
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    def get_cwt_features(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Organise sampled data into feature groups for CWT.
        """

        attack_signals = {}
        for attack_type in self.df_sampled["Attack"].unique():
            subset = self.df_sampled[self.df_sampled["Attack"] == attack_type]

            attack_signals[attack_type] = {
                # "packet_size": subset[self.packet_size].values,
                "src_to_dst_timing": subset[self.src_to_dst_timing].values,
                "dst_to_src_timing": subset[self.dst_to_src_timing].values,
                # "byte_signal": subset[self.byte_signal].values,
                "throughput_signal": subset[self.throughput_signal].values,
            }

        return attack_signals
