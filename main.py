import os

from src.config import (
    BASELINE_RESULTS_FOLDER,
    BENIGN_RATIO,
    CWT_OUTPUT_CSV,
    CWT_RESULTS_FOLDER,
    RANDOM_STATE,
    RAW_DATA_CSV,
    SCALES,
    TOTAL_SAMPLES,
    WAVELET,
)
from src.features.cwt import CalculateCwt
from src.models.evaluate import ModelEvaluator
from src.models.train_baseline import BaselineTraining
from src.models.train_cwt import CwtTraining
from src.preprocessing.clean_data import ProcessDataset

# Full ML pipeline:


def run_pipeline():
    #  Load and preprocess dataset
    dataset = ProcessDataset(csv_path=RAW_DATA_CSV)
    dataset.load_data()
    dataset.clean_data()
    dataset.sample_data(total_samples=TOTAL_SAMPLES, benign_ratio=BENIGN_RATIO)
    df_cleaned = dataset.df_sampled

    #  Compute CWT features
    signal_categories = {
        # "pkt": dataset.packet_size,
        "src": dataset.src_to_dst_timing,
        "dst": dataset.dst_to_src_timing,
        "thr": dataset.throughput_signal,
    }
    cwt_creator = CalculateCwt(wavelet=WAVELET, scales=SCALES)
    df_cwt = cwt_creator.calculate_all_categories(df_cleaned, signal_categories)

    output_csv = CWT_OUTPUT_CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    cwt_creator.save_supervised_csv(df_cwt, output_csv)

    #  Train CWT models
    trainer_cwt = CwtTraining(csv_path=output_csv)
    trainer_cwt.load_and_prepare_data()
    trainer_cwt.train_logistic_regression()
    trainer_cwt.train_knn()
    trainer_cwt.train_random_forest()

    # Evaluate CWT models
    evaluator_cwt = ModelEvaluator(save_folder=CWT_RESULTS_FOLDER)
    evaluator_cwt.compute_metrics(
        trained_models=trainer_cwt.trained_models,
        X_test=trainer_cwt.X_test,
        y_test=trainer_cwt.y_test,
        feature_type="CWT",
    )
    evaluator_cwt.plot_confusion_matrices(
        trained_models=trainer_cwt.trained_models,
        X_test=trainer_cwt.X_test,
        y_test=trainer_cwt.y_test,
        feature_type="CWT",
    )
    evaluator_cwt.plot_metric_histograms()

    # Train Baseline models
    df_baseline_sample = df_cleaned.sample(n=TOTAL_SAMPLES, random_state=RANDOM_STATE)
    trainer_baseline = BaselineTraining(df=df_baseline_sample)
    trainer_baseline.load_and_prepare_data()
    trainer_baseline.train_logistic_regression()
    trainer_baseline.train_knn()
    trainer_baseline.train_random_forest()

    # Evaluate Baseline models
    evaluator_baseline = ModelEvaluator(save_folder=BASELINE_RESULTS_FOLDER)
    evaluator_baseline.compute_metrics(
        trained_models=trainer_baseline.trained_models,
        X_test=trainer_baseline.X_test_scaled,
        y_test=trainer_baseline.y_test,
        feature_type="Baseline",
    )
    evaluator_baseline.plot_confusion_matrices(
        trained_models=trainer_baseline.trained_models,
        X_test=trainer_baseline.X_test_scaled,
        y_test=trainer_baseline.y_test,
        feature_type="Baseline",
    )
    evaluator_baseline.plot_metric_histograms()


if __name__ == "__main__":
    run_pipeline()
