import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from typing import Dict, Any
import os

class ModelEvaluator:
    """
    Evaluate trained models on test data,
    log metrics, save confusion matrices, and plot metric histograms.
    """

    def __init__(self, save_folder: str = "results"):
        self.results: pd.DataFrame = pd.DataFrame()
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.log_file = os.path.join(self.save_folder, "metrics_log.txt")
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def compute_metrics(
        self,
        trained_models: Dict[str, Any],
        X_test,
        y_test,
        feature_type: str = "CWT"
    ) -> pd.DataFrame:
        """
        Compute Accuracy, F1, Recall for each model and append to results DataFrame.
        Logs metrics to a text file.
        """
        metrics = []
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            metrics.append({
                "Model": name,
                "Feature_Type": feature_type,
                "Accuracy": acc,
                "F1": f1,
                "Recall": rec
            })

            with open(self.log_file, "a") as f:
                f.write(f"{feature_type} - {name} Metrics:\n")
                f.write(f"  Accuracy: {acc:.4f}\n")
                f.write(f"  F1 Score: {f1:.4f}\n")
                f.write(f"  Recall: {rec:.4f}\n\n")

        df_metrics = pd.DataFrame(metrics)
        self.results = pd.concat([self.results, df_metrics], ignore_index=True)

        print(df_metrics)

        return df_metrics

    def plot_confusion_matrices(
        self,
        trained_models: Dict[str, Any],
        X_test,
        y_test,
        feature_type: str = "CWT"
    ):
        """
        Plot and save confusion matrix for each model.
        """
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"{name} ({feature_type})")

            filename = os.path.join(self.save_folder, f"{feature_type}_{name}_confusion_matrix.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

    def plot_metric_histograms(self):
        """
        Plot a combined histogram comparing metrics across models and feature types.
        """
        if self.results.empty:
            raise ValueError("No metrics to plot. Run compute_metrics() first.")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics_list = ["Accuracy", "F1", "Recall"]

        for ax, metric in zip(axes, metrics_list):
            for feature_type in self.results["Feature_Type"].unique():
                subset = self.results[self.results["Feature_Type"] == feature_type]
                ax.bar(
                    subset["Model"] + f" ({feature_type})",
                    subset[metric],
                    alpha=0.7,
                    label=feature_type
                )
            ax.set_title(metric)
            ax.set_ylim(0, 1)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()

        filename = os.path.join(self.save_folder, "metrics_histogram.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()