from ml4xcube.evaluation.metrics import *
from typing import Dict, List, Tuple, Callable

class Evaluator:
    def __init__(self, framework: str):
        """
        Initializes the Evaluator class, which handles metric evaluation for a specific framework.

        Args:
            framework (str): The deep learning framework being used ('pytorch', 'tensorflow' or 'sklearn').

        Attributes:
            metric_functions (Dict[str, Callable]): A dictionary mapping metric names to their corresponding functions.
                Supported metrics include:
                - 'mae': Mean Absolute Error
                - 'mse': Mean Squared Error
                - 'rmse': Root Mean Squared Error
                - 'r2': R-squared
                - 'huber_loss': Huber Loss
                - 'mape': Mean Absolute Percentage Error
                - 'med_ae': Median Absolute Error
                - 'explained_variance': Explained Variance
                - 'accuracy': Accuracy
                - 'roc_auc': ROC AUC score
                - 'cross_entropy': Cross-Entropy Loss
                - 'precision': Precision score
                - 'recall': Recall score
                - 'f1_score': F1 Score
        """
        self.framework = framework
        self.metric_functions = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'huber_loss': huber_loss,
            'mape': mape,
            'med_ae': median_ae,
            'explained_variance': explained_variance,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cross_entropy': cross_entropy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }


    def get_metrics(self, metric_names: List[str], average: str = 'macro', delta: float = 1.0) -> Dict[str, Callable]:
        """
        Get a dictionary of metric functions based on the selected framework, metric names, and optional parameters.

        Args:
            metric_names (List[str]): List of metric names to retrieve.
            average (str): Averaging method for precision, recall, and F1 score (default: 'macro').
            delta (float): Delta parameter for Huber loss (default: 1.0).

        Returns:
            Dict[str, Callable]: Dictionary with metric names as keys and their corresponding functions as values.
        """
        metrics = {}
        for name in metric_names:
            if name in self.metric_functions:
                if name in ['precision', 'recall', 'f1_score']:
                    # Pass the `average` parameter
                    metrics[name] = self.metric_functions[name](framework=self.framework, default_average=average)
                elif name == 'huber_loss':
                    # Pass the `delta` parameter
                    metrics[name] = self.metric_functions[name](framework=self.framework, delta=delta)
                else:
                    # No additional parameters
                    metrics[name] = self.metric_functions[name](framework=self.framework)
            else:
                raise KeyError(f"Metric '{name}' is not supported.")
        return metrics
