import numpy as np
from typing import Dict, List, Tuple, Callable


def get_tf_metrics(y_true: 'tf.Tensor', y_pred: 'tf.Tensor', metric: str) -> Tuple[int, Dict[str, List['tf.Tensor']]]:
    """
    Calculate precision, recall, or F1 scores across all classes and return the number of classes and the calculated metrics.

    Args:
        y_true (tf.Tensor): Ground truth labels.
        y_pred (tf.Tensor): Predicted scores or logits.
        metric (str): The metric to compute ('f1', 'precision', 'recall').

    Returns:
        Tuple[int, Dict[str, List[tf.Tensor]]: A tuple containing:
            int: The number of classes determined from y_true.
            Dict: A dictionary containing the requested metrics including lists of Tensors for precision,
                recall, true positives, false positives, and false negatives as applicable.
    """
    import tensorflow as tf

    y_true = tf.cast(y_true, tf.int32)  # Ensure y_true is int32
    y_pred_classes = tf.argmax(y_pred, axis=1, output_type=tf.int32)  # Convert logits to predicted class labels

    num_classes = tf.reduce_max(y_true) + 1
    metrics = {
        'precision': [],
        'recall': [],
        'class_true_positives': [],
        'class_false_positives': [],
        'class_false_negatives': []
    }

    for class_id in tf.range(num_classes):
        class_id_int = tf.cast(class_id, tf.int32)

        # Class specific predictions and labels
        class_y_true = tf.equal(y_true, class_id_int)
        class_y_pred = tf.equal(y_pred_classes, class_id_int)

        true_positives = tf.reduce_sum(tf.cast(class_y_true & class_y_pred, tf.float32))
        false_positives = tf.reduce_sum(tf.cast(~class_y_true & class_y_pred, tf.float32))
        false_negatives = tf.reduce_sum(tf.cast(class_y_true & ~class_y_pred, tf.float32))

        if metric in ['f1', 'recall']:
            recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
            metrics['recall'].append(recall)
            metrics['class_true_positives'].append(true_positives)
            metrics['class_false_negatives'].append(false_negatives)

        if metric in ['f1', 'precision']:
            precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
            metrics['precision'].append(precision)
            if metric == 'precision':
                metrics['class_true_positives'].append(true_positives)
            metrics['class_false_positives'].append(false_positives)

    return num_classes, {key: value for key, value in metrics.items() if value}


def mae(framework: str = None) -> Callable:
    """
    Returns the mean absolute error (MAE) function for the specified framework.

    Args:
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').

    Returns:
        Callable: The MAE function for the specified framework.
    """
    if framework == 'pytorch':
        import torch
        return torch.nn.functional.l1_loss
    elif framework == 'sklearn':
        import sklearn.metrics as skm
        return skm.mean_absolute_error
    elif framework == 'tensorflow':
        import tensorflow as tf
        return lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_true - y_pred))


def mse(framework: str = None) -> Callable:
    """
    Returns the mean squared error (MSE) function for the specified framework.

    Args:
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').

    Returns:
        Callable: The MSE function for the specified framework.
    """
    if framework == 'pytorch':
        import torch
        return torch.nn.functional.mse_loss
    elif framework == 'sklearn':
        import sklearn.metrics as skm
        return skm.mean_squared_error
    elif framework == 'tensorflow':
        import tensorflow as tf
        return tf.keras.losses.MSE


def rmse(framework: str = None) -> Callable:
    """
    Returns the root mean squared error (RMSE) function for the specified framework.

    Args:
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').

    Returns:
        Callable: The RMSE function for the specified framework.
    """
    if framework == 'pytorch':
        import torch
        return lambda y_true, y_pred: torch.sqrt(torch.nn.functional.mse_loss(y_pred, y_true))
    elif framework == 'sklearn':
        import sklearn.metrics as skm
        return lambda y_true, y_pred: np.sqrt(skm.mean_squared_error(y_true, y_pred))
    elif framework == 'tensorflow':
        import tensorflow as tf
        return lambda y_true, y_pred: tf.sqrt(tf.keras.losses.MSE(y_true, y_pred))


def r2(framework: str = None) -> Callable:
    """
    Returns the R-squared (R2) score function for the specified framework.

    Args:
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').

    Returns:
        Callable: The R2 function for the specified framework.
    """
    if framework == 'pytorch':
        import torch
        return lambda y_true, y_pred: 1 - torch.sum((y_true - y_pred) ** 2) / torch.sum((y_true - torch.mean(y_true)) ** 2)
    elif framework == 'sklearn':
        import tensorflow.metrics as skm
        return skm.r2_score
    elif framework == 'tensorflow':
        import tensorflow as tf
        return lambda y_true, y_pred: 1 - tf.reduce_sum(tf.square(y_true - y_pred)) / tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))


def huber_loss(framework: str = None, delta=1.0) -> Callable:
    """
    Returns the Huber loss function for the specified framework.

    Args:
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').
        delta (float): The delta parameter for the Huber loss.

    Returns:
        Callable: The Huber loss function for the specified framework.
    """
    if framework == 'pytorch':
        import torch
        return lambda y_true, y_pred: torch.nn.functional.huber_loss(y_pred, y_true, delta=delta)
    elif framework == 'sklearn':
        return lambda y_true, y_pred: np.mean(np.where(np.abs(y_true - y_pred) < delta, 0.5 * (y_true - y_pred) ** 2, delta * (np.abs(y_true - y_pred) - 0.5 * delta)))
    elif framework == 'tensorflow':
        import tensorflow as tf
        return tf.keras.losses.Huber(delta=delta)


def mape(framework: str = None) -> Callable:
    """
    Returns the mean absolute percentage error (MAPE) function for the specified framework.

    Args:
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').

    Returns:
        Callable: The MAPE function for the specified framework.
    """
    if framework == 'pytorch':
        import torch
        return lambda y_true, y_pred: torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
    elif framework == 'sklearn':
        return lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    elif framework == 'tensorflow':
        import tensorflow as tf
        return lambda y_true, y_pred: tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100


def median_ae(framework: str = None) -> Callable:
    """
    Returns the median absolute error (MedAE) function for the specified framework.

    Args:
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').

    Returns:
        Callable: The MedAE function for the specified framework.
    """
    if framework == 'pytorch':
        import torch
        return lambda y_true, y_pred: torch.median(torch.abs(y_true - y_pred))
    elif framework == 'sklearn':
        import sklearn.metrics as skm
        return skm.median_absolute_error
    elif framework == 'tensorflow':
        import tensorflow as tf
        return lambda y_true, y_pred: (tf.sort(tf.abs(y_true - y_pred))[tf.shape(y_true)[0] // 2])


def explained_variance(framework: str = None) -> Callable:
    """
    Returns the explained variance score function for the specified framework.

    Args:
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').

    Returns:
        Callable: The explained variance function for the specified framework.
    """
    if framework == 'pytorch':
        import torch
        return lambda y_true, y_pred: 1 - torch.var(y_true - y_pred) / torch.var(y_true)
    elif framework == 'sklearn':
        import sklearn.metrics as skm
        return skm.explained_variance_score
    elif framework == 'tensorflow':
        import tensorflow as tf
        return lambda y_true, y_pred: 1 - tf.math.reduce_variance(y_true - y_pred) / tf.math.reduce_variance(y_true)


def accuracy(framework: str = None) -> Callable:
    """
    Returns the accuracy score function for the specified framework.

    Args:
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').

    Returns:
        Callable: The accuracy function for the specified framework.
    """
    if framework == 'pytorch':
        import torch
        return lambda y_true, y_pred: torch.mean((y_pred.argmax(dim=-1) == y_true).float()).item()
    elif framework == 'sklearn':
        import sklearn.metrics as skm
        return lambda y_true, y_pred: skm.accuracy_score(y_true, y_pred.argmax(axis=1) if y_pred.ndim > 1 else y_pred)
    elif framework == 'tensorflow':
        import tensorflow as tf
        return lambda y_true, y_pred: tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), tf.cast(y_true, tf.int64)), tf.float32))


def roc_auc(framework: str = None, average: str = 'macro') -> Callable:
    """
    Returns the ROC AUC score function for the specified framework.

    Args:
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').
        average (str): The averaging method for multiclass ROC AUC ('macro', 'micro', etc.).

    Returns:
        Callable: The ROC AUC function for the specified framework.
    """
    import sklearn.metrics as skm

    if framework == 'pytorch':
        import torch
        return lambda y_true, y_pred: skm.roc_auc_score(
            torch.nn.functional.one_hot(y_true, num_classes=y_pred.shape[1]).numpy(),
            y_pred.numpy(),
            multi_class="ovr",
            average=average
        ) if y_pred.ndim == 2 else skm.roc_auc_score(y_true.numpy(), y_pred.numpy(), average=average)
    elif framework == 'sklearn':
        from sklearn.preprocessing import label_binarize
        return lambda y_true, y_pred: skm.roc_auc_score(
            label_binarize(y_true, classes=np.unique(y_true)),
            y_pred if y_pred.ndim > 1 else label_binarize(y_pred, classes=np.unique(y_true)),
            multi_class="ovr" if len(np.unique(y_true)) > 2 else "raise",
            average=average
        )
    elif framework == 'tensorflow':
        import tensorflow as tf
        return lambda y_true, y_pred: skm.roc_auc_score(
            tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1]).numpy(),
            y_pred.numpy(),
            multi_class="ovr",
            average=average
        )


def cross_entropy(framework: str = None) -> Callable:
    """
    Returns the cross-entropy loss function for the specified framework.

    Args:
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').

    Returns:
        Callable: The cross-entropy loss function for the specified framework.
    """
    if framework == 'pytorch':
        import torch
        # Assuming y_true is not one-hot encoded and y_pred are logits
        return torch.nn.functional.cross_entropy
    elif framework == 'sklearn':
        import sklearn.metrics as skm
        # For sklearn, y_pred should be probabilities
        return skm.log_loss
    elif framework == 'tensorflow':
        import tensorflow as tf
        # TensorFlow expects y_true to be one-hot encoded for categorical cross-entropy
        return lambda y_true, y_pred: tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
            tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1]), y_pred))


def tf_precision(y_true: 'tf.Tensor', y_pred: 'tf.Tensor', average: str = 'macro') -> 'tf.Tensor':
    """
    Calculates the precision for TensorFlow models, supporting various averaging methods (macro, micro, weighted).

    Args:
        y_true (tf.Tensor): Ground truth labels.
        y_pred (tf.Tensor): Predicted logits or scores.
        average (str): Averaging method for the precision score ('macro', 'micro', 'weighted').

    Returns:
        tf.Tensor: The calculated precision score.
    """
    import tensorflow as tf

    num_classes, metrics = get_tf_metrics(y_true, y_pred, 'precision')
    class_true_positives = metrics['class_true_positives']
    class_false_positives = metrics['class_false_positives']
    all_precisions = metrics['precision']

    if average == 'macro':
        # Macro average: calculate metrics independently for each class and then average them
        macro_precision = tf.reduce_mean(all_precisions)
        return macro_precision
    elif average == 'micro':
        # Micro average: aggregate the contributions of all classes to compute the overall metric
        total_true_positives = tf.reduce_sum(class_true_positives)
        print('total TP ', total_true_positives)
        total_false_positives = tf.reduce_sum(class_false_positives)
        print('total FP ', total_false_positives)

        # Compute micro precision
        micro_precision = total_true_positives / (
                total_true_positives + total_false_positives + tf.keras.backend.epsilon())
        return micro_precision
    elif average == 'weighted':
        # Weighted average: calculate metrics for each class, weighted by the number of true instances
        weights = tf.cast(tf.math.bincount(tf.cast(y_true, tf.int32)), tf.float32)
        total = tf.reduce_sum(weights)
        weighted_precision = tf.reduce_sum(all_precisions * weights / total)
        return weighted_precision


def precision(framework: str = None, default_average: str = 'macro') -> Callable:
    """
    Returns the precision score function for the specified framework, with support for different averaging methods.

    Args:
        default_average (str): Default averaging method ('macro', 'micro', 'weighted').
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').

    Returns:
        Callable: The precision function for the specified framework.
    """
    if framework == 'pytorch':
        import torch
        import torcheval.metrics as tmetrics
        return lambda y_true, y_pred, average=default_average: tmetrics.MulticlassPrecision(
            num_classes=len(y_true.unique()), average=average).update(torch.argmax(y_pred, dim=1), y_true).compute()
    elif framework == 'tensorflow':
        return lambda y_true, y_pred, average=default_average: tf_precision(y_true, y_pred, average)
    elif framework == 'sklearn':
        import sklearn.metrics as skm
        return lambda y_true, y_pred, average=default_average: skm.precision_score(
            y_true, y_pred.argmax(axis=1) if y_pred.ndim > 1 else y_pred, average=average)


def tf_recall(y_true: 'tf.Tensor', y_pred: 'tf.Tensor', average: str = 'macro') -> 'tf.Tensor':
    """
    Calculates the recall for TensorFlow models, supporting various averaging methods (macro, micro, weighted).

    Args:
        y_true (tf.Tensor): Ground truth labels.
        y_pred (tf.Tensor): Predicted logits or scores.
        average (str): Averaging method for the recall score ('macro', 'micro', 'weighted').

    Returns:
        tf.Tensor: The calculated recall score.
    """
    import tensorflow as tf

    num_classes, metrics = get_tf_metrics(y_true, y_pred, 'recall')
    class_true_positives = metrics['class_true_positives']
    class_false_negatives = metrics['class_false_negatives']
    all_recalls = metrics['recall']

    if average == 'macro':
        # Macro average: calculate metrics independently for each class and then average them
        macro_recall = tf.reduce_mean(all_recalls)
        return macro_recall
    elif average == 'micro':
        # Micro average: aggregate the contributions of all classes to compute the overall metric
        total_true_positives = tf.reduce_sum(class_true_positives)
        total_condition_positives = tf.reduce_sum(class_true_positives) + tf.reduce_sum(class_false_negatives)

        # Compute micro recall
        micro_recall = total_true_positives / (total_condition_positives + tf.keras.backend.epsilon())
        return micro_recall
    elif average == 'weighted':
        # Weighted average: calculate metrics for each class, weighted by the number of true instances
        weights = tf.cast(tf.math.bincount(tf.cast(y_true, tf.int32)), tf.float32)
        total = tf.reduce_sum(weights)
        weighted_recall = tf.reduce_sum(all_recalls * weights / total)
        return weighted_recall


def recall(framework: str = None, default_average = 'macro') -> Callable:
    """
    Returns the recall score function for the specified framework, with support for different averaging methods.

    Args:
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').
        default_average (str): Default averaging method ('macro', 'micro', 'weighted').

    Returns:
        Callable: The recall function for the specified framework.
    """
    if framework == 'pytorch':
        import torch
        import torcheval.metrics as tmetrics
        return lambda y_true, y_pred, average=default_average: tmetrics.MulticlassRecall(
            num_classes=len(y_true.unique()), average=average).update(torch.argmax(y_pred, dim=1), y_true).compute()
    elif framework == 'tensorflow':
        return lambda y_true, y_pred, average=default_average: tf_recall(y_true, y_pred, average)
    elif framework == 'sklearn':
        import sklearn.metrics as skm
        return lambda y_true, y_pred, average=default_average: skm.recall_score(
            y_true, y_pred.argmax(axis=1) if y_pred.ndim > 1 else y_pred, average=average)


def tf_f1_score(y_true: 'tf.Tensor', y_pred: 'tf.Tensor', average: str = 'macro') -> 'tf.Tensor':
    """
    Calculates the F1 score for TensorFlow models, supporting various averaging methods (macro, micro, weighted).

    Args:
        y_true (tf.Tensor): Ground truth labels.
        y_pred (tf.Tensor): Predicted logits or scores.
        average (str): Averaging method for the F1 score ('macro', 'micro', 'weighted').

    Returns:
        tf.Tensor: The calculated F1 score.
    """
    import tensorflow as tf

    num_classes, metrics = get_tf_metrics(y_true, y_pred, 'f1')
    class_true_positives = metrics['class_true_positives']
    class_false_positives = metrics['class_false_positives']
    class_false_negatives = metrics['class_false_negatives']
    all_precisions = metrics['precision']
    all_recalls = metrics['recall']

    if average == 'macro':
        # Macro average: Calculate F1 for each class independently and average them
        f1_scores = [2 * (p * r) / (p + r + tf.keras.backend.epsilon()) for p, r in
                     zip(all_precisions, all_recalls)]
        macro_f1 = tf.reduce_mean(f1_scores)
        return macro_f1
    elif average == 'micro':
        # Micro average: Aggregate the contributions of all classes
        total_tp = tf.reduce_sum(class_true_positives)
        total_fp = tf.reduce_sum(class_false_positives)
        total_fn = tf.reduce_sum(class_false_negatives)
        micro_precision = total_tp / (total_tp + total_fp + tf.keras.backend.epsilon())
        micro_recall = total_tp / (total_tp + total_fn + tf.keras.backend.epsilon())
        micro_f1 = 2 * (micro_precision * micro_recall) / (
                micro_precision + micro_recall + tf.keras.backend.epsilon())
        return micro_f1
    elif average == 'weighted':
        # Weighted average: Calculate F1 for each class, weighted by the number of true instances
        weights = tf.cast(tf.math.bincount(tf.cast(y_true, tf.int32)), tf.float32)
        total = tf.reduce_sum(weights)
        f1_scores = [2 * (p * r) / (p + r + tf.keras.backend.epsilon()) for p, r in
                     zip(all_precisions, all_recalls)]
        weighted_f1 = tf.reduce_sum(f1_scores * weights / total)
        return weighted_f1


def f1_score(framework: str = None, default_average: str = 'macro') -> Callable:
    """
    Returns the F1 score function for the specified framework, with support for different averaging methods.

    Args:
        framework (str): The framework to use ('pytorch', 'tensorflow', 'sklearn').
        default_average (str): Default averaging method ('macro', 'micro', 'weighted').

    Returns:
        Callable: The F1 score function for the specified framework.
    """
    if framework == 'pytorch':
        import torch
        import torcheval.metrics as tmetrics
        return lambda y_true, y_pred, average=default_average: tmetrics.MulticlassF1Score(num_classes=len(y_true.unique()), average=average).update(torch.argmax(y_pred, dim=1), y_true).compute()
    elif framework == 'tensorflow':
        return lambda y_true, y_pred, average=default_average: tf_f1_score(y_true, y_pred, average)
    elif framework == 'sklearn':
        import sklearn.metrics as skm
        return lambda y_true, y_pred, average=default_average: skm.f1_score(
            y_true, y_pred.argmax(axis=1) if y_pred.ndim > 1 else y_pred, average=average)

