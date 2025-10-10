from typing import List, Optional, Tuple, Any

import pandas as pd
from quart import current_app
from sklearn.cluster import DBSCAN

from common.data.projection.model import ModelView
from common.data.model import Metric


def _calculate_thresholds(
        metrics_df: pd.DataFrame,
        metrics: list[Metric],
        preferences: dict[Metric, float]
) -> Tuple[dict[Metric, Any], dict[Metric, Any]]:
    """
    Calculate the acceptable and nearly acceptable thresholds for each metric.

    Parameters:
    metrics_df (pd.DataFrame): DataFrame containing metric values.
    metrics (list[Metric]): List of metrics.
    preferences (dict[Metric, float]): Dictionary with performance preferences (tolerance factors per metric).

    Returns:
    A tuple containing:
        - A dictionary mapping metrics to acceptable thresholds.
        - A dictionary mapping metrics to nearly acceptable thresholds.
    """
    max_values = metrics_df.max()
    min_values = metrics_df.min()

    thresholds_acc = {}
    thresholds_nacc = {}

    for metric in metrics:
        if metric.optimization_goal == 'maximize':
            thresholds_acc[metric] = max_values[metric] * (1 - preferences[metric])
            thresholds_nacc[metric] = max_values[metric] * (1 - (preferences[metric] * 2))
        elif metric.optimization_goal == 'minimize':
            thresholds_acc[metric] = min_values[metric] / (1 - preferences[metric])
            thresholds_nacc[metric] = min_values[metric] / (1 - (preferences[metric] * 2)) if (preferences[metric] * 2) < 1 else float('inf')

    return thresholds_acc, thresholds_nacc

def _calculate_inside_cluster_distrust_points(inside_ratio: float) -> int:
    """
    Calculate distrust points based on the inside ratio.

    Parameters:
    inside_ratio (float): Inside ratio.

    Returns:
    int: Distrust points.
    """
    if inside_ratio == 1:
        return 0
    elif 0.5 <= inside_ratio < 1:
        return 1
    elif 0 < inside_ratio < 0.5:
        return 2
    else:
        return 3

def _calculate_metrics_distrust_points(used_metric_ratio: float) -> int:
    """
    Calculate distrust points based on the ratio of used metrics.

    Parameters:
    used_metric_ratio (float): Ratio of used metrics to requested metrics.

    Returns:
    int: Distrust points.
    """
    if used_metric_ratio == 1:
        return 0
    elif 0.67 <= used_metric_ratio < 1:
        return 1
    elif 0.33 < used_metric_ratio < 0.67:
        return 2
    else:
        return 3

def _compute_cluster_fit(
        cluster_labels: list,
        metrics_df: pd.DataFrame,
        metrics: list[Metric], condition_fn
) -> dict:
    """
    Compute the cluster fitness for each cluster in metrics_df['dbscan'].

    For each metric, the condition_fn returns whether the metric value of a model in the cluster
    meets the condition of the region. The overall cluster fitness is the average of the mean values
    of the conditions for all metrics.

    Parameters:
        df: DataFrame containing metric values and a 'dbscan' column.
        metrics: List of metric names.
        condition_fn: Function taking (df, metric) and returning a boolean Series.

    Returns:
        A dictionary mapping cluster labels to fitness values.
    """
    cluster_fit = {}
    for label in cluster_labels:
        cluster_data = metrics_df.loc[metrics_df['dbscan'] == label]
        fits = {}
        for metric in metrics:
            condition = condition_fn(cluster_data, metric)
            fits[metric] = condition.mean()

        cluster_fit[label] = sum(fits.values()) / len(fits) if len(fits) > 0 else 0
    return cluster_fit

def _filter_metrics_df(
        metrics_df: pd.DataFrame,
        preferences: dict[Metric, Any]
) -> Tuple[pd.DataFrame, List[Metric]]:
    """
    Filter the metrics DataFrame based on performance preferences.

    - Selects only the metrics specified in preferences.
    - Drops metrics with more than 50% missing values.
    - Drops rows with missing values.

    Returns:
        A tuple containing:
            - The filtered DataFrame.
            - The updated list of models.
            - The list of retained metric names.
    """
    # Use only the metrics specified in the preferences
    metrics = list(preferences.keys())
    metrics_df = metrics_df[metrics].copy()

    # Drop metrics that have more than 50% missing values
    for metric in metrics:
        missing_ratio = metrics_df[metric].isna().mean()
        if missing_ratio > 0.5:
            current_app.logger.warning(
                f"Metric {metric} has {missing_ratio*100:.1f}% missing values. Dropping it.")
            metrics_df.drop(metric, axis=1, inplace=True)
    metrics = [Metric(metric) for metric in list(metrics_df.columns)]

    # Drop rows with missing values
    metrics_df.dropna(inplace=True)
    return metrics_df, metrics

def cluster_models(
        selected_models: List[ModelView],
        preferences: dict[Metric, float]
) -> Tuple[List[ModelView], List[ModelView], int, int, Optional[int]]:
    """
    Cluster models using DBSCAN and classify them into "acceptable" and "nearly acceptable" groups
    based on user performance preferences.

    Parameters:
        selected_models (List[ModelView]): List of selected models.
        preferences (Dict[str, Any]): Dictionary with performance preferences (tolerance factors per metric).

    Returns:
        A tuple containing:
            - List of acceptable models.
            - List of nearly acceptable models.
            - Distrust points for the requested metrics.
            - Distrust points for the acceptable region.
            - Distrust points for the nearly acceptable region (or None).
    """
    majority_ratio = 0.51

    raw_metrics_df = pd.DataFrame([model.metrics for model in selected_models])
    metrics_df, metrics = _filter_metrics_df(raw_metrics_df, preferences)
    used_metric_ratio = len(metrics) / len(preferences)
    distrust_pts_metrics = _calculate_metrics_distrust_points(used_metric_ratio)

    dbscan = DBSCAN(eps=0.05, min_samples=3, algorithm='kd_tree')
    metrics_df['dbscan'] = dbscan.fit_predict(metrics_df)

    thresholds_acc, thresholds_nacc = _calculate_thresholds(metrics_df, metrics, preferences)

    # Retrieve distinct cluster labels and remove the noise label (-1) if present
    distinct_labels = sorted(metrics_df['dbscan'].unique())
    if -1 in distinct_labels:
        distinct_labels.remove(-1)

    if len(distinct_labels) == 0:
        return [], [], distrust_pts_metrics, 0, 0

    # Compute cluster fitness for the Acceptable region (threshold <= value)
    cluster_fit_acc = _compute_cluster_fit(
        distinct_labels,
        metrics_df,
        metrics,
        lambda cluster_data, metric: (thresholds_acc[metric] <= cluster_data[metric] if metric.optimization_goal == 'maximize' else thresholds_acc[metric] >= cluster_data[metric])
    )
    clusters_acc = {label: fit for label, fit in cluster_fit_acc.items() if fit >= majority_ratio}

    # Calculate the inside ratio for acceptable clusters (fraction of clusters with perfect fit)
    inside_ratio_acc = sum(1 for fit in clusters_acc.values() if fit == 1) / len(clusters_acc) if len(clusters_acc) > 0 else 0
    distrust_pts_acc = _calculate_inside_cluster_distrust_points(inside_ratio_acc)

    # Compute cluster fitness for the Nearly Acceptable region:
    # condition: value is between the nearly acceptable threshold and the acceptable threshold.
    cluster_fit_nacc = _compute_cluster_fit(
        distinct_labels,
        metrics_df,
        metrics,
        lambda cluster_data, metric: ((thresholds_nacc[metric] <= cluster_data[metric]) & (cluster_data[metric] < thresholds_acc[metric]) if metric.optimization_goal == 'maximize' else (thresholds_acc[metric] < cluster_data[metric]) & (cluster_data[metric] <= thresholds_nacc[metric]))
    )
    clusters_nacc = {label: fit for label, fit in cluster_fit_nacc.items() if fit >= majority_ratio}

    if len(clusters_nacc) > 0:
        inside_ratio_nacc = sum(1 for fit in clusters_nacc.values() if fit == 1) / len(clusters_nacc)
        distrust_pts_nacc = _calculate_inside_cluster_distrust_points(inside_ratio_nacc)
    else:
        distrust_pts_nacc = None

    # Assign models to acceptable or nearly acceptable groups based on the cluster label.
    acceptable_models: List[ModelView] = []
    nearly_acceptable_models: List[ModelView] = []
    for idx, row in metrics_df.iterrows():
        label = row['dbscan']
        if label in clusters_acc.keys():
            acceptable_models.append(selected_models[idx])
        elif label in clusters_nacc.keys():
            nearly_acceptable_models.append(selected_models[idx])

    return acceptable_models, nearly_acceptable_models, distrust_pts_metrics, distrust_pts_acc, distrust_pts_nacc
