from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from assistml.model_recommender.ranking.normalizer import Normalizer
from common.data.model import Metric

DEFAULT_METRIC_WEIGHTS = MappingProxyType({metric: 1.0 for metric in Metric})

class DescriptiveStatistics(TypedDict):
    mean: float
    std: float

class MetricAnalytics:

    _metric_values: Dict[Metric, List[Any]]
    _normalizers: Dict[Metric, Normalizer]

    def __init__(self):
        self._metric_values = {}
        self._normalizers = {}

    def add_metric_values(self, metrics: Dict[Metric, Any]):
        for metric, value in metrics.items():
            if metric not in self._metric_values:
                self._metric_values[metric] = []
                self._normalizers[metric] = Normalizer(metric.optimization_goal)
            self._metric_values[metric].append(value)

    def fit_normalizers(self):
        for metric, values in self._metric_values.items():
            self._normalizers[metric].fit(values)

    def normalize_metric_value(self, metric: Metric, value: Any) -> float:
        if metric not in self._normalizers:
            raise ValueError(f"Unknown metric: {metric}")
        return self._normalizers[metric].transform(value)

    def denormalize_metric_value(self, metric: Metric, normalized_value: float) -> Any:
        if metric not in self._normalizers:
            raise ValueError(f"Unknown metric: {metric}")
        return self._normalizers[metric].inverse_transform(normalized_value)

    def denormalize_metric_std(self, metric: Metric, normalized_std: float) -> float:
        if metric not in self._normalizers:
            raise ValueError(f"Unknown metric: {metric}")
        return self._normalizers[metric].inverse_transform_std(normalized_std)

    def get_label(self, metric: Metric, normalized_value: Optional[float] = None, raw_value: Optional[float] = None) -> str:
        if normalized_value is not None:
            value = self.denormalize_metric_value(metric, normalized_value)
        elif raw_value is not None:
            value = raw_value
        else:
            raise ValueError("Either normalized_value or raw_value must be provided")
        if metric not in self._normalizers:
            raise ValueError(f"Unknown metric: {metric}")
        return self._normalizers[metric].get_label(value)

    @staticmethod
    def _calculate_statistics(weighted_normalized_value_list: List[Tuple[float, float]]) -> DescriptiveStatistics:
        total_weight = sum(weight for weight, _ in weighted_normalized_value_list)
        mean_value = sum(weight * value for weight, value in weighted_normalized_value_list) / total_weight
        std_value = (sum(weight * (value - mean_value) ** 2 for weight, value in weighted_normalized_value_list) / total_weight) ** 0.5
        return {"mean": mean_value, "std": std_value}

    @staticmethod
    def _calculate_pooled_statistics(weighted_descriptive_list: List[Tuple[float, DescriptiveStatistics]]) -> DescriptiveStatistics:
        total_weight = sum(weight for weight, _ in weighted_descriptive_list)
        mean_value = sum(weight * value["mean"] for weight, value in weighted_descriptive_list) / total_weight
        std_value = (sum(weight * (value["mean"] - mean_value) ** 2 + weight * value["std"] ** 2 for weight, value in weighted_descriptive_list) / total_weight) ** 0.5
        return {"mean": mean_value, "std": std_value}

    def aggregate_list(
            self,
            raw_metric_list: Union[List[Dict[Metric, Any]], List[Tuple[float, Dict[Metric, Any]]]]
    ) -> Dict[Metric, DescriptiveStatistics]:
        if not raw_metric_list:
            raise ValueError("No metrics to aggregate")

        if isinstance(raw_metric_list[0], tuple):
            _weighted_raw_metric_list = raw_metric_list
        else:
            _weighted_raw_metric_list = [(1.0, metrics) for metrics in raw_metric_list]

        # check if values are scalar or descriptive statistics
        _sample_metrics = _weighted_raw_metric_list[0][1]
        _sample_metric_value = _sample_metrics[list(_sample_metrics.keys())[0]]
        if isinstance(_sample_metric_value, dict):
            _pooled = True
        elif isinstance(_sample_metric_value, (int, float)):
            _pooled = False
        else:
            raise ValueError(f"Unknown metric value type: {type(_sample_metric_value)}")

        normalized_metrics: Dict[Metric, List[Tuple[float, Union[Union[int, float], DescriptiveStatistics]]]] = {}
        for weight, metrics in _weighted_raw_metric_list:
            for metric, value in metrics.items():
                if metric not in normalized_metrics:
                    normalized_metrics[metric] = []
                metric_value = value if _pooled else self.normalize_metric_value(metric, value)
                normalized_metrics[metric].append((weight, metric_value))

        aggregated_metrics = {}
        for metric, weighted_values in normalized_metrics.items():
            if _pooled:
                aggregated_metrics[metric] = MetricAnalytics._calculate_pooled_statistics(weighted_values)
            else:
                aggregated_metrics[metric] = MetricAnalytics._calculate_statistics(weighted_values)
        return aggregated_metrics

    @staticmethod
    def calculate_overall_score(
            aggregated_metrics: Dict[Metric, DescriptiveStatistics],
            weights: Dict[Metric, float] = DEFAULT_METRIC_WEIGHTS
    ) -> DescriptiveStatistics:
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("Sum of weights is zero")
        weighted_mean = sum(weights[metric] * aggregated_metrics[metric]["mean"] for metric in aggregated_metrics if metric in weights) / total_weight
        weighted_std = sum(weights[metric] * aggregated_metrics[metric]["std"] for metric in aggregated_metrics if metric in weights) / total_weight
        return {"mean": weighted_mean, "std": weighted_std}
