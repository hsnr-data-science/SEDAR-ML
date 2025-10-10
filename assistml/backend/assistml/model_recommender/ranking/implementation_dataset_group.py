from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

from beanie import Link
from bson import DBRef

from assistml.model_recommender.ranking.hyperparameter_analytics import HyperparameterAnalytics
from assistml.model_recommender.ranking.hyperparameter_configuration import HyperparameterConfiguration
from assistml.model_recommender.ranking.metric_analytics import DescriptiveStatistics, MetricAnalytics
from common.data import Dataset, Implementation
from common.data.model import Metric
from common.data.projection.model import ModelView
from common.data.query import HyperparameterConfigurationReport, ImplementationDatasetGroupReport, \
    PartialHyperparameterConfiguration, PerformanceReport, Query
from common.utils.dataset_descriptor_normalizer import DatasetDescriptorNormalizer
from common.utils.document_cache import DocumentCache


class ImplementationDatasetGroup:

    _immutable: bool
    _implementation: Implementation
    _dataset: Dataset
    _models: List[Tuple[ModelView, HyperparameterConfiguration]]
    _models_grouped_by_configuration: Optional[
        DefaultDict[HyperparameterConfiguration, List[Tuple[ModelView, HyperparameterConfiguration]]]]
    _document_cache: DocumentCache
    _metric_analytics: MetricAnalytics
    _hyperparameter_analytics: HyperparameterAnalytics
    _dataset_descriptor_normalizer: DatasetDescriptorNormalizer
    _aggregated_metrics_by_configuration: Optional[Dict[HyperparameterConfiguration, Dict[Metric, DescriptiveStatistics]]]
    _aggregated_metrics: Optional[Dict[Metric, Any]]
    _ranked_configurations: Optional[List[Tuple[float, HyperparameterConfiguration]]]

    def __init__(
            self,
            implementation: Implementation,
            dataset: Dataset,
            document_cache: DocumentCache,
            metric_analytics: MetricAnalytics,
            hyperparameter_analytics: HyperparameterAnalytics,
            dataset_descriptor_normalizer: DatasetDescriptorNormalizer
    ):
        self._immutable = False
        self._implementation = implementation
        self._dataset = dataset
        self._models = []
        self._models_grouped_by_configuration = None
        self._document_cache = document_cache
        self._metric_analytics = metric_analytics
        self._hyperparameter_analytics = hyperparameter_analytics
        self._dataset_descriptor_normalizer = dataset_descriptor_normalizer
        self._aggregated_metrics_by_configuration = None
        self._ranked_configurations = None

    @classmethod
    async def create(
            cls,
            implementation: Implementation,
            dataset_ref: Union[Link[Dataset], Dataset],
            document_cache: DocumentCache,
            metric_analytics: MetricAnalytics,
            hyperparameter_analytics: HyperparameterAnalytics,
            dataset_descriptor_normalizer: DatasetDescriptorNormalizer
    ) -> "ImplementationDatasetGroup":
        dataset = await document_cache.get_dataset(dataset_ref)
        dataset_descriptor_normalizer.add_dataset(dataset)
        return cls(implementation, dataset, document_cache, metric_analytics, hyperparameter_analytics, dataset_descriptor_normalizer)

    async def add_model(self, model: ModelView) -> None:
        if self._immutable:
            raise ValueError("Group is immutable")
        configuration = await HyperparameterConfiguration.from_setup(
            model.setup, self._document_cache, self._hyperparameter_analytics)
        self._metric_analytics.add_metric_values(model.metrics)
        await self._hyperparameter_analytics.add_configuration(configuration)
        self._models.append((model, configuration))

    def _group_models_by_hyperparameters(self) -> None:
        if not self._hyperparameter_analytics.are_standardizers_fitted():
            raise ValueError("Standardizers are not fitted. Call fit_standardizers() first.")
        self._models_grouped_by_configuration = defaultdict(list)
        for (model, configuration) in self._models:
            self._models_grouped_by_configuration[configuration].append((model, configuration))
        self._immutable = True

    def _aggregate_metrics(self) -> None:
        if self._models_grouped_by_configuration is None:
            self._group_models_by_hyperparameters()

        # aggregate metrics for each configuration group
        self._aggregated_metrics_by_configuration: Dict[HyperparameterConfiguration, Dict[Metric, Dict[str, float]]] = {}
        for configuration, models in self._models_grouped_by_configuration.items():
            raw_metrics = [model.metrics for model, _ in models]
            self._aggregated_metrics_by_configuration[configuration] = self._metric_analytics.aggregate_list(raw_metrics)

    def _rank_configuration_groups(
            self,
            selected_metrics: List[Metric],
            lambda_penalty: float = 0.5
    ) -> None:
        if any(metric.datatype not in [float, int] for metric in selected_metrics):
            raise ValueError("Unsupported metric datatype")
        if self._aggregated_metrics_by_configuration is None:
            self._aggregate_metrics()
        metric_weights = {metric: 1.0 if metric in selected_metrics else 0.0 for metric in Metric}
        ranked_groups = []
        for configuration, metrics in self._aggregated_metrics_by_configuration.items():
            score_vector = self._metric_analytics.calculate_overall_score(metrics, metric_weights)
            score = score_vector['mean'] - lambda_penalty * score_vector['std']
            ranked_groups.append((score, configuration))
        ranked_groups.sort(key=lambda x: x[0], reverse=True)
        self._ranked_configurations = ranked_groups

    def get_best_configuration(self, selected_metrics: List[Metric]) -> HyperparameterConfiguration:
        if self._ranked_configurations is None:
            self._rank_configuration_groups(selected_metrics)
        return self._ranked_configurations[0][1]

    def get_top_m_configurations(self, selected_metrics: List[Metric], m: int) -> List[HyperparameterConfiguration]:
        if self._ranked_configurations is None:
            self._rank_configuration_groups(selected_metrics)
        return [configuration for _, configuration in self._ranked_configurations[:m]]

    def get_metrics_of_best_configuration(self, selected_metrics: List[Metric]) -> Dict[Metric, Dict[str, float]]:
        if self._ranked_configurations is None:
            self._rank_configuration_groups(selected_metrics)
        return self._aggregated_metrics_by_configuration[self._ranked_configurations[0][1]]

    def get_dataset_similarity(self, dataset: Dataset) -> float:
        return self._dataset.similarity(dataset, self._dataset_descriptor_normalizer)

    async def generate_report(self, query: Query, top_m) -> ImplementationDatasetGroupReport:
        query_dataset = await self._document_cache.get_dataset(query.dataset)
        selected_metrics = list(query.preferences.keys())
        configurations = [
            HyperparameterConfigurationReport(
                hyperparameters=[
                    PartialHyperparameterConfiguration(
                        implementation=Link(
                            DBRef(Implementation.get_collection_name(), implementation_id), Implementation
                            ),
                        hyperparameters=hyperparameters
                    ) for implementation_id, hyperparameters in configuration.get_representational_configuration().items()
                ],
                performance={
                    metric: PerformanceReport(
                        quantile_label=self._metric_analytics.get_label(metric, normalized_value=metric_values['mean']),
                        normalized_mean=metric_values['mean'],
                        normalized_std=metric_values['std'],
                        mean=self._metric_analytics.denormalize_metric_value(metric, metric_values['mean']),
                        std=self._metric_analytics.denormalize_metric_std(metric, metric_values['std'])
                    ) for metric, metric_values in self._aggregated_metrics_by_configuration[configuration].items()
                }
            )
            for configuration in self.get_top_m_configurations(selected_metrics, top_m)
        ]
        report = ImplementationDatasetGroupReport(
            dataset=Link(self._dataset.to_ref(), Dataset),
            dataset_name=self._dataset.info.dataset_name,
            dataset_similarity=self.get_dataset_similarity(query_dataset),
            dataset_features=self._dataset.info.nr_total_features,
            dataset_observations=self._dataset.info.observations,
            model_count=len(self._models),
            configurations=configurations
        )
        return report

    def __repr__(self) -> str:
        return f"ImplementationDatasetGroup(implementation={self._implementation.title}, dataset={self._dataset.info.dataset_name}, models={len(self._models)})"

    def __str__(self) -> str:
        return f"ImplementationDatasetGroup: {self._implementation.title}, dataset={self._dataset.info.dataset_name}, #models={len(self._models)}"
