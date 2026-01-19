from collections import OrderedDict
from typing import Any, Dict

from bson import ObjectId

from assistml.model_recommender.ranking.hyperparameter_analytics import HyperparameterAnalytics
from common.data.model import Setup
from common.utils.document_cache import DocumentCache


class HyperparameterConfiguration:
    _configuration: OrderedDict[ObjectId, OrderedDict[str, Any]]
    _hyperparameter_analytics: HyperparameterAnalytics

    def __init__(self, configuration: Dict[ObjectId, Dict[str, Any]], hyperparameter_analytics: HyperparameterAnalytics):
        self._configuration = HyperparameterConfiguration.order_configuration(configuration)
        self._hyperparameter_analytics = hyperparameter_analytics

    @classmethod
    async def from_setup(cls, setup: Setup, document_cache: DocumentCache, hyperparameter_analytics: HyperparameterAnalytics) -> "HyperparameterConfiguration":
        configuration: Dict[ObjectId, Dict[str, Any]] = {}
        for hyperparameter in setup.hyper_parameters:
            parameter_implementation = await document_cache.get_implementation(hyperparameter.implementation)
            if hyperparameter.name not in parameter_implementation.parameters:
                raise ValueError(f"Hyperparameter {hyperparameter.name} not found in implementation {parameter_implementation.title}")
            if hyperparameter.value is not None and hyperparameter.value == parameter_implementation.parameters[hyperparameter.name].default_value:
                continue  # do not store default values
            if parameter_implementation.id not in configuration:
                configuration[parameter_implementation.id] = {}
            if hyperparameter.data_type == "flag":
                configuration[parameter_implementation.id][hyperparameter.name] = True
            else:
                configuration[parameter_implementation.id][hyperparameter.name] = hyperparameter.value
        return cls(configuration, hyperparameter_analytics)

    @staticmethod
    def order_configuration(configuration: Dict[ObjectId, Dict[str, Any]]) -> OrderedDict[ObjectId, OrderedDict[str, Any]]:
        sorted_implementation_ids = sorted(configuration.keys())
        sorted_configuration = OrderedDict()
        for implementation_id in sorted_implementation_ids:
            sorted_configuration[implementation_id] = OrderedDict(sorted(configuration[implementation_id].items(), key=lambda x: x[0]))
        return sorted_configuration

    def get_standardized_configuration(self) -> OrderedDict[ObjectId, OrderedDict[str, Any]]:
        if not self._hyperparameter_analytics.are_standardizers_fitted():
            raise ValueError("Standardizers are not fitted. Call fit_standardizers() first.")
        standardized_configuration = OrderedDict()
        for implementation_id, hyperparameters in self._configuration.items():
            standardized_configuration[implementation_id] = OrderedDict()
            for name, value in hyperparameters.items():
                standardized_configuration[implementation_id][name] = self._hyperparameter_analytics.standardize_hyperparameter_value(implementation_id, name, value)
        return standardized_configuration

    def get_representational_configuration(self) -> OrderedDict[ObjectId, OrderedDict[str, Any]]:
        if not self._hyperparameter_analytics.are_standardizers_fitted():
            raise ValueError("Standardizers are not fitted. Call fit_standardizers() first.")
        representational_configuration = OrderedDict()
        for implementation_id, hyperparameters in self.get_standardized_configuration().items():
            representational_configuration[implementation_id] = OrderedDict()
            for name, value in hyperparameters.items():
                representational_configuration[implementation_id][name] = self._hyperparameter_analytics.reverse_standardize_hyperparameter_value(implementation_id, name, value)
        return representational_configuration

    def get_raw_configuration(self):
        return self._configuration

    def get_configuration(self):
        if self._hyperparameter_analytics.are_standardizers_fitted():
            return self.get_standardized_configuration()
        return self.get_raw_configuration()

    def __str__(self):
        if self._hyperparameter_analytics.are_standardizers_fitted():
            return f"HyperParameterConfiguration(hyperparameters={self.get_standardized_configuration()}, standardized=True)"
        return f"HyperParameterConfiguration(hyperparameters={self.get_raw_configuration()}, standardized=False)"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, HyperparameterConfiguration):
            return False
        self_fitted = self._hyperparameter_analytics.are_standardizers_fitted()
        other_fitted = other._hyperparameter_analytics.are_standardizers_fitted()
        if self_fitted != other_fitted:
            return False
        return self.get_configuration() == other.get_configuration()

    def __hash__(self):
        return hash(str(self))
