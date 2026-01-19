from email.policy import default
from typing import Dict, Any, ForwardRef

from beanie import PydanticObjectId

from assistml.model_recommender.ranking.standardizer import Standardizer
from common.data import Implementation
from common.utils.document_cache import DocumentCache

HyperparameterConfiguration = ForwardRef("HyperparameterConfiguration")


class HyperparameterAnalytics:

    _implementation: Implementation
    _document_cache: DocumentCache
    _standardizers: Dict[PydanticObjectId, Dict[str, Standardizer]]
    _hyperparameter_values: Dict[PydanticObjectId, Dict[str, list]]
    _are_standardizers_fitted: bool

    def __init__(self, implementation: Implementation, document_cache: DocumentCache):
        self._implementation = implementation
        self._document_cache = document_cache
        self._standardizers = {}
        self._hyperparameter_values = {}
        self._are_standardizers_fitted = False

    async def _is_default_value(self, implementation_id: PydanticObjectId, hyperparameter_name: str, value: Any) -> bool:
        parameter_implementation = await self._document_cache.get_implementation(implementation_id)
        if hyperparameter_name not in parameter_implementation.parameters:
            raise ValueError(f"Hyperparameter {hyperparameter_name} not found in implementation {parameter_implementation.title}")
        implementation_parameter = parameter_implementation.parameters[hyperparameter_name]
        if implementation_parameter.type == "flag" and implementation_parameter.default_value is None:
            return False
        return value == implementation_parameter.default_value

    async def add_hyperparameter_value(self, implementation_id: PydanticObjectId, hyperparameter_name: str, value: Any):
        if await self._is_default_value(implementation_id, hyperparameter_name, value):
            return  # do not store default values
        if implementation_id not in self._hyperparameter_values:
            self._hyperparameter_values[implementation_id] = {}
            self._standardizers[implementation_id] = {}
        if hyperparameter_name not in self._hyperparameter_values[implementation_id]:
            self._hyperparameter_values[implementation_id][hyperparameter_name] = []
            self._standardizers[implementation_id][hyperparameter_name] = Standardizer()
        self._hyperparameter_values[implementation_id][hyperparameter_name].append(value)
        self._are_standardizers_fitted = False

    async def add_configuration(self, configuration: HyperparameterConfiguration):
        for implementation_id, hyperparameters in configuration.get_raw_configuration().items():
            for hyperparameter_name, value in hyperparameters.items():
                await self.add_hyperparameter_value(implementation_id, hyperparameter_name, value)

    def fit_standardizers(self):
        for implementation_id, hyperparameters in self._hyperparameter_values.items():
            for hyperparameter_name, values in hyperparameters.items():
                self._standardizers[implementation_id][hyperparameter_name].fit(values)
        self._are_standardizers_fitted = True

    def standardize_hyperparameter_value(self, implementation_id: PydanticObjectId, hyperparameter_name: str, value: Any) -> Any:
        if not self._are_standardizers_fitted:
            raise ValueError("Standardizers are not fitted. Call fit_standardizers() first.")
        return self._standardizers[implementation_id][hyperparameter_name].transform(value)

    def reverse_standardize_hyperparameter_value(self, implementation_id: PydanticObjectId, hyperparameter_name: str, value: Any) -> Any:
        if not self._are_standardizers_fitted:
            raise ValueError("Standardizers are not fitted. Call fit_standardizers() first.")
        return self._standardizers[implementation_id][hyperparameter_name].inverse_transform(value)

    def are_standardizers_fitted(self) -> bool:
        return self._are_standardizers_fitted
