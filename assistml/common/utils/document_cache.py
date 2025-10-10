import asyncio
from collections import defaultdict
from typing import DefaultDict, Dict, Union, TypeVar, Type

from beanie import PydanticObjectId, Link, Document

from common.data import Dataset, Task, Implementation

T = TypeVar("T", bound=Document)

class DocumentCache:
    _cache: Dict[Type[T], Dict[PydanticObjectId, T]]
    _locks: Dict[Type[T], DefaultDict[PydanticObjectId, asyncio.Lock]]

    def __init__(self):
        self._cache = {}
        self._locks = {}

    async def _get_document(self, document_type: Type[T], reference: Union[Link[T], PydanticObjectId, T]) -> T:
        if document_type not in self._cache:
            self._cache[document_type] = {}
            self._locks[document_type] = defaultdict(asyncio.Lock)

        cache = self._cache[document_type]

        if isinstance(reference, document_type):
            document = reference
            document_id = document.id
            lock = self._locks[document_type][document_id]
            async with lock:
                if document_id not in cache:
                    cache[document_id] = document

        elif isinstance(reference, Link):
            document_id = reference.to_ref().id
            lock = self._locks[document_type][document_id]
            async with lock:
                if document_id not in cache:
                    document = await reference.fetch()
                    cache[document_id] = document
            document = cache[document_id]

        elif isinstance(reference, PydanticObjectId):
            document_id = reference
            lock = self._locks[document_type][document_id]
            async with lock:
                if reference not in cache:
                    document = await document_type.get(document_id)
                    cache[document_id] = document
            document = cache[document_id]

        else:
            raise ValueError(f"Unknown document type: {type(reference)}")

        return document

    async def get_dataset(self, dataset: Union[Link[Dataset], PydanticObjectId, Dataset]) -> Dataset:
        return await self._get_document(Dataset, dataset)

    async def get_implementation(self, implementation: Union[Link[Implementation], PydanticObjectId, Implementation], cache_components=True) -> Implementation:
        implementation = await self._get_document(Implementation, implementation)

        if cache_components and implementation.components:
            for component in implementation.components.values():
                await self.get_implementation(component, cache_components)

        return implementation

    async def get_task(self, task: Union[Link[Task], PydanticObjectId, Task]) -> Task:
        return await self._get_document(Task, task)
