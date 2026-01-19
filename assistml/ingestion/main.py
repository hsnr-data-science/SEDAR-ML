import asyncio

from common.data import ObjectDocumentMapper
from common.data.task import TaskType
from mlsea import mlsea_repository as mlsea
from processing.dataset import process_all_datasets
from processing.types import ProcessingOptions

POSSIBLE_ENTITIES = ['dataset', 'task', 'run']

def process_initial_offset(entity: str, entity_id: int):
    offset_ids = {entity: entity_id}
    if 'run' in offset_ids:
        task_id = mlsea.retrieve_task_id_for_run_id(offset_ids['run'])
        print(f"run {offset_ids['run']} belongs to task {task_id}")
        offset_ids['task'] = task_id
    if 'task' in offset_ids:
        dataset_id = mlsea.retrieve_dataset_id_for_task_id(offset_ids['task'])
        print(f"task {offset_ids['task']} belongs to dataset {dataset_id}")
        offset_ids['dataset'] = dataset_id

    if 'run' in offset_ids:
        offset_ids['task'] -= 1
    if 'task' in offset_ids:
        offset_ids['dataset'] -= 1
    return offset_ids

async def main(initial_offset=None, head=None, task_type=None):
    options = ProcessingOptions(recursive=True)

    # Input validation
    if initial_offset is not None:
        entity, entity_id = initial_offset.split(':')
        if entity not in POSSIBLE_ENTITIES:
            raise ValueError(f'Invalid entity: {entity}. Possible entities: {", ".join(POSSIBLE_ENTITIES)}')
        options.offset = process_initial_offset(entity, int(entity_id))

    if head is not None:
        if isinstance(head, str) and not head.isdigit():
            raise ValueError(f'Invalid head: {head}. Must be a number.')
        options.head = int(head)

    if task_type is not None:
        try:
            options.task_type = TaskType(task_type)
        except ValueError as e:
            print(e)
            raise ValueError(f'Invalid task type: {task_type}. Possible task types: {", ".join([tt.value for tt in TaskType])}')

    # Initialize Database connection
    odm = ObjectDocumentMapper()
    await odm.connect()

    # Start processing
    await process_all_datasets(options=options)


if __name__ == '__main__':
    asyncio.run(main())
