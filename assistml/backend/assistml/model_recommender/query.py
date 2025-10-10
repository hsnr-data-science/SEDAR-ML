import time

from beanie import Link
from quart import current_app

from common.data import Dataset, Query
from common.dto import ReportRequestDto


async def handle_query(request: ReportRequestDto) -> Query:
    """
    Handle a query request.
    """
    issued_at = time.strftime('%Y%m%d-%H%M')

    current_app.logger.info("Forming query record with fields...")
    current_app.logger.info(
        f"Query issued at {issued_at} for task {getattr(request, 'task_type', 'unknown')} (target_type: {getattr(request, 'classification_type', 'n/a')})"
    )

    dataset = await Dataset.get(request.dataset_id)
    if not dataset:
        raise ValueError(f"Dataset not found")

    query = Query(
        made_at=issued_at,
        dataset=Link(dataset.to_ref(), Dataset),
        **request.model_dump()
    )
    await query.insert()
    return query
