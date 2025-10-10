import time

from quart import current_app

from assistml.model_recommender.select.aggregation_pipelines import calculate_dataset_similarity, \
    clear_dataset_similarity_context, clear_similar_models_context, get_similar_models
from common.data import Dataset, Query
from common.data.projection.model import ModelView

TOLERANCES = {"feature_ratio": 0.1, "monotonous_filtering": 0.1, "mutual_info": 0.1, "similarity_ratio": 0.5}


async def select_models_on_dataset_similarity(query: Query) -> tuple[list[ModelView], int]:
    current_app.logger.info(f"Task-Type in Query: {query.task_type}")
    new_dataset: Dataset = await query.dataset.fetch()
    current_app.logger.info(f"Task-Type in Query: {query.task_type}")
    if not new_dataset:
        raise ValueError("Dataset not found")

    current_app.logger.info("Selecting models based on dataset similarity...")
    current_app.logger.info("Calculating similarity context...")
    start_time = time.time()
    resp = await calculate_dataset_similarity(query.id, new_dataset, TOLERANCES["feature_ratio"],
                                       TOLERANCES["monotonous_filtering"], TOLERANCES["mutual_info"],
                                       TOLERANCES["similarity_ratio"])
    current_app.logger.info(f"Response: {resp}")
    context_built_time = time.time()
    current_app.logger.info("Calculated similarity context took {} seconds".format(context_built_time - start_time))

    lowest_sim_level = 0 if current_app.config["INCLUDE_SIMILARITY_LEVEL_0"] else 1

    for similarity_level in range(3, lowest_sim_level-1, -1):
        sim_start_time = time.time()
        current_app.logger.info(f"Trying to find models with similarity level {similarity_level}...")
        models = await get_similar_models(query.id, query.task_type, similarity_level)
        sim_end_time = time.time()

        if len(models) > 0:
            current_app.logger.info(f"Found {len(models)} models with similarity level {similarity_level} in {sim_end_time - sim_start_time} seconds")
            current_app.logger.info("Total time for selecting models based on dataset similarity: {} seconds".format(sim_end_time - start_time))
            await clear_dataset_similarity_context(query.id)
            await clear_similar_models_context(query.id)
            return models, similarity_level

    current_app.logger.info("No models were found")
    if not current_app.config["INCLUDE_SIMILARITY_LEVEL_0"]:
        current_app.logger.warn("You might want to consider including similarity level 0, but be aware that it might take a long time and consume a lot of resources")
        raise ValueError("No models found with similarity level 3, 2 or 1")
    raise ValueError("No models found with similarity level 1, skipping similarity level 0 due to performance reasons")
