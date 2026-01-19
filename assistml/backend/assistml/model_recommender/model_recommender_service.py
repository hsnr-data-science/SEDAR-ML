import time

from beanie import WriteRules
from quart import current_app

from assistml.model_recommender.cluster import cluster_models
from assistml.model_recommender.query import handle_query
from assistml.model_recommender.ranking import Report
from assistml.model_recommender.ranking.report import DistrustPointCategory
from assistml.model_recommender.select import select_models_on_dataset_similarity
from common.dto import ReportRequestDto


async def generate_report(request: ReportRequestDto):
    """
    Generate a report based on the given request.
    """
    start_time = time.time()

    query = await handle_query(request)
    report = Report(query)

    models, similarity_level = await select_models_on_dataset_similarity(query)
    if len(models) == 0:
        raise ValueError("No models found")

    report.set_distrust_points(DistrustPointCategory.DATASET_SIMILARITY, 3-similarity_level)

    acceptable_models, nearly_acceptable_models, distrust_pts_metrics, distrust_pts_acc, distrust_pts_nacc = cluster_models(models, query.preferences)
    await report.set_models(acceptable_models, nearly_acceptable_models)
    report.set_distrust_points(DistrustPointCategory.METRICS_SUPPORT, distrust_pts_metrics)
    report.set_distrust_points(DistrustPointCategory.CLUSTER_INSIDE_RATIO_ACC, distrust_pts_acc)
    report.set_distrust_points(DistrustPointCategory.CLUSTER_INSIDE_RATIO_NACC, distrust_pts_nacc or 0)

    query.report = await report.generate_report()
    await query.save(link_rule=WriteRules.DO_NOTHING)

    end_time = time.time()
    time_taken = end_time - start_time
    current_app.logger.info(f"Time taken for end to end execution {time_taken}")
    return query.report
