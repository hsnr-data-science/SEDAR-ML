import json

from quart import request, jsonify

from assistml.api import bp
from assistml.data_profiler import profile_dataset
from common.dto import AnalyseDatasetRequestDto, AnalyseDatasetResponseDto


@bp.route('/analyse-dataset', methods=['POST'])
async def analyse_dataset():
    form = await request.form
    data = form.get("json")
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        data = json.loads(data)
        request_payload = AnalyseDatasetRequestDto(**data)
    except json.JSONDecodeError as e:
        return jsonify({"error": str(e)}), 400

    files = await request.files
    file = files.get("file")
    if file is None:
        return jsonify({"error": "No file part"}), 400

    dataset_profile, db_write_status = await profile_dataset(request_payload, file)

    response = AnalyseDatasetResponseDto(
        data_profile=dataset_profile if dataset_profile else None,
        db_write_status=db_write_status
    )

    return jsonify(response.model_dump(by_alias=True, mode="json"))
