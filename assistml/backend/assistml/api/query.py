from pydantic import ValidationError
from quart import jsonify, request

from assistml.api import bp
from assistml.model_recommender import generate_report
from common.dto import ReportRequestDto, ReportResponseDto


@bp.route('/query', methods=['POST'])
async def query():
    """
        ---
        post:
          summary: AssistML analysis for new data
          description: Recommends ML models for a given query based on a base of known trained models.
          parameters:
            - in: body
              name: body
              required: true
              schema:
                $ref: '#/definitions/ReportRequestDto'
        """
    try:
        data = await request.get_json()
        report_request = ReportRequestDto(**data)
    except ValidationError as e:
        return jsonify({"error": f"Invalid request payload: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 400

    response: ReportResponseDto = await generate_report(report_request)

    return jsonify(response.model_dump(by_alias=True, mode="json"))
