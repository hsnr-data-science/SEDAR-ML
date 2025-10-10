import asyncio

from quart import Quart, jsonify
from config import Config
from common.data import ObjectDocumentMapper


def create_app(config_class=Config):
    app = Quart(__name__)
    odm = ObjectDocumentMapper()
    app.config.from_object(config_class)

    @app.before_serving
    async def connect_db():
        await odm.connect()

    #asyncio.run(async_init())

    from assistml.api import bp
    app.register_blueprint(bp)

    @app.route('/<path:any_other_path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
    def block_other_paths(any_other_path):
        """
        Blocksing access to any other path than the ones defined in the API.
        """
        return jsonify({"error": "Access to this endpoint is not allowed"}), 403

    return app
