import asyncio
import logging

from flash import Flash
from quart import g

from assistml_dashboard.client import BackendClient
from assistml_dashboard.components import create_layout, register_callbacks
from common.data import ObjectDocumentMapper
from common.utils.document_cache import DocumentCache
from config import Config

logging.basicConfig(level=logging.getLevelName(Config.LOG_LEVEL), format='%(asctime)s - %(levelname)s - %(message)s')

external_stylesheets = ["https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
                        "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                        "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                        "https://codepen.io/bcd/pen/KQrXdb.css",
                        "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                        "https://codepen.io/dmcomfort/pen/JzdzEZ.css"]

def create_app():
    app = Flash(__name__, external_stylesheets=external_stylesheets)
    odm = ObjectDocumentMapper()
    app.server.config.from_object(Config)

    async def _initialize_app():
        await odm.connect()

        async with app.server.app_context():
            g.backend_client = BackendClient(app.server.config)
            g.document_cache = DocumentCache()
            register_callbacks(app)

        app.layout = await create_layout()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_initialize_app())

    return app
