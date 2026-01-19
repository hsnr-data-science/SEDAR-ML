import asyncio

from hypercorn.asyncio import serve
from hypercorn.config import Config

from assistml import create_app

app = create_app()

if __name__ == "__main__":
    #app.run(host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'])
    config = Config()
    config.bind = [f"{app.config['HOST']}:{app.config['PORT']}"]
    config.use_reloader = app.config['DEBUG']
    config.debug = app.config['DEBUG']

    asyncio.run(serve(app, config))
