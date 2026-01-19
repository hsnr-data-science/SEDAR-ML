from flash import Flash

from assistml_dashboard.components.sidebar.sidebar_callbacks import register_sidebar_callbacks
from assistml_dashboard.components.content_callbacks import register_content_callbacks


def register_callbacks(app: Flash):
    register_sidebar_callbacks(app)
    register_content_callbacks(app)
