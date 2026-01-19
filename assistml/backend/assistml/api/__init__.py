from quart import Blueprint

bp = Blueprint('api', __name__)


from assistml.api import query, analyse_dataset
