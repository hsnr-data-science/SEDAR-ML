from beanie.odm.utils.encoder import Encoder
from pydantic import BaseModel
from pydantic.alias_generators import to_camel


def alias_generator(field_name: str) -> str:
    if field_name == "id":
        return "_id"

    if field_name.startswith("_"):
        return field_name

    if field_name in ["revision_id"]:
        return field_name

    return to_camel(field_name)

def encode_dict(d: dict) -> dict:
    """
    Encode a dictionary. By default, Beanie can't handle dictionaries with non-string keys.
    :param d:
    :return:
    """
    result: dict = {}
    encoder = Encoder()
    for k, v in d.items():
        encoded_key = encoder.encode(k)
        if not isinstance(encoded_key, str):
            raise ValueError(f"Key {k} is not a string")
        encoded_value = encoder.encode(v)
        result[encoded_key] = encoded_value
    return result

class CustomBaseModel(BaseModel):

    class Config:
        ser_json_inf_nan = 'constants'
        populate_by_name = True
        alias_generator = alias_generator
