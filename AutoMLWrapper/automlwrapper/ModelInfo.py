from typing import Any, Dict, Optional, List
from PIL import Image
from .PyfuncModel import PyfuncModel
class ModelInfo:
    #---------------------------------------------------------------------------------------------#
    def __init__(
        self, 
        model_name: str,
        model_library: str,
        model_object: PyfuncModel,
        model_params_dict: Dict[str, Any],
        model_metrics_dict: Dict[str, Any],
        model_tags_dict: Dict[str, str] = {},
        model_path: Optional[str] = None,
        model_imgs_as_pil_dict: Optional[Dict[str, Image.Image]] = {},
        model_files_as_path_list: Optional[List[str]] = {}
    ):
        self.model_name = model_name
        self.model_library = model_library
        self.model_object = model_object
        self.model_path = model_path
        self.model_params_dict = model_params_dict
        self.model_metrics_dict = model_metrics_dict
        self.model_tags_dict = model_tags_dict
        self.model_images_dict = model_imgs_as_pil_dict
        self.model_files_as_path_list = model_files_as_path_list
    
    #---------------------------------------------------------------------------------------------#
    def to_dict(self):
        return {
            'tags': {
                'model_name': self.model_name,
                'model_library': self.model_library,
                **(self.model_tags_dict or {})
            },
            'params': self.model_params_dict,
            'metrics': self.model_metrics_dict,
            'model': self.model_object,
            **({'model_path': self.model_path} if self.model_path is not None else {}),
            'images': self.model_images_dict,
            'files': self.model_files_as_path_list

        }