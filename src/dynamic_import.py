from copy import deepcopy
from typing import Dict

from src.config import Config
from src.ml_models import GNNHead, SemanticGraphContextGenerator, TopicModel


def instantiate_class_from_config(config: Config) -> Dict[str, list]:
    config = deepcopy(config)

    obj_map = {}
    for component_name in ["semantic_graph_context_generator", "gnn_heads", "topic_model"]:
        if component_name not in obj_map:
            obj_map[component_name] = []

        for args in getattr(config, component_name):
            impl = args.pop("__impl__")
            try:
                _cls = {
                    "SemanticGraphContextGenerator": SemanticGraphContextGenerator,
                    "GNNHead": GNNHead,
                    "TopicModel": TopicModel,
                }[impl]
            except KeyError:
                raise Exception(f"{impl} cannot be found in module my_module")
            obj_map[component_name].append(_cls(config=args))
    # Create a list of SemanticGraphContextGenerator objects
    sgcg_list = obj_map["semantic_graph_context_generator"]

    # Replace the SemanticGraphContextGenerator objects in obj_map with the list
    obj_map["semantic_graph_context_generator"] = sgcg_list
    dict_from_obj_map = {}
    for key, value in obj_map.items():
        dict_from_obj_map[key] = value

    # Print the recovered dictionary
    return dict_from_obj_map
