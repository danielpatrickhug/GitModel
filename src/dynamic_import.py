from copy import deepcopy
from typing import Dict

from src.config import Config, PipelineSettings
from src.ml_models.graph_networks.gnn_head import GNNHead
from src.ml_models.topic_modeling.topic_model import TopicModel
from src.ml_models.transformers.semantic_graph_context_generator import SemanticGraphContextGenerator


def instantiate_class_from_config(config: Config) -> Dict[str, list]:
    config = deepcopy(config)

    obj_map = {}
    for component_name in ["pipeline_settings", "semantic_graph_context_generator", "gnn_heads", "topic_model"]:
        if component_name not in obj_map:
            obj_map[component_name] = []

        for args in getattr(config, component_name):
            impl = args.pop("__impl__")
            try:
                _cls = {
                    "PipelineSettings": PipelineSettings,
                    "SemanticGraphContextGenerator": SemanticGraphContextGenerator,
                    "GNNHead": GNNHead,
                    "TopicModel": TopicModel,
                }[impl]
            except KeyError:
                raise Exception(f"{impl} cannot be found in module my_module")
            obj_map[component_name].append(_cls(config=args))

    # Print the recovered dictionary
    return obj_map
