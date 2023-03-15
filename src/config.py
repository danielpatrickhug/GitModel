import yaml


class Config:
    def __init__(self, pipeline_settings, semantic_graph_context_generator, gnn_heads, topic_model):
        self.semantic_graph_context_generator = semantic_graph_context_generator
        self.gnn_heads = gnn_heads
        self.topic_model = topic_model
        self.pipeline_settings = pipeline_settings

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class PipelineSettings:
    def __init__(self, config: Config):
        self.config = config

    def __repr__(self) -> str:
        return f"PipelineSettings(config={self.config})"
