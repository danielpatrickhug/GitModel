import json
import logging
import os
import random
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import Config
from src.dynamic_import import instantiate_class_from_config
from src.fetch_repo import clone_and_create_context_folder
from src.ml_models.graph_networks.gnn_head import combine_graphs_with_gat, linearly_sum_gnn_heads

lg = logging.getLogger(__name__)
lg.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s"))
lg.addHandler(handler)


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self._obj_map = instantiate_class_from_config(config)
        self.pipeline_settings = self._obj_map["pipeline_settings"][0]
        self.semantic_graph_context_generator = [
            self._obj_map["semantic_graph_context_generator"][i]
            for i in range(len(self._obj_map["semantic_graph_context_generator"]))
        ]
        self.gnn_heads = [self._obj_map["gnn_heads"][i] for i in range(len(self._obj_map["gnn_heads"]))]
        self.topic_model = self._obj_map["topic_model"][0]

    @classmethod
    def from_config(cls, config: Config):
        return cls(config)

    @classmethod
    def from_yaml(cls, yaml_path):
        config = Config.from_yaml(yaml_path)
        return cls(config)

    def find_files_with_substring(self, root_dir, substring):
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if substring in filename:
                    yield os.path.join(dirpath, filename)

    def load_jsonl(self, filepaths):
        res = []
        for filepath in filepaths:
            with open(filepath, "r") as f:
                for line in f:
                    res.append(json.loads(line))
        sents = []
        for r in res:
            messages = r["conversation_history"]
            reply = r["assistant_reply"]
            sents.append(reply)
            sents.append(messages[-2]["content"])
        data = pd.DataFrame(sents, columns=["query"])
        data["_id"] = data.index
        return data

    def run(self, git_repo: str, repo_name: str) -> Tuple[Tuple[np.ndarray, np.ndarray], dict]:
        """
        Run the pipeline."""
        # replace with lg.info
        lg.info("Running pipeline...")
        lg.info("Fetching repo...")
        repo_folder, context_folder = clone_and_create_context_folder(git_repo, repo_name)
        lg.info("Generating semantic graph context...")

        semantic_graph_context = [
            context_generator.decompose_repo(
                repo_folder,
                repo_name,
                context_folder,
                skip_graph_generation=self.pipeline_settings.config["skip_graph_creation"],
            )
            for context_generator in self.semantic_graph_context_generator
        ]
        context_files = []
        for context in semantic_graph_context[0]:
            context_files.append(context)
        context_files = self.find_files_with_substring(context_folder, repo_name)
        lg.info(context_files)
        data = self.load_jsonl(context_files)

        lg.info("Running GNN heads...")
        gnn_head_outputs = [gnn_head.generate_graph(data) for gnn_head in self.gnn_heads]
        lg.info("Combining GNN heads...")
        lg.info(self.pipeline_settings)

        if self.pipeline_settings.config["combine_gnn_strategy"] == "sum":
            combined_gnn_head = linearly_sum_gnn_heads(gnn_head_outputs, self.pipeline_settings.config["norm_fn"])
        elif self.pipeline_settings.config["combine_gnn_strategy"] == "gat":
            combined_gnn_head = combine_graphs_with_gat(gnn_head_outputs)
        elif self.pipeline_settings.config["combine_gnn_strategy"] == "none":
            combined_gnn_head = gnn_head_outputs[0]
        elif self.pipeline_settings.config["combine_gnn_strategy"] == "random":
            # Choose a random GNN head as the final output
            combined_gnn_head = random.choice(gnn_head_outputs)
        else:
            raise ValueError(f"Unknown combine_gnn_strategy setting: {self.pipeline_settings['combine_gnn_strategy']}")

        lg.info("Running topic model...")

        topic_model_outputs = self.topic_model.run(data, combined_gnn_head)

        return gnn_head_outputs, [topic_model_outputs]

    def get_repo_contents(self, git_repo_path):
        contents = self.semantic_graph_context_generator.get_repo_contents(git_repo_path)
        return contents

    def decompose_repo(self, contents, git_repo_path, name_id, topic_tree, out_path):
        self.semantic_graph_context_generator.decompose_repo(contents, git_repo_path, name_id, topic_tree, out_path)
