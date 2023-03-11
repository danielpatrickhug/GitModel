import json
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import Config
from src.dynamic_import import instantiate_class_from_config
from src.fetch_repo import clone_and_create_context_folder

lg = logging.getLogger(__name__)
lg.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s"))
lg.addHandler(handler)


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self._obj_map = instantiate_class_from_config(config)
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
            context_generator.decompose_repo(repo_folder, repo_name, context_folder)
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
        lg.info("Running topic model...")
        topic_model_outputs = [self.topic_model.run(data, gnn_head) for gnn_head in gnn_head_outputs]
        return gnn_head_outputs, topic_model_outputs

    def get_repo_contents(self, git_repo_path):
        contents = self.semantic_graph_context_generator.get_repo_contents(git_repo_path)
        return contents

    def decompose_repo(self, contents, git_repo_path, name_id, topic_tree, out_path):
        self.semantic_graph_context_generator.decompose_repo(contents, git_repo_path, name_id, topic_tree, out_path)
