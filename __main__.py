import argparse
from getpass import getpass

import openai

from src import Pipeline

if __name__ == "__main__":
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--config", type=str, default="./test_config.yaml")
    argsparse.add_argument("--repo", type=str, default="https://github.com/LAION-AI/Open-Assistant.git")
    argsparse.add_argument("--repo_name", type=str, default="OA")

    args = argsparse.parse_args()

    openai_secret = getpass("Enter the secret key: ")
    # Set up OpenAI API credentials
    openai.api_key = openai_secret

    print("starting pipeline")
    pipeline = Pipeline.from_yaml(args.config)
    gnn_head_outputs, topic_model_outputs = pipeline.run(args.repo, args.repo_name)
    for i, topic_model_output in enumerate(topic_model_outputs):
        topic_model_output["data"].to_csv(f"context/{args.repo_name}_topic_model_outputs_{i}.csv")
        topic_model_output["topic_info"].to_csv(f"context/{args.repo_name}_topic_info_{i}.csv")
        with open(f"context/{args.repo_name}_tree_{i}.txt", "w", encoding="utf-8") as f:
            f.write(topic_model_output["tree"])
