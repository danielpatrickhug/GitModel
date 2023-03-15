import json

import openai

from src.ast_parsers.python_ast_parser import get_methods, parse_github_repo
from src.config import Config
from src.system_prompts.format_system_prompts import format_system_prompts, format_system_prompts_with_tree


class SemanticGraphContextGenerator:
    def __init__(self, config: Config):
        self.config = config

    def __repr__(self) -> str:
        return f"SemanticGraphContextGenerator(config={self.config})"

    def get_repo_contents(self, git_repo_path):
        contents = parse_github_repo(git_repo_path)
        print(len(contents))
        pruned_contents = []
        for cont in contents:
            fp = cont["file_name"]
            fn = fp.split("/")[-1]
            fn_ = fn.split(".")[0]
            if fn_ in ["__init__"] or fn_.split("_")[-1] in ["test"]:
                continue
            else:
                pruned_contents.append(cont)
        return pruned_contents

    def chat_gpt_inference(self, messages: list):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
        )
        return response

    def create_prompt_message_template(self, text, role="user"):
        if role not in ["user", "assistant"]:
            raise ValueError("Not a valid role. Please use 'user' or 'assistant'.")
        return {"role": role, "content": text}

    def compose_inference(self, text_block, messages):
        user_template = self.create_prompt_message_template(text_block, role="user")
        messages.append(user_template)
        chat_resp = self.chat_gpt_inference(messages)
        reply_text = chat_resp["choices"][0]["message"]["content"]
        assistant_template = self.create_prompt_message_template(reply_text, role="assistant")
        messages.append(assistant_template)
        return messages, reply_text

    def process_transcript(self, segments, file_name, git_repo_path, output_file_path, system_prompt, task, code_type):
        messages = [{"role": "system", "content": system_prompt}]

        with open(output_file_path, "a") as f:
            for i, sent in enumerate(segments):
                text_block = f"""```{sent}```"""
                try:
                    messages, reply_text = self.compose_inference(text_block[:2000], messages)

                except Exception:
                    messages, reply_text = self.compose_inference(
                        text_block[:2000], [{"role": "system", "content": system_prompt}]
                    )
                row = {
                    "git_repo_path": git_repo_path,
                    "file_name": file_name,
                    "code_type": code_type,
                    "system_task": task,
                    "system_prompt": system_prompt,
                    "conversation_history": messages,
                    "assistant_reply": reply_text,
                }
                json.dump(row, f)
                f.write("\n")

        return messages

    def decompose_repo(self, git_repo_path, name_id, out_path, skip_graph_generation=False):
        contents = self.get_repo_contents(git_repo_path)
        context_paths = []
        for cont in contents:
            if not self.config["with_tree"]:
                system_prompts = format_system_prompts(git_repo_path, cont["file_name"])
            else:
                system_prompts = format_system_prompts_with_tree(
                    git_repo_path, cont["file_name"], self.config["topic_tree"]
                )
            for k, v in zip(system_prompts.keys(), system_prompts.values()):
                func_task = k
                out_file_name = f"{name_id}_{func_task}"
                print(f"file_name: {cont['file_name']}")
                num_funcs = len(cont["functions"])
                num_classes = len(cont["classes"])
                print(f"Imports: {cont['imports']}")
                context_paths.append(f"{out_path}/{out_file_name}.jsonl")
                if skip_graph_generation:
                    continue
                try:
                    if num_funcs > 0 or num_classes > 0:
                        print(f" len of functions: {len(cont['functions'])}")
                        _ = self.process_transcript(
                            cont["functions"],
                            cont["file_name"],
                            git_repo_path,
                            f"{out_path}/{out_file_name}.jsonl",
                            system_prompts[func_task],
                            func_task,
                            "functions",
                        )
                        _ = self.process_transcript(
                            cont["classes"],
                            cont["file_name"],
                            git_repo_path,
                            f"{out_path}/{out_file_name}.jsonl",
                            system_prompts[func_task],
                            func_task,
                            "classes",
                        )
                        for cls in cont["classes"]:
                            cls_funcs = get_methods(cls)

                            print(f"len of class: {len(cls)}")
                            for method in cls_funcs:
                                print(f"len of method: {len(method)}")
                            _ = self.process_transcript(
                                cls_funcs,
                                cont["file_name"],
                                git_repo_path,
                                f"{out_path}/{out_file_name}.jsonl",
                                system_prompts[func_task],
                                func_task,
                                "methods",
                            )
                except Exception as e:
                    print(e)
                    continue
                print("\n\n")
        return set(context_paths)
