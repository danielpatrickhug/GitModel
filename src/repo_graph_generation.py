from ast_parsers.python_ast_parser import get_methods, parse_github_repo
from models.transformers.chatgpt_api_inference import process_transcript
from system_prompts.format_system_prompts import format_system_prompts, format_system_prompts_with_tree


def get_repo_contents(git_repo_path):
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
            print(cont["file_name"])
            pruned_contents.append(cont)
    return pruned_contents


def decompose_repo(contents, git_repo_path, name_id, topic_tree, out_path, with_tree=False):
    # This could be done better
    for cont in contents:
        if with_tree:
            system_prompts = format_system_prompts(git_repo_path, cont["file_name"])
        else:
            system_prompts = format_system_prompts_with_tree(git_repo_path, cont["file_name"], topic_tree)
        for k, v in zip(system_prompts.keys(), system_prompts.values()):
            func_task = k
            out_file_name = f"{name_id}_{func_task}"
            print(f"file_name: {cont['file_name']}")
            num_funcs = len(cont["functions"])
            num_classes = len(cont["classes"])
            print(f"Imports: {cont['imports']}")
            try:
                if num_funcs > 0 or num_classes > 0:
                    print(f"functions: {cont['functions']}")
                    _ = process_transcript(
                        cont["functions"],
                        cont["file_name"],
                        git_repo_path,
                        f"{out_path}/{out_file_name}.jsonl",
                        system_prompts[func_task],
                        func_task,
                        "functions",
                    )
                    print(f"Classes: {cont['classes']}")
                    _ = process_transcript(
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
                        _ = process_transcript(
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
