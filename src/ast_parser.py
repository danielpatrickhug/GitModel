import ast
import os

import astor


def parse_python_file(file_path):
    parsed_contents = {
        "imports": [],
        "globals": [],
        "classes": [],
        "functions": [],
    }

    with open(file_path, "r") as file:
        file_contents = file.read()
        parsed_tree = ast.parse(file_contents)

    for node in ast.iter_child_nodes(parsed_tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            parsed_contents["imports"].append(astor.to_source(node).strip())
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            parsed_contents["globals"].append(astor.to_source(node).strip())
        elif isinstance(node, ast.FunctionDef):
            if node.name == "main":
                parsed_contents["functions"].append(ast.get_source_segment(file_contents, node))
            else:
                parsed_contents["functions"].append(ast.get_source_segment(file_contents, node))
        elif isinstance(node, ast.ClassDef):
            parsed_contents["classes"].append(ast.get_source_segment(file_contents, node))

    return parsed_contents


def get_methods(class_or_str):
    if isinstance(class_or_str, str):
        class_or_str = ast.parse(class_or_str)

    method_nodes = [node for node in ast.iter_child_nodes(class_or_str) if isinstance(node, ast.FunctionDef)]
    method_sources = []
    for node in method_nodes:
        source_lines, _ = ast.get_source_segment(class_or_str, node)
        method_sources.append("".join(source_lines).strip())
    return method_sources


def parse_github_repo(local_dir):
    parsed_files = []
    content_labels = {0: "imports", 1: "globals", 2: "classes", 3: "functions", 4: "main", 5: "file_name"}

    for root, dirs, files in os.walk(local_dir):
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                try:
                    parsed_contents = parse_python_file(file_path)
                    content = {content_labels[i]: v for i, v in enumerate(parsed_contents.values())}
                    content[content_labels[5]] = file_path
                    parsed_files.append(content)
                except Exception as e:
                    print(e)
                    continue

    return parsed_files
