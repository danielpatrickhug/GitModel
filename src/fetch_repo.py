import os

from git import Repo


def clone_and_create_context_folder(repo_name: str, id_name: str) -> str:
    # Create a work folder in the current working directory if it doesn't already exist
    work_folder = os.path.join(os.getcwd(), "work")
    if not os.path.exists(work_folder):
        os.makedirs(work_folder)

    # Clone the repository into the work folder
    repo_path = os.path.join(work_folder, id_name)
    if not os.path.exists(repo_path):
        Repo.clone_from(repo_name, repo_path)

    # Create a context folder with the specified id_name in the current working directory
    context_folder = os.path.join(os.getcwd(), f"context/{id_name}_context")
    if not os.path.exists(context_folder):
        os.makedirs(context_folder)

    return repo_path, context_folder
