def format_system_prompts(git_repo_path, file_name):
    system_prompts = {
        "summary": f"""
                    Summarize the code the GitHub repository: {git_repo_path} you're currently in the file {file_name}
                    ChatGPT will use its advanced natural language processing capabilities to analyze the code and generate a concise summary that captures
                    the main functionality and purpose of the codebase. Additionally, ChatGPT can provide insights into the programming languages and libraries used in the repository,
                    as well as any notable features or functionalities that are present. Simply provide the necessary information, and let ChatGPT do the rest!
                    """,
        "question_asking": f"""
                    Ask ChatGPT questions about the codebase of the GitHub repository: {git_repo_path} you're currently in the file {file_name}.
                    ChatGPT asks questions that a new developer may ask about the codebase used in the repository,
                    as well as answer the question with step by step reasoning as a senoir dev would. All responses should first ask a question and then answer with reasoning.
                    """,
    }
    return system_prompts


def format_system_prompts_with_tree(git_repo_path, file_name, topic_tree):
    system_prompts = {
        "summary2": f"""
                    Provide a concise overview of the codebase located at {git_repo_path} and the file {file_name}, including its main features and capabilities.
                    ChatGPT will use its natural language processing capabilities to analyze the codebase and generate a brief summary of its main functionalities, tools, and technologies used. Here is the repos heirarchical topic tree
                    {topic_tree} You can use it to better understand the global structure of the repository
                    """,
        "summary3": f"""
                    Can you summarize the main functionality of the codebase located at {git_repo_path} and the file {file_name}, as well as any notable libraries or frameworks used?
                    ChatGPT will use its natural language processing capabilities to analyze the codebase and provide a brief summary of its main functionalities, libraries, and frameworks used.Here is the repos heirarchical topic tree
                    {topic_tree} You can use it to better understand the global structure of the repository
                    """,
    }
    return system_prompts
