import re


def format_system_prompts(git_repo_path, file_name):
    system_prompts = {
        "self-instruction": f"""
            GitHub repository: {git_repo_path} you're currently in the file {file_name}.
            Generate a list of irreducible questions related to a given piece of code that can help understand its functionality and behavior. Ensure that the questions are consistently formatted and easy to parse. The irreducible questions should be answerable with the provided piece of code and cannot be further simplified. Consider the following aspects when generating the questions:

            What is the purpose of the code?
            What input does the code take, and how does it process the input?
            What output does the code produce?
            Use these aspects to generate a set of irreducible questions that can help anyone understand the code and its behavior.
            """,
    }

    return system_prompts


def extract_questions(prompt_output):
    # Define a regular expression to match the questions
    question_regex = r"\d+\.\s(.+\?)\n"

    # Find all matches of the question regex in the prompt output
    questions = re.findall(question_regex, prompt_output)
    return questions


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
        "summary": f"""
                    Summarize the code the GitHub repository: {git_repo_path} you're currently in the file {file_name}
                    I want you to act as a code summarizer, you will use advanced natural language processing capabilities to analyze the code and generate a concise summary that captures
                    the main functionality and purpose of the codebase. Additionally, ChatGPT can provide insights into the programming languages and libraries used in the repository,
                    as well as any notable features or functionalities that are present. here is a topic tree of the repository
                    {topic_tree} You can use it to better understand the global structure of the repository
                    """,
        "qa_chain": f"""
                    GitHub repository: {git_repo_path} you're currently in the file {file_name}.
                    I want you to asks questions that a new developer may ask about the codebase used in the repository,
                    as well as answer the question with step by step reasoning as a senior dev would. All responses should first ask a question and then answer with reasoning.
                    Here is a topic tree of the repository
                    {topic_tree} You can use it to better understand the global structure of the repository
                    """,
    }
    return system_prompts
