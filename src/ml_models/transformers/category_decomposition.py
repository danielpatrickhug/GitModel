import json
from getpass import getpass

import openai

openai_secret = getpass("Enter the secret key: ")
# Set up OpenAI API credentials
openai.api_key = openai_secret



def chat_gpt_inference(messages: list):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1500,
        temperature=0.2,
    )
    return response


def create_prompt_message_template(text, role="user"):
    if role not in ["user", "assistant"]:
        raise ValueError("Not a valid role. Please use 'user' or 'assistant'.")
    return {"role": role, "content": text}


def compose_inference(text_block, messages):
    user_template = create_prompt_message_template(text_block, role="user")
    messages.append(user_template)
    chat_resp = chat_gpt_inference(messages)
    reply_text = chat_resp["choices"][0]["message"]["content"]
    assistant_template = create_prompt_message_template(reply_text, role="assistant")
    messages.append(assistant_template)
    return messages, reply_text

def category_generation(topic: str, topic_reasoning_chain: list):
    """Decompose a category into its constituent parts.

    Args:
        category (str): The category to decompose.

    Returns:
        list: The list of constituent parts of the category.
    """
    # Define the prompt
    system_prompt = """Please design a categorical framework that captures the essential qualities and thought processes of a genius category theorist. Begin by identifying the key objects and morphisms that represent the foundational elements of a genius category theorist's intellectual toolbox. Then, outline the relationships and transformations between these objects and morphisms, highlighting the critical reasoning, creativity, and innovative problem-solving strategies employed by such a theorist. Finally, explain how this category can serve as a foundation for further exploration and development of novel ideas within the realm of category theory."""

    messages = [{"role": "system", "content": system_prompt}]

    with open(topic_reasoning_chain, "r") as f:
        segments = f.readlines()


    with open("ct_output_file.jsonl", "a") as f:
        for i, line in enumerate(segments):
            # print(line)
            topic = eval(line)["label"]
            print(topic)
            text_block = prompt = f"""Please provide a description of {topic} within a categorical framework,
                                        detailing the objects and morphisms that represent its essential components. Offer a brief summary of these objects and
                                        morphisms as they pertain to the category specific to {topic}."""
            try:
                messages, reply_text = compose_inference(text_block, messages)

            except Exception:
                messages, reply_text = compose_inference(text_block, [{"role": "system", "content": system_prompt}])
            print(reply_text)
            print()
            row = {
                "topic": topic,
                "assistant_reply": reply_text,
            }
            json.dump(row, f)
            f.write("\n")
