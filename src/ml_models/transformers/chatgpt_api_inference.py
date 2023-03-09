import json

import openai


def chat_gpt_inference(messages: list, max_tokens=400, temperature=0.4):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, max_tokens=max_tokens, temperature=temperature
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


def process_transcript(segments, file_name, git_repo_path, output_file_path, system_prompt, task, code_type):
    messages = [{"role": "system", "content": system_prompt}]

    with open(output_file_path, "a") as f:
        for i, sent in enumerate(segments):
            text_block = f"""```{sent}```"""
            try:
                messages, reply_text = compose_inference(text_block[:2000], messages)

            except Exception:
                messages, reply_text = compose_inference(
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
