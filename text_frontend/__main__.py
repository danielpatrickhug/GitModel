from getpass import getpass

import openai

openai_secret = getpass("Enter the secret key: ")
# Set up OpenAI API credentials
openai.api_key = openai_secret
