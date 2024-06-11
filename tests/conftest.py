import os

import runpod
from dotenv import load_dotenv

load_dotenv(override=True)

ENDPOINT_ID = "qwp9kkjqhfb6t8"
RUNDPOD_API_KEY = os.environ["RUNPOD_API_KEY"]

runpod.api_key = RUNDPOD_API_KEY
