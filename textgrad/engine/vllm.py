from vllm import LLM, SamplingParams
from .base import EngineLM, CachedEngine

class ChatVLLM(ChatOpenAI):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
        
    def __init__(self, model_name_or_path, system_prompt=DEFAULT_SYSTEM_PROMPT, **kwargs):
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_vllm_{model_name_or_path}.db")
        CachedEngine.__init__(self, cache_path)
        self.system_prompt = system_prompt
        # Set OpenAI's API key and API base to use vLLM's API server.
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"

        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )



    