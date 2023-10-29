import Globals
import requests
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional
import langchain
langchain.verbose = False


class CustomLLM(LLM):

    n: int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        response = requests.post(
            Globals.MODEL_URL+"/generate/",

            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": 0,
                    "max_tokens": 512,
                }
            }
        )
        return response.json()["generated_text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
