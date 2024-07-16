## __all__ is a dict:
##   key is model_name in `model_zoo/chatglm-xx.json`
##   value is vendor specify model impl
# __all__ = {
#     "chatglm" : ChatGLMForConditionalGeneration,
#     "chatglm2" : ChatGLM2ForConditionalGeneration
# }

from typing import Dict, Tuple, Any

import torch
import torch.nn as nn

from .gpu_chatglm2 import GPUChatGLM2
from .gpu_meta_llama3 import GPUMetaLlama3
from .gpu_falcon import GPUFalcon
from .gpu_mixtral import GPUMixtral

from llm_perf.utils.logger import logger

__all__ = {
    "chatglm2": GPUChatGLM2,
    "meta_llama3": GPUMetaLlama3,
    "falcon": GPUFalcon,
    "mixtral": GPUMixtral
}