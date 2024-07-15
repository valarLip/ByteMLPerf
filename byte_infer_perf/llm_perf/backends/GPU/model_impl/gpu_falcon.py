import os
import torch
import torch.distributed as dist
import torch.nn as nn

from typing import Dict, Any
from llm_perf.utils.logger import logger
from llm_perf.utils.ps_utils import check_memory_usage
from llm_perf.utils.dist_utils import check_dist

from accelerate import init_empty_weights

from llm_perf.backends.GPU.gpu_ckpt_loader import GpuCkptLoader

from .falcon import FalconForCausalLM, FalconConfig

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class GPUFalconLoader(GpuCkptLoader):
    def __init__(
        self, 
        prefix, 
        model, 
        mp_size=1, 
        mp_rank=0, 
        ckpt_path: str = ""
    ):
        super().__init__(prefix, model, mp_size, mp_rank, ckpt_path)

    def parallel_loader(self):
        self.state_dict = None
        if self.mp_rank == 0:
            self.state_dict = self.torch_load_wrapper(
                self.ckpt_path, map_location=torch.device("cpu"))

        if self.mp_size == 1:
            return self.state_dict

        # mp_size > 2
        # broadcast state_dict from rank 0 to other ranks
        self.broadcast_meta()
        
        self.scatter_weight("transformer.word_embeddings.weight", dim=0)
        self.scatter_weight("lm_head.weight", dim=0)
        self.broadcast_weight("transformer.ln_f.weight")
        self.broadcast_weight("transformer.ln_f.bias")

        for i, block in enumerate(self.model.transformer.h):
            pass
            self.broadcast_weight(f"transformer.h.{i}.ln_attn.bias")
            self.broadcast_weight(f"transformer.h.{i}.ln_attn.weight")

            self.broadcast_weight(f"transformer.h.{i}.ln_mlp.bias")
            self.broadcast_weight(f"transformer.h.{i}.ln_mlp.weight")

            self.scatter_weight(f"transformer.h.{i}.mlp.dense_h_to_4h.weight", dim=0)
            self.scatter_weight(f"transformer.h.{i}.mlp.dense_4h_to_h.weight", dim=-1)

            self.scatter_weight(f"transformer.h.{i}.self_attention.dense.weight", dim=-1)
            # TODO: to speciy the scatter method for query_key_value weight
            self.scatter_weight(f"transformer.h.{i}.self_attention.query_key_value.weight", dim=0, split_mode='split_outter', outter=[232, 8, 8])
            

        return self.state_dict

    def infusion_to_model(self):
        self.model.lm_head.weight = self.to_parameter(
            self.state_dict[f"lm_head.weight"]
        )
        self.model.transformer.word_embeddings.weight = self.to_parameter(
            self.state_dict[f"transformer.word_embeddings.weight"]
        )
        self.model.transformer.ln_f.weight = self.to_parameter(
            self.state_dict[f"transformer.ln_f.weight"]
        )
        self.model.transformer.ln_f.bias = self.to_parameter(
            self.state_dict[f"transformer.ln_f.bias"]
        )
        for i, block in enumerate(self.model.transformer.h):
            block.ln_attn.weight = self.to_parameter(
                self.state_dict[f"transformer.h.{i}.ln_attn.weight"]
            )
            block.ln_attn.bias = self.to_parameter(
                self.state_dict[f"transformer.h.{i}.ln_attn.bias"]
            )
            
            block.ln_mlp.weight = self.to_parameter(
                self.state_dict[f"transformer.h.{i}.ln_mlp.weight"]
            )
            block.ln_mlp.bias = self.to_parameter(
                self.state_dict[f"transformer.h.{i}.ln_mlp.bias"]
            )


            block.mlp.dense_h_to_4h.weight = self.to_parameter(
                self.state_dict[f"transformer.h.{i}.mlp.dense_h_to_4h.weight"]
            )
            block.mlp.dense_4h_to_h.weight = self.to_parameter(
                self.state_dict[f"transformer.h.{i}.mlp.dense_4h_to_h.weight"]
            )

            block.self_attention.dense.weight = self.to_parameter(
                self.state_dict[f"transformer.h.{i}.self_attention.dense.weight"]
            )
            block.self_attention.query_key_value.weight = self.to_parameter(
                self.state_dict[f"transformer.h.{i}.self_attention.query_key_value.weight"]
            )

        return self.model


class GPUFalcon(nn.Module):
    def __init__(self, xpu_cfg: Dict[str, Any]) -> None:
        super().__init__()

        self.xpu_cfg = xpu_cfg
        self.model_config = xpu_cfg["model_config"]

        self.model_name = self.model_config["model_name"]
        self.model_path = self.model_config["model_path"]
        self.model_network = self.model_config["network"]

        self.falcon_config = FalconConfig(**self.model_network)

        # dist config
        self.mp_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))


        self.transformer_model : FalconForCausalLM = None

        self.prefix = "transformer.h"

    def init_inference(self):
        torch.cuda.set_device(self.local_rank)

        if self.mp_size > 1:
            logger.info(f"RANK: {self.local_rank} {self.mp_size} init_process_group...")
            dist.init_process_group(
                backend="nccl", 
                world_size=self.mp_size, 
                rank=self.local_rank
            )
            check_dist()
            
        check_memory_usage("Begin")

        with init_empty_weights():
            self.transformer_model = FalconForCausalLM(self.falcon_config)
            self.transformer_model.eval()

        check_memory_usage("After build model")

        self.load_weight(self.model_path)

        check_memory_usage("After load_weight")

        self.transformer_model.half().cuda()

        check_memory_usage("After model to device")

        self.kv_cache = self.init_kvcache(torch.float16)

        logger.info(f"cuda model {self.model_path} loaded {self.transformer_model}")


    def load_weight(self, ckpt_path):
        p_loader = GPUFalconLoader(
            self.prefix, self.transformer_model, 
            self.mp_size, self.local_rank, 
            ckpt_path
        )
        p_loader.load()
        p_loader.infusion_to_model()


    def init_kvcache(self, dtype):
        # TODO: finish the kv cache initiating
        # max_seq_len = 4096
        # max_batch_size = self.xpu_cfg["max_batch_size"]
        # kv_head_num = self.falcon_config.num_kv_heads
        # kv_head_dim = self.falcon_config.head_dim
        
        # kv_head_num = kv_head_num // self.mp_size if self.mp_size % kv_head_num else 1

        # past_key_values = ()
        # layer_num = self.falcon_config.num_hidden_layers
        # for i in range(layer_num):
        #     # [max_seq_len, max_batch_size, kv_head_num, kv_head_dim]
        #     key_cache = torch.zeros(
        #         (max_seq_len, max_batch_size, kv_head_num, kv_head_dim), 
        #         dtype=dtype, 
        #         device='cuda'
        #     )
        #     value_cache = torch.zeros(
        #         (max_seq_len, max_batch_size, kv_head_num, kv_head_dim), 
        #         dtype=dtype, 
        #         device='cuda'
        #     )
        #     past_key_values += ((key_cache, value_cache),)

        return None
    
    def forward(self, inputs : Dict[str, torch.Tensor]):
        model_outputs = self.transformer_model.forward(
            **inputs, 
            past_key_values=self.kv_cache, 
            use_cache=True, 
            output_attentions=False, 
            output_hidden_states=False, 
            return_dict=True
        )
        output_dict = {
            "logits": model_outputs.logits
        }
        self.kv_cahce = model_outputs.past_key_values
        return output_dict