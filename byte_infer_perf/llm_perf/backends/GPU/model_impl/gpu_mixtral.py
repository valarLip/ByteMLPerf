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

from .mixtral import MixtralForCausalLM, MixtralConfig

class GPUMixtralLoader(GpuCkptLoader):
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

        self.scatter_weight("model.embed_tokens.weight", dim=0)
        self.scatter_weight("lm_head.weight", dim=0)
        self.broadcast_weight("model.norm.weight")

        for i, layer in enumerate(self.model.model.layers):
            pass
            self.scatter_weight(f"model.layers.{i}.self_attn.q_proj.weight", dim=0)
            self.scatter_weight(f"model.layers.{i}.self_attn.k_proj.weight", dim=0)
            self.scatter_weight(f"model.layers.{i}.self_attn.v_proj.weight", dim=0)
            self.scatter_weight(f"model.layers.{i}.self_attn.o_proj.weight", dim=-1)
            self.scatter_weight(f"model.layers.{i}.block_sparse_moe.gate.weight", dim=0)
            for j, expert in enumerate(layer.block_sparse_moe.experts):
                pass
                self.scatter_weight(f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight", dim=0)
                self.scatter_weight(f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight", dim=-1)
                self.scatter_weight(f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight", dim=0)
            self.broadcast_weight(f"model.layers.{i}.input_layernorm.weight")
            self.broadcast_weight(f"model.layers.{i}.post_attention_layernorm.weight")
            

        return self.state_dict

    def infusion_to_model(self):
        self.model.lm_head.weight = self.to_parameter(
            self.state_dict[f"lm_head.weight"]
        )
        self.model.model.embed_tokens.weight = self.to_parameter(
            self.state_dict[f"model.embed_tokens.weight"]
        )
        self.model.model.norm.weight = self.to_parameter(
            self.state_dict[f"model.norm.weight"]
        )
        for i, layer in enumerate(self.model.model.layers):
            layer.self_attn.q_proj.weight = self.to_parameter(
                self.state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
            )
            layer.self_attn.k_proj.weight = self.to_parameter(
                self.state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
            )
            layer.self_attn.v_proj.weight = self.to_parameter(
                self.state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
            )
            layer.self_attn.o_proj.weight = self.to_parameter(
                self.state_dict[f"model.layers.{i}.self_attn.o_proj.weight"]
            )
            layer.block_sparse_moe.weight = self.to_parameter(
                self.state_dict[f"model.layers.{i}.block_sparse_moe.gate.weight"]
            )
            for j, expert in enumerate(layer.block_sparse_moe.experts):
                expert.w1.weight = self.to_parameter(
                    self.state_dict[f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight"]
                )
                expert.w2.weight = self.to_parameter(
                    self.state_dict[f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight"]
                )
                expert.w3.weight = self.to_parameter(
                    self.state_dict[f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight"]
                )
            layer.input_layernorm.weight = self.to_parameter(
                self.state_dict[f"model.layers.{i}.input_layernorm.weight"]
            )
            layer.post_attention_layernorm.weight = self.to_parameter(
                self.state_dict[f"model.layers.{i}.post_attention_layernorm.weight"]
            )
        return self.model


class GPUMixtral(nn.Module):
    def __init__(self, xpu_cfg: Dict[str, Any]) -> None:
        super().__init__()

        self.xpu_cfg = xpu_cfg
        self.model_config = xpu_cfg["model_config"]

        self.model_name = self.model_config["model_name"]
        self.model_path = self.model_config["model_path"]
        self.model_network = self.model_config["network"]

        self.mixtral_config = MixtralConfig(**self.model_network)

        # dist config
        self.mp_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        self.prefix = "model.layers"
        self.transformer_model : MixtralForCausalLM = None

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

        self.transformer_model = MixtralForCausalLM(self.mixtral_config)
        self.transformer_model.eval()

        check_memory_usage("After build model")

        self.load_weight(self.model_path)

        check_memory_usage("After load_weight")

        # self.transformer_model.lm_head.weight.half().cuda()
        # self.transformer_model.model.embed_tokens.weight.half().cuda()
        # self.transformer_model.model.norm.weight.half().cuda()
        # for i, layer in enumerate(self.transformer_model.model.layers):
        #     layer.self_attn.q_proj.weight.half().cuda()
        #     layer.self_attn.k_proj.weight.half().cuda()
        #     layer.self_attn.v_proj.weight.half().cuda()
        #     layer.self_attn.o_proj.weight.half().cuda()
        #     layer.block_sparse_moe.weight.half().cuda()
        #     for j, expert in enumerate(layer.block_sparse_moe.experts):
        #         expert.w1.weight.half().cuda()
        #         expert.w2.weight.half().cuda()
        #         expert.w3.weight.half().cuda()
        #     layer.input_layernorm.weight.half().cuda()
        #     layer.post_attention_layernorm.weight.half().cuda()
        self.transformer_model.half().cuda()

        check_memory_usage("After model to device")

        self.kv_cache = self.init_kvcache(torch.float16)

        logger.info(f"cuda model {self.model_path} loaded {self.transformer_model}")


    def load_weight(self, ckpt_path):
        p_loader = GPUMixtralLoader(
            self.prefix, self.transformer_model, 
            self.mp_size, self.local_rank, 
            ckpt_path
        )
        p_loader.load()
        p_loader.infusion_to_model()


    def init_kvcache(self, dtype):
        # TODO: finish the kv cache initiating
        return None
    
    def forward(self, inputs : Dict[str, torch.Tensor]):
        model_outputs = self.transformer_model.forward(
            **inputs, 
            # past_key_values=self.kv_cache, 
            use_cache=True, 
            output_attentions=False, 
            output_hidden_states=False, 
            return_dict=True
        )
        output_dict = {
            "logits": model_outputs.logits
        }
        return output_dict