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

        raise NotImplementedError

        # mp_size > 2
        # broadcast state_dict from rank 0 to other ranks
        self.scatter_weight("transformer.word_embeddings.weight", dim=0)
        self.scatter_weight("lm_head.weight", dim=0)
        self.broadcast_weight("transformer.ln_f.weight")
        self.broadcast_weight("transformer.ln_f.bias")

        # TODO: layers enumeration and scatter
        for i, block in enumerate(self.model.transformer.encoder.layers):
            # self.scatter_weight(f"transformer.h.{i}.ln_attn.bias", dim=0)
            # self.scatter_weight(f"transformer.h.{i}.ln_attn.weight", dim=0)

            # self.scatter_weight(f"transformer.h.{i}.ln_mlp.bias", dim=0)
            # self.scatter_weight(f"transformer.h.{i}.ln_mlp.weight", dim=0)

            # self.scatter_weight(f"transformer.h.{i}.mlp.dense_h_to_4h.weight", dim=0, split_mode='with_outter', outter=2)
            # self.scatter_weight(f"transformer.h.{i}.mlp.dense_4h_to_h.weight", dim=-1)

            # self.scatter_weight(f"transformer.h.{i}.self_attention.dense.weight", dim=-1)
            # self.scatter_weight(f"transformer.h.{i}.self_attention.query_key_value.weight", dim=0)
            

        return self.state_dict

    def infusion_to_model(self):
        # TODO: infusion
        raise NotImplementedError

        return self.model


class GPUFalcon(nn.Module):
    def __init__(self, xpu_cfg: Dict[str, Any]) -> None:
        super().__init__()

        self.xpu_cfg = xpu_cfg
        self.model_config = xpu_cfg["model_config"]

        self.model_name = self.model_config["model_name"]
        self.model_path = self.model_config["model_path"]
        self.model_network = self.model_config["network"]

        self.llama3_config = ModelArgs(**self.model_network)

        # dist config
        self.mp_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))


        self.transformer_model : Transformer = None

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
            self.transformer_model = Transformer(self.llama3_config)
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
        # No need to init kv cache here
        kv_cache = ()
        return kv_cache
    
    def forward(self, inputs : Dict[str, torch.Tensor]):
        # TODO: finish the forwarding
        raise NotImplementedError
        
        output_dict = {
            "logits": logits
        }
        return output_dict