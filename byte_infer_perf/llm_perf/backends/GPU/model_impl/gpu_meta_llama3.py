import os
import torch
import torch.distributed as dist
import torch.nn as nn
import pathlib

from typing import Dict, Any
from llm_perf.utils.logger import logger
from llm_perf.utils.ps_utils import check_memory_usage
from llm_perf.utils.dist_utils import check_dist

from accelerate import init_empty_weights
from datetime import timedelta

from llm_perf.backends.GPU.gpu_ckpt_loader import GpuCkptLoader
from llm_perf.core.ckpt_loader import CoreCkptLoader, MetaLlama3_ModelLoader

from .meta_llama3 import ModelArgs, Transformer


class GPUMetaLlama3Loader(GpuCkptLoader):
    def __init__(
        self, 
        prefix, 
        model, 
        mp_size=1, 
        mp_rank=0, 
        ckpt_path: str = ""
    ):
        super().__init__(prefix, model, mp_size, mp_rank, ckpt_path)

    def get_concat_dim(self, weight_name: str) -> int:
        """
        Get the concat dimension according different weight name.

        Args:
            weight_name: the weight name
        
        Returns:
            the dimension to concat (-1 means no need to concat)
        """
        if (weight_name.endswith('.attention.wq.weight')
            or weight_name.endswith('.attention.wk.weight')
            or weight_name.endswith('.attention.wv.weight')
            or weight_name.endswith('.feed_forward.w1.weight')
            or weight_name.endswith('.feed_forward.w3.weight')
            or weight_name == "tok_embeddings.weight"
            or weight_name == "output.weight"):
            return 0
        elif (weight_name.endswith('.attention.wo.weight')
              or weight_name.endswith('.feed_forward.w2.weight')):
            return 1
        return -1

    def parallel_loader(self):
        self.state_dict = {}

        model_dir = pathlib.Path(self.ckpt_path).absolute()
        if not model_dir.exists() or not model_dir.is_dir():
            if self.mp_rank == 0:
                print(f"{model_dir} not exists or is not a directory")
            return
        
        split_model_dir = model_dir.joinpath(f"TP{self.mp_size}")
        if not split_model_dir.exists() or not split_model_dir.is_dir():
            if self.mp_rank == 0:
                print(f"{split_model_dir} not exists or is not a directory, please split model first.")
            return

        model_loader = MetaLlama3_ModelLoader(split_model_dir / f"device_{self.mp_rank}")
        self.state_dict = model_loader.load_weight()
        
    
    

    def infusion_to_model(self):
        self.model.tok_embeddings.weight = self.to_parameter(
            self.state_dict[f"tok_embeddings.weight"]
        )
        self.model.output.weight = self.to_parameter(
            self.state_dict[f"output.weight"]
        )
        self.model.norm.weight = self.to_parameter(
            self.state_dict[f"norm.weight"]
        )
        for i, block in enumerate(self.model.layers):
            block.attention.wq.weight = self.to_parameter(
                self.state_dict[f"layers.{i}.attention.wq.weight"]
            )
            block.attention.wk.weight = self.to_parameter(
                self.state_dict[f"layers.{i}.attention.wk.weight"]
            )
            block.attention.wv.weight = self.to_parameter(
                self.state_dict[f"layers.{i}.attention.wv.weight"]
            )
            block.attention.wo.weight = self.to_parameter(
                self.state_dict[f"layers.{i}.attention.wo.weight"]
            )

            block.feed_forward.w1.weight = self.to_parameter(
                self.state_dict[f"layers.{i}.feed_forward.w1.weight"]
            )
            block.feed_forward.w2.weight = self.to_parameter(
                self.state_dict[f"layers.{i}.feed_forward.w2.weight"]
            )
            block.feed_forward.w3.weight = self.to_parameter(
                self.state_dict[f"layers.{i}.feed_forward.w3.weight"]
            )

            block.attention_norm.weight = self.to_parameter(
                self.state_dict[f"layers.{i}.attention_norm.weight"]
            )
            block.ffn_norm.weight = self.to_parameter(
                self.state_dict[f"layers.{i}.ffn_norm.weight"]
            )

        return self.model


class GPUMetaLlama3(nn.Module):
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


        self.prefix = "layers"
        self.transformer_model : Transformer = None

    def init_inference(self):
        torch.cuda.set_device(self.local_rank)

        logger.info(f"RANK: {self.local_rank} {self.mp_size} init_process_group...")
        if self.mp_size > 1:
            dist.init_process_group(
                backend="nccl", 
                world_size=self.mp_size, 
                rank=self.local_rank,
                timeout=timedelta(seconds=7200000)
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
        p_loader = GPUMetaLlama3Loader(
            self.prefix, self.transformer_model, 
            self.mp_size, self.local_rank, 
            ckpt_path
        )
        p_loader.parallel_loader()
        p_loader.infusion_to_model()


    def init_kvcache(self, dtype):
        # llama3 model already has its own kvcache
        # max_seq_len = self.llama3_config.max_seq_len
        # max_batch_size = self.llama3_config.max_batch_size
        # n_heads = self.llama3_config.n_heads
        # head_dim = self.llama3_config.dim // n_heads
        
        # n_local_heads = n_heads // self.mp_size if self.mp_size % n_heads else 1

        # past_key_values = ()
        # layer_num = self.llama3_config.n_layers
        # for i in range(layer_num):
        #     # [max_batch_size, max_seq_len, kv_head_num, kv_head_dim]
        #     key_cache = torch.zeros(
        #         (max_batch_size, max_seq_len, n_local_heads, head_dim), 
        #         dtype=dtype, 
        #         device='cuda'
        #     )
        #     value_cache = torch.zeros(
        #         (max_batch_size, max_seq_len, n_local_heads, head_dim), 
        #         dtype=dtype, 
        #         device='cuda'
        #     )
        #     past_key_values += ((key_cache, value_cache),)

        return None
    
    def forward(self, inputs : Dict[str, torch.Tensor]):
        logits = self.transformer_model.forward(
            tokens=inputs["input_ids"],
            start_pos=inputs["position_ids"][0][0]
        )
        output_dict = {
            "logits": logits
        }
        return output_dict