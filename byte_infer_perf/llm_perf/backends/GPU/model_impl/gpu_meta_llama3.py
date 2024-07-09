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
        self.state_dict = None
        if self.mp_rank == 0:
            self.state_dict = self.torch_load_wrapper(
                self.ckpt_path, map_location=torch.device("cpu"))
            for weight_name, weights in self.state_dict.items():
                concat_dim = self.get_concat_dim(weight_name)
                if concat_dim == -1:
                    assert all(weight == weights[0] for weight in weights[1:])
                    self.state_dict[weight_name] = weights[0]
                else:
                    self.state_dict[weight_name] = torch.cat(weights, dim=concat_dim)




        if self.mp_size == 1:
            return self.state_dict

        # mp_size > 2
        # broadcast state_dict from rank 0 to other ranks
        self.broadcast_meta()

        # TODO: not finished, need to specify
        self.broadcast_weight(f"norm.weight")
        self.scatter_weight("tok_embeddings.weight", dim=0)
        self.scatter_weight("output.weight", dim = 0)

        for i in range(self.model.n_layers):
            self.broadcast_weight(f"layers.{i}.attention_norm.weight")
            self.broadcast_weight(f"layers.{i}.ffn_norm.weight")

            self.scatter_weight(f"layers.{i}.attention.wq.weight", dim=0)
            self.scatter_weight(f"layers.{i}.attention.wk.weight", dim=0)
            self.scatter_weight(f"layers.{i}.attention.wv.weight", dim=0)
            self.scatter_weight(f"layers.{i}.attention.wo.weight", dim=1)

            self.scatter_weight(f"layers.{i}.feed_forward.w1.weight", dim=0)
            self.scatter_weight(f"layers.{i}.feed_forward.w3.weight", dim=0)
            self.scatter_weight(f"layers.{i}.feed_forward.w2.weight", dim=1)

        return self.state_dict

    def infusion_to_model(self):
        self.model.load_state_dict(self.state_dict)
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


        self.prefix = "transformer.encoder.layers"
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
        logger.info(f"{self.local_rank}: gpu_meta_llama3 model init begin")

        with init_empty_weights():
            self.transformer_model = Transformer(self.llama3_config)
            self.transformer_model.eval()

        logger.info(f"{self.local_rank}: gpu_meta_llama3 model init end")
        check_memory_usage("After build model")
        logger.info(f"{self.local_rank}: gpu_meta_llama3 model load weight begin")

        self.load_weight(self.model_path)

        logger.info(f"{self.local_rank}: gpu_meta_llama3 model load weight end")
        check_memory_usage("After load_weight")
        logger.info(f"{self.local_rank}: gpu_meta_llama3 model to device begin")

        self.transformer_model.half().cuda()

        logger.info(f"{self.local_rank}: gpu_meta_llama3 model to device end")
        check_memory_usage("After model to device")
        logger.info(f"{self.local_rank}: gpu_meta_llama3 model init kv_cache begin")

        self.kv_cache = self.init_kvcache(torch.float16)

        logger.info(f"cuda model {self.model_path} loaded {self.transformer_model}")


    def load_weight(self, ckpt_path):
        p_loader = GPUMetaLlama3Loader(
            self.prefix, self.transformer_model, 
            self.mp_size, self.local_rank, 
            ckpt_path
        )
        p_loader.load()
        p_loader.infusion_to_model()


    def init_kvcache(self, dtype):
        # TODO: do nothing, need to implement after
        kv_cache = ()
        return kv_cache
    
    def forward(self, inputs : Dict[str, torch.Tensor]):
        raise NotImplemented
        # model_outputs = self.transformer_model.forward(
        #     **inputs, 
        #     past_key_values=self.kv_cache, 
        #     use_cache=True, 
        #     output_attentions=False, 
        #     output_hidden_states=False, 
        #     return_dict=True, 
        #     return_last_logit=(not inputs["get_input_logits"])
        #     tokens= kwargs["input_ids"]
        # )
        output_dict = {
            "logits": model_outputs.logits
        }
        return output_dict