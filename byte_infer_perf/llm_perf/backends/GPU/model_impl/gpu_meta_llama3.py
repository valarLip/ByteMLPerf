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

        # not finished, need to specify
        raise NotImplementedError
        self.broadcast_weight("transformer.embedding.word_embeddings.weight")
        # self.broadcast_weight("transformer.output_layer.weight")
        # self.broadcast_weight("transformer.rotary_pos_emb.inv_freq")
        # self.broadcast_weight("transformer.encoder.final_layernorm.weight")

        # for i, block in enumerate(self.model.transformer.encoder.layers):
        #     self.broadcast_weight(f"transformer.encoder.layers.{i}.input_layernorm.weight")
        #     self.broadcast_weight(f"transformer.encoder.layers.{i}.post_attention_layernorm.weight")

        #     self.scatter_weight(f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight", dim=0, split_mode='with_outter', outter=2)
        #     self.scatter_weight(f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight", dim=-1)
            
        #     self.scatter_weight(f"transformer.encoder.layers.{i}.self_attention.query_key_value.weight", dim=0, split_mode='split_outter', outter=[32, 2, 2])
        #     self.scatter_weight(f"transformer.encoder.layers.{i}.self_attention.query_key_value.bias", dim=0, split_mode='split_outter', outter=[32, 2, 2])
    
        #     self.scatter_weight(f"transformer.encoder.layers.{i}.self_attention.dense.weight", dim=-1)

        return self.state_dict

    def infusion_to_model(self):
        # not finished, need to specify
        raise NotImplementedError
        # self.model.transformer.embedding.word_embeddings.weight = self.to_parameter(
        #     self.state_dict[f"transformer.embedding.word_embeddings.weight"]
        # )
        # self.model.transformer.output_layer.weight = self.to_parameter(
        #     self.state_dict[f"transformer.output_layer.weight"]
        # )
        # self.model.transformer.rotary_pos_emb.inv_freq = self.to_parameter(
        #     self.state_dict[f"transformer.rotary_pos_emb.inv_freq"]
        # )
        # self.model.transformer.encoder.final_layernorm.weight = self.to_parameter(
        #     self.state_dict[f"transformer.encoder.final_layernorm.weight"]
        # )

        # for i, block in enumerate(self.model.transformer.encoder.layers):
        #     block.input_layernorm.weight = self.to_parameter(
        #         self.state_dict[f"transformer.encoder.layers.{i}.input_layernorm.weight"]
        #     )

        #     block.mlp.dense_4h_to_h.weight = self.to_parameter(
        #         self.state_dict[f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight"]
        #     )
        #     block.mlp.dense_h_to_4h.weight = self.to_parameter(
        #         self.state_dict[f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight"]
        #     )

        #     block.post_attention_layernorm.weight = self.to_parameter(
        #         self.state_dict[f"transformer.encoder.layers.{i}.post_attention_layernorm.weight"]
        #     )

        #     block.self_attention.dense.weight = self.to_parameter(
        #         self.state_dict[f"transformer.encoder.layers.{i}.self_attention.dense.weight"]
        #     )
        #     block.self_attention.query_key_value.bias = self.to_parameter(
        #         self.state_dict[f"transformer.encoder.layers.{i}.self_attention.query_key_value.bias"]
        #     )
        #     block.self_attention.query_key_value.weight = self.to_parameter(
        #         self.state_dict[f"transformer.encoder.layers.{i}.self_attention.query_key_value.weight"]
        #     )

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