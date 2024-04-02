import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import torch

import llm_perf.backends.GPU.common as comm
from llm_perf.core.common import Packet
from llm_perf.core.engine import CoreEngine
from llm_perf.core.sampler import CoreSampler
from llm_perf.core.scheduler import CoreScheduler
from llm_perf.utils.logger import logger


class GpuScheduler(CoreScheduler):
    def __init__(
        self,
        engine: CoreEngine,
        sampler: CoreSampler,
        **kwargs,
    ) -> None:
        super().__init__(
            engine=engine, sampler=sampler, comm=comm, **kwargs
        )
        self.max_batch_size = kwargs.get("max_batch_size")

    @torch.inference_mode()
    def scheduler_loop(self):
        batch: List[Packet] = []
        while True:
            # 1. select batch --> batch
            batch = self.select_batch(batch)
            if not batch:
                with self.packet_queue.not_empty:
                    self.packet_queue.not_empty.wait(0.1)
                continue

            logger.debug(f"get batch size: {len(batch)}")

            # 2. do inference -> logits
            outputs = self.engine.do_inference(batch)

            # 3. sample logits -> tokens
            next_tokens, softmax_out = self.sampler.sample(
                packets=batch, logits=outputs["last_logits"]
            )

            # 4.postprocess -> gen result
            generation_results = self.sampler.postprocess(
                packets=batch,
                infer_outputs=outputs,
                next_tokens=next_tokens,
            )

            # 5. add result to packet
            for i, gen_res in enumerate(generation_results):
                batch[i].add_result(gen_res)
                if gen_res.finish_reason:
                    batch[i].finish()

            # 6. is not finished -> remain
            remained: List[Packet] = []
            for packet in batch:
                if not packet.is_finished():
                    remained.append(packet)
            batch = remained

    def select_batch(self, batch):
        batching_size: int = len(batch)
        new_select_packets: List[Packet] = []

        while not self.packet_queue.empty():
            if batching_size == self.max_batch_size:
                break
            batching_size += 1
            new_select_packets.append(self.packet_queue.get())

        return batch + new_select_packets
