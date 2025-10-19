import copy
import os
import pickle
import time
from typing import List, Optional, Tuple, Union

import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qpva3 import QPVA3 as QPVA3Simple
from lmms_eval.protocol import ChatMessages

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qpva3_chat")
class QPVA3_Chat(QPVA3Simple):
    is_simple = False

    def generate_until(self, requests: List[Instance]) -> List[str]:
        # if os.path.exists('/dev/shm/tmp/metrics.pkl'):
        #     eval_logger.info("Load generate result from /dev/shm/tmp/metrics.pkl")
        #     return pickle.load(open("/dev/shm/tmp/metrics.pkl", "rb"))
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0]['question_id']

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        req_args = [reg.args for reg in requests]
        re_ords = utils.Collator(req_args, _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_words = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            docs = [self._preprocess_doc(self.task_dict[task][split][ids]) for (ids, task, split) in zip(doc_id, task, split)]
            chat_messages = [doc_to_messages[idx](doc) for idx, doc in enumerate(docs)]

            if self.model.planner is None or self.model.verifier is None or self.model.planner_perception is None:
                question_graphs = [None for _ in docs]
            else:
                question_graphs = [self._parse_agqa_hierarchy(doc) for doc in docs]

            start_time = time.time()
            results = self._model(
                message_templates=chat_messages,
                context_dicts=docs,
                question_graphs=question_graphs,
                all_gen_kwargs=all_gen_kwargs,
            )
            end_time = time.time()
            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time
            total_words += sum(len(result.final_answer.split()) for result in results)

            for result in results:
                clean_ans = parse_reasoning_model_answer(result.final_answer)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (ctx, all_gen_kwargs), clean_ans)
                pbar.update(1)

                eval_logger.debug(f"Question: {ctx}")
                eval_logger.debug(f"Model Raw Response: {result.final_answer}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_words / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_words": total_words,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)
        pickle.dump(res, open("/dev/shm/tmp/metrics.pkl", "wb"))

        pbar.close()
        return res
