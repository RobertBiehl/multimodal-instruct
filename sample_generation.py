import hashlib
import json
from textwrap import dedent
import random
from typing import Callable, Dict

from conversation import ConversationBatchService
from data_definitions import *


def generate_samples(context: Context, prompt_config) -> [Dict]:
    include_captions = "captions" in prompt_config["inputs"]
    include_boxes = "boxes" in prompt_config["inputs"]

    captions_str = "\n".join([x.caption for x in context.captions]) if include_captions else ""
    # TODO: limit decimal places for bbox
    object_str = ("\n\n" + "\n".join([f"{box.category_name}: {box.bbox}" for box in context.boxes])) \
        if len(context.boxes) > 0 and include_boxes else ""

    instruction = captions_str + object_str

    # construct prompt
    messages = [
        {"role": "system", "content": prompt_config["system_prompt"]}
    ]
    for sample in prompt_config["examples"]:
        messages.append({"role": "user", "content": sample["input"]})
        messages.append({"role": "assistant", "content": sample["output"]})
    messages.append({"role": "user", "content": instruction})

    return messages

def process_llm_result(question_id: str, result: str, context: Context, prompt_config) -> [Sample]:
    samples: List[Sample] = []

    type = prompt_config["type"]

    def remove_stopwords_strip(text: str) -> str:
        if "stopwords" in prompt_config:
            for sw in prompt_config["stopwords"]:
                text = text.replace(sw, "")
        return text.strip()

    if "split_user_assistant" in prompt_config and prompt_config["split_user_assistant"]:
        result = result.split(prompt_config["split_user_assistant"])
        assert len(
            result) % 2 == 0, f"{type}: Expecting on assistant answer for every user question. Got {len(result)} messages. \n result: {result}"
        for i in range(0, len(result), 2):
            samples.append(Sample(
                id=question_id,
                instruction=remove_stopwords_strip(result[i]),
                response=remove_stopwords_strip(result[i + 1]),
                image=context.sample_id,
                image_source=context.source,
                type=type
            ))
    else:
        samples.append(Sample(
            id=question_id,
            instruction=random.choice(prompt_config["instructions"]),
            response=remove_stopwords_strip(result),
            image=context.sample_id,
            image_source=context.source,
            type=type
        ))

    return samples