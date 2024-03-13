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
    include_boxes_openimages_format = "boxes_openimages" in prompt_config["inputs"]

    captions_str = "\n".join([x.caption for x in context.captions]) if include_captions else ""

    def round_bbox(bbox):
        return [round(bbox[0], 3), round(bbox[1], 3), round(bbox[2], 3), round(bbox[3], 3)]

    # TODO: limit decimal places for bbox
    object_str = ""
    if context.boxes:
        if include_boxes:
            object_str = "\n\n" + "\n".join([f"{box.category_name}: {round_bbox(box.bbox)}" for box in context.boxes])
        elif include_boxes_openimages_format:
            object_str = "\n\n" + "\n".join([f"{box.category_name}: {round_bbox(box.bbox)} {[box.confidence, box.is_occluded,box.is_truncated,box.is_group_of, box.is_depiction,box.is_inside]}" for box in context.boxes])

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

    if isinstance(result, dict):
        print(f"{question_id} Error result: {result}")
        return []

    type = prompt_config["type"]

    def remove_stopwords_strip(text: str) -> str:
        if "stopwords" in prompt_config:
            for sw in prompt_config["stopwords"]:
                text = text.replace(sw, "")
        return text.strip()

    if "split_user_assistant" in prompt_config and prompt_config["split_user_assistant"]:
        result = result.split(prompt_config["split_user_assistant"])
        if not len(result) % 2 == 0:
            print(f"Error {type}: Expecting on assistant answer for every user question. Got {len(result)} messages. \n result: {result}")

            # try cleaning up:
            result_recovered = []
            for res in result:
                if res.strip().startswith("Question:\n"):
                    if "\nAnswer:\n" in res:
                        result_recovered = result_recovered + res.split("\nAnswer:\n")
                    else:
                        result_recovered.append(res)
                elif res.strip().startswith("Answer:\n"):
                    if "\nQuestion:\n" in res:
                        result_recovered = result_recovered + res.split("\nQuestion:\n")
                    else:
                        result_recovered.append(res)

            result_recovered = [x for x in result_recovered if x.strip()]
            if len(result_recovered) % 2 == 0:
                print(f"Warning {type}: Recovered {len(result_recovered)} messages from {len(result)}")
                result = result_recovered
            else:
                return []
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