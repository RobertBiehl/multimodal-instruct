import hashlib
import shelve
from textwrap import dedent
import random
from typing import Callable

from data_definitions import *

caches = {}


def generate_samples(context: Context, pipe: Callable[[List], str], prompt_config, pipe_name: str) -> [Sample]:
    cache = caches.get(pipe_name, shelve.open(f"cache/response_cache_{pipe_name}"))

    samples: [Sample] = []
    for (key, current_config) in prompt_config.items():
        print(f"generate_samples {key}")

        include_captions = "captions" in current_config["inputs"]
        include_boxes = "boxes" in current_config["inputs"]

        captions_str = "\n".join([x.caption for x in context.captions]) if include_captions else ""
        object_str = ("\n\n" + "\n".join([f"{box.category_name}: {box.bbox}" for box in context.boxes])) \
            if len(context.boxes) > 0 and include_boxes else ""

        instruction = captions_str + object_str

        question_id = f"{pipe_name}_{hashlib.sha256(instruction.encode('utf-8')).hexdigest()}_{key}"

        if question_id not in cache:
            messages = []
            messages.append({"role": "system", "content": current_config["system_prompt"]})

            for sample in current_config["examples"]:
                messages.append({"role": "user", "content": sample["input"]})
                messages.append({"role": "assistant", "content": sample["output"]})

            messages.append({"role": "user", "content": instruction})

            result = pipe(messages)

            cache[question_id] = result
        else:
            result = cache[question_id]

        def remove_stopwords_strip(text: str) -> str:
            if "stopwords" in current_config:
                for sw in current_config["stopwords"]:
                    text = text.replace(sw, "")
            return text.strip()

        if "split_user_assistant" in current_config and current_config["split_user_assistant"]:
            result = result.split(current_config["split_user_assistant"])
            assert len(result) % 2 == 0, f"{key}: Expecting on assistant answer for every user question. Got {len(result)} messages. \n result: {result}"
            for i in range(0, len(result), 2):
                samples.append(Sample(
                    id=question_id,
                    instruction=remove_stopwords_strip(result[i]),
                    response=remove_stopwords_strip(result[i + 1]),
                    image=context.sample_id,
                    image_source=context.source,
                    type=key
                ))
        else:
            samples.append(Sample(
                id=question_id,
                instruction=random.choice(current_config["instructions"]),
                response=remove_stopwords_strip(result),
                image=context.sample_id,
                image_source=context.source,
                type=key
            ))

    return samples
