import argparse
import json
import re

from tqdm import tqdm

from enum_definitions import *
from data_definitions import *
from typing import Iterator
import yaml

import transformers
import torch

from dataset import coco
from sample_generation import *


def main():
    parser = argparse.ArgumentParser(description="Generate a question-answer instruction tuning dataset.")

    # Mandatory parameter
    parser.add_argument("sources", nargs='+', type=Source, choices=list(Source),
                        help="Datasets to use as sources. Available: " + ", ".join([s.value for s in Source]))

    # Optional parameters
    parser.add_argument("--dataset_storage_path", default="./cache/", type=str,
                        help="Storage path for downloaded dataset files")

    parser.add_argument("--output_path", default="instruct.jsonl", type=str,
                        help="Output file path for the created dataset (default: llava_instruct.json)")

    parser.add_argument("--output_format", default=OutputFormat.JSONL, type=OutputFormat,
                        choices=list(OutputFormat),
                        help="Output format of the dataset (default: llava-instruct). Available: " + ", ".join(
                            [o.value for o in OutputFormat]))

    # parser.add_argument("--context", default=[ContextProperties.CAPTIONS, ContextProperties.BOXES], nargs='*',
    #                     type=ContextProperties,
    #                     choices=list(ContextProperties),
    #                     help="Context appended to question and response generation queries (default: captions, boxes). Available: " + ", ".join(
    #                         [c.value for c in ContextProperties]))
    #
    # parser.add_argument("--question_context", default=[], nargs='*', type=QuestionContext,
    #                     choices=list(QuestionContext),
    #                     help="Context appended to output questions, visible at test time (default: none). Available: " + ", ".join(
    #                         [qc.value for qc in QuestionContext]))

    parser.add_argument("--model_source", default="huggingface")

    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf", type=str,
                        help="LLM to use for dataset generation. Huggingface model with chat support.")

    parser.add_argument("--prompt_config", default="prompt_config_llava.yaml", type=str,
                        help="Prompt config")

    args = parser.parse_args()

    # Placeholder for the dataset generation logic.
    print("Generating dataset with the following parameters:")
    print("Sources: ", args.sources)
    print("Output Path: ", args.output_path)
    print("Output Format: ", args.output_format)

    # model inference
    if args.model_source == "openai":
        def pipe(messages):
            from openai import OpenAI
            client = OpenAI()

            response = client.chat.completions.create(model=args.model, messages=messages)

            return response.choices[0].message.content
    else:
        internal_pipe = transformers.pipeline("conversational", args.model)

        def pipe(messages):
            return internal_pipe(messages).generated_responses[-1]

    pipe_name = re.sub(r'\W+', '_', args.model).lower()

    with open(args.prompt_config, 'r') as file:
        prompt_config = yaml.safe_load(file)

    with open(args.output_path, 'w') as file:
        for source in args.sources:
            # TODO support more than coco
            generator: Iterator[Context] = coco.COCOLoader(source.value, args.dataset_storage_path)

            generator_wrapper = tqdm(generator, desc="Generating samples")
            for context in generator_wrapper:
                samples: [Sample] = generate_samples(context, pipe, prompt_config, pipe_name)
                for sample in samples:
                    write_str = json.dumps(sample, default=lambda x: x.__dict__)
                    generator_wrapper.set_description(write_str)
                    file.write(write_str + '\n')


if __name__ == "__main__":
    main()
