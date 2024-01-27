# Multimodal Instruct

Instruction tuning dataset generation inspired by LLaVA-Instruct-158k via any LLM.

## Goals
1. become independent of LLaVA-Instruct-158k which cannot be used commercially.
2. Add more datasets, like OpenImages to overcome limited object class range in Coco.
3. Add more modalities other than images.

## Sponsoring data generation

I currently plan to create the following datasets:
* equivalent to LLaVA-Instruct-158k on COCO dataset using Llama2 70b and Mixtral 8x7B
* a more powerful instruction dataset on Open Images V7 including localized narratives, bounding boxes with metadata, image level labels and object relationships.
* adding more data sources to COCO and Opent Images, specifically

These improved datasets will help multimodal LLM architectures like LLaVA which require pretraining, but even more so architectures like [LaVIN](https://github.com/luogen1996/LaVIN) which only have instruction tuning steps.


## Work in progress
- [x] improve inference speed of OpenAI API with parallel requests
- [x] OpenImages v7 support: captions, boxes
- [ ] OpenImages v7 support for positive and negative image labels in  dataset
- [ ] fully process with Mistral 7b for first commercially usable version for LLaVA training.
- [ ] Add token amount estimation tool for cost estimation
- [ ] add LICENSE information to generated files
- [ ] add support for motion data instruction dataset creation (e.g. from HumanML3D)
- [ ] fully process with LLAMA-2 for first commercially usable version for LLaVA training.
- [ ] improve inference speed of huggingface models with batching
- [ ] improve inference speed of llama.cpp models with batching


# Datasets and  Prompt Configs

To generate a multimodal instruction dataset:
1. pick a dataset
2. pick or set up a prompt config

## Datasets

Available datasets are

1. `Source.COCO2014` and `Source.COCO2017`
   - COCO has been used to generate the LLaVA-Instruct-158k dataset.
   - Provides the following data for instruction dataset generation:
     - captions: 5 sentences by different annotators describing the image
     - object bounding boxes in the format `category_name: [min_x, min_y, max_x, max_y]`
2. `Source.OPENIMAGESV7`
   - Provides the following data for instruction dataset generation:
     - captions: narratives from voice recordings of annotators describing the image in one or more sentences. 
     - object bounding boxes in the format `category_name: [min_x, min_y, max_x, max_y] [confidence, is_occluded, is_truncated, is_group_of, is_depiction, is_inside]`

## Usage Examples

### Generate dataset with huggingface chat model
```python generate.py COCO2014 --model_source huggingface --model meta-llama/Llama-2-7b-chat-hf```

### Generate dataset with adjusted prompts for sub 4096 token context model
```python generate.py COCO2014 --model_source huggingface --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt_config prompt_config_llava_smallcontext.yaml```

### Generate dataset with llama.cpp gguf model
```python generate.py COCO2014 --model_source llama.cpp --model ./PATH/TO/MODEL.gguf```

### Generate dataset with OpenAI API
```python generate.py COCO2014 --model_source openai --model gpt-3.5-turbo```

### Generate dataset with custom OpenAI API endpoint
```python generate.py COCO2014 --model_source openai --model mymodel --openai_base_url BASE_URL```

### Generate OpenImages dataset
```python generate.py OPENIMAGESV7 --prompt_config prompt_config_openimagesv7.yaml ...```

## Notes
* Huggingface chat models: only supports models with chat templates in `tokenizer_config.json`