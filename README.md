# Multimodal Instruct

Instruction tuning dataset generation inspired by LLaVA-Instruct-158k via any LLM.

Goals:
1. become independent of LLaVA-Instruct-158k which cannot be used commercially.
2. Add more datasets, like OpenImages to overcome limited object class range in Coco.
3. Add more modalities other than images.

## Work in progress
- [ ] improve inference speed
- [ ] fully process with LLAMA-2 for first commercially usable version for LLaVA training.
- [ ] add LICENSE information to generated files
- [ ] add support for vision instruct dataset creation from OpenImages v7 dataset (captions, boxes, positive and negative image labels, etc.)
- [ ] add support for motion data instruction dataset creation (e.g. from HumanML3D)

## Usage Examples

### Generate dataset with huggingface chat model
```python generate.py COCO2014 --model_source huggingface --model meta-llama/Llama-2-7b-chat-hf```

### Generate dataset with adjusted prompts for sub 4096 token context model
```python generate.py COCO2014 --model_source huggingface --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt_config prompt_config_llava_smallcontext.yaml```

### Generate dataset with openai API
```python generate.py COCO2014 --model_source openai --model gpt-3.5-turbo```

## Notes
* Huggingface chat models: only supports models with chat templates in `tokenizer_config.json`