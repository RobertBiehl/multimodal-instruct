from enum import Enum


class Source(Enum):
    COCO2014 = "COCO2014"
    COCO2017 = "COCO2017"
    OPENIMAGESV7 = "OPENIMAGESV7"

    def __str__(self):
        return self.value

class OutputFormat(Enum):
    JSONL = "jsonl"
    LLAVA_INSTRUCT = "llava-instruct"

    def __str__(self):
        return self.value


class ContextProperties(Enum):
    CAPTIONS = "captions"
    BOXES = "boxes"

    def __str__(self):
        return self.value


class QuestionContext(Enum):
    NONE = "none"
    CAPTIONS = "captions"
    BOXES = "boxes"

    def __str__(self):
        return self.value


class ModelSourceType(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    LLAMA_CPP = "llama.cpp"

    def __str__(self):
        return self.value
