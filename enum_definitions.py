from enum import Enum

class Source(Enum):
    COCO2014 = "COCO2014"
    COCO2017 = "COCO2017"

class OutputFormat(Enum):
    JSONL = "jsonl"
    LLAVA_INSTRUCT = "llava-instruct"

class ContextProperties(Enum):
    CAPTIONS = "captions"
    BOXES = "boxes"

class QuestionContext(Enum):
    NONE = "none"
    CAPTIONS = "captions"
    BOXES = "boxes"
