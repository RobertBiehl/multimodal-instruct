import json
import random
from collections import defaultdict

from data_definitions import Sample


class JSONLWriter:

    def __init__(self, file_path):
        self.file = open(file_path, 'w')

    def write(self, sample: Sample):
        write_str = json.dumps(sample, default=lambda x: x.__dict__)
        self.file.write(write_str + '\n')

    def close(self):
        self.file.close()

    def __del__(self):
        self.close()


class JSONLLaVAInstructWriter:
    def __init__(self, file_path: str, modality_key: str = "image"):
        self.file_path = file_path
        self.modality_key = modality_key
        self.data = {}

    def write(self, sample: Sample):
        if sample.id not in self.data:
            self.data[sample.id] = {
                "id": sample.id,
                self.modality_key: sample.image,
                "conversations": [
                    {"from": "human", "value":
                     sample.instruction + "\n<image>" if random.randint(0, 1) == 0 else "<image>\n" + sample.instruction
                     },
                    {"from": "gpt", "value": sample.response},
                ]
            }
        else:
            lst = self.data[sample.id]["conversations"]
            lst.append({"from": "human", "value": sample.instruction})
            lst.append({"from": "gpt", "value": sample.response})

    def close(self):
        json_str = json.dumps(list(self.data.values()))
        json_str = json_str.replace('{"id":', '\n{"id":')
        json_str = json_str.replace('"from":', '\n"from":')

        with open(self.file_path, 'w') as f:
            f.write(json_str)

    def __del__(self):
        self.close()
