import dataclasses
import glob
import json
import os.path
import re
import urllib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Iterator

from tqdm import tqdm

from data_definitions import *

file_urls = [
    "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_captions.jsonl",
    # "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00000-of-00010.jsonl",
    # "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00001-of-00010.jsonl",
    # "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00002-of-00010.jsonl",
    # "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00003-of-00010.jsonl",
    # "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00004-of-00010.jsonl",
    # "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00005-of-00010.jsonl",
    # "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00006-of-00010.jsonl",
    # "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00007-of-00010.jsonl",
    # "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00008-of-00010.jsonl",
    # "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00009-of-00010.jsonl",
    "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv",
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
]


class OpenImagesLoader:
    def __init__(self, storage_path: str = '.'):
        self.data_dir: Path = Path(storage_path) / self.name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.download()

        self.class_names: Dict[str, str] = self._generate_class_name_lookup()
        self.captions: Dict[str, List[Caption]] = self._generate_captions()
        self.boxes: Dict[str, List[Any]] = self._generate_boxes()
        self.length = len(self.captions)

    def _generate_class_name_lookup(self) -> Dict[str, str]:
        file_path = self.data_dir / "oidv7-class-descriptions-boxable.csv"
        with open(file_path) as f:
            data_iterator = map(lambda x: x.split(","), f)
            res = {x[0]: x[1].strip().lower() for x in data_iterator}

        return res

    def _generate_boxes(self) -> Dict[str, List[Any]]:
        boxes = defaultdict(list)

        file_path = self.data_dir / "oidv6-train-annotations-bbox.csv"
        with open(file_path) as f:
            labels = f.readline().split(",")

            def normalize_box(xMin, xMax, yMin, yMax):
                return [xMin, yMin, xMax, yMax]

            def extract_box(line):
                data = line.split(",")
                return (data[0], Box(
                    category_name=self.class_names[data[2]],
                    confidence=float(data[3]),
                    bbox=normalize_box(*[float(data[4]), float(data[5]), float(data[6]), float(data[7])]),
                    is_occluded=bool(int(data[8])),
                    is_truncated=bool(int(data[9])),
                    is_group_of=bool(int(data[10])),
                    is_depiction=bool(int(data[11])),
                    is_inside=bool(int(data[12])),
                ))

            for line in f:
                image_id, data = extract_box(line)
                boxes[image_id].append(data)

        return boxes

    def _generate_captions(self) -> Dict[str, List[Caption]]:
        def extract_caption(json_line: str) -> Tuple[str, str]:
            pattern = r'"image_id": "([^"]+)"|"caption": "([^"]+)"'
            matches = re.findall(pattern, json_line)

            # Extracting values from matches
            image_id, caption = None, None
            for match in matches:
                if match[0]:  # if the first group (image_id) is not empty
                    image_id = match[0]
                elif match[1]:  # if the second group (caption) is not empty
                    caption = match[1]

            return image_id, caption

        captions: Dict[str, List[Caption]] = defaultdict(list)
        file_path = self.data_dir / "open_images_train_v6_captions.jsonl"
        with open(file_path) as f:
            data_iterator = map(lambda x: extract_caption(x), f)
            for image_id, caption in data_iterator:
                captions[image_id].append(Caption(caption=caption))

        # for file_path in glob.glob(str(self.data_dir / "open_images_train_v6_localized_narratives*.jsonl")):
        #     with open(file_path) as f:
        #         data_iterator = map(lambda x: extract_caption(x), f)
        #         captions = captions | {data[0]: data[1] for data in data_iterator}

        return captions

    @property
    def name(self) -> str:
        return "open_images"

    def download(self) -> None:
        def reporthook(blocknum, blocksize, totalsize):
            if not t.total:
                t.total = totalsize
            t.update(blocknum * blocksize - t.n)

        for url in file_urls:
            file_path = self.data_dir / Path(url).name
            if not os.path.exists(file_path):
                print(f"Downloading {url}. This may take a while...")
                with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
                    urllib.request.urlretrieve(url, self.data_dir / Path(url).name, reporthook=reporthook)
                print("Download complete.")

    def __iter__(self) -> Iterator[Context]:
        for image_id, captions in self.captions.items():
            boxes = self.boxes.get(image_id, [])

            yield Context(sample_id=image_id, source=self.name, captions=captions, boxes=boxes)

    def __len__(self) -> int:
        return self.length
