import os
import zipfile
from enum import Enum
from pathlib import Path
from typing import List, Iterator

import pandas as pd
import requests
from itertools import combinations

from data_definitions import Context, Caption

URL_INDEX_CSV = "https://raw.githubusercontent.com/EricGuo5513/HumanML3D/main/index.csv"
URL_TEXTS_ZIP = "https://github.com/EricGuo5513/HumanML3D/raw/main/HumanML3D/texts.zip"


class HumanML3DLoader:
    class HumanML3DType(Enum):
        NONE = 0
        CMU = "CMU"
        KIT = "KIT"
        HDM05 = "HDM05"

        @classmethod
        def ALL(clz) -> List:
            return [clz.CMU, clz.KIT, clz.HDM05]

    def __init__(self, storage_path: str = '.', sub_datasets: List[HumanML3DType] = HumanML3DType.ALL()):
        self.data_dir: Path = Path(storage_path) / self.name
        self.texts_path = self.data_dir / "texts"

        self.download_and_unzip(URL_TEXTS_ZIP, self.data_dir)

        df = pd.read_csv(URL_INDEX_CSV)

        pattern_allowed = f"{'|'.join([tp.name for tp in sub_datasets])}"
        filter = df['source_path'].str.contains(pattern_allowed)
        df = df[filter]
        df["new_name"] = df["new_name"].apply(lambda x: os.path.splitext(x)[0])

        self.index = df
        self.sub_datasets = sub_datasets

        self.blacklist = {
            "000604", "000768", "005631", "009616", "010125", # HDM_dg_03-04_03
            "006067", "014169", # HDM_mm_05-02_03
            "000037", "002692" # CMU/144/144_08_
        }

    @property
    def name(self) -> str:
        return "humanml3d"

    @staticmethod
    def download_and_unzip(url: str, path: Path):
        # Extract filename from URL
        filename = url.split('/')[-1]
        # Define the directory name by removing the .zip extension
        dir_name = filename.replace('.zip', '')
        # Full path to the directory where we want to extract the contents
        full_dir_path = path / dir_name

        # Check if the directory already exists
        if not os.path.exists(full_dir_path):
            # Directory does not exist, proceed with download and unzip
            # Download the zip file
            print(f"Downloading {filename}...")
            resp = requests.get(url)
            zip_path = path / filename
            path.mkdir(parents=True, exist_ok=True)
            with open(zip_path, 'wb') as f:
                f.write(resp.content)
            print(f"Downloaded {filename} to {zip_path}")

            # Unzip the file
            print(f"Unzipping {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(path)
            print(f"Extracted to {full_dir_path}")

            # Optionally, remove the zip file after extraction
            os.remove(zip_path)
            print(f"Removed {zip_path}")
        else:
            # Directory exists, no need to download or unzip
            print(f"Directory {full_dir_path} already exists. Skipping download and unzip.")

    def __iter__(self) -> Iterator[Context]:
        for k, el in self.index.iterrows():
            motion_id = el["new_name"]
            if motion_id in self.blacklist:
                print(f"Skipping motion {motion_id} (blacklist) {el}")
                continue
            if el["end_frame"] != -1 and (el["end_frame"] - el["start_frame"]) < 5:
                print(f"Skipping motion {motion_id} (too short) {el}")
                continue

            with open(self.texts_path / f"{motion_id}.txt") as file:
                raw_captions = [x[: x.index("#")] for x in file.readlines()]

            captions = [Caption(caption=x) for x in raw_captions]
            if len(captions) == 0:
                continue

            yield Context(sample_id=motion_id, source=self.name, captions=captions, boxes=[])

    def __len__(self) -> int:
        return len(self.index)

class HumanML3DNGramLoader(HumanML3DLoader):

    def __init__(self, ngrams: List[int] = None, *vargs, **kwargs):
        super().__len__(*vargs, **kwargs)

        if not ngrams:
            ngrams = [2, 3, 4]
        self.index_ngram = HumanML3DNGramLoader.generate_ngrams_dataframe(self.index)

    def __iter__(self) -> Iterator[Context]:
        for k, el in self.index_ngram.iterrows():
            motion_ids = el["new_names"]

            # if motion_id in self.blacklist:
            #     print(f"Skipping motion {motion_id} (blacklist) {el}")
            #     continue
            # if el["end_frame"] != -1 and (el["end_frame"] - el["start_frame"]) < 5:
            #     print(f"Skipping motion {motion_id} (too short) {el}")
            #     continue

            #
            # with open(self.texts_path / f"{motion_id}.txt") as file:
            #     raw_captions = [x[: x.index("#")] for x in file.readlines()]
            #
            # captions = [Caption(caption=x) for x in raw_captions]
            # if len(captions) == 0:
            #     continue
            #
            # yield Context(sample_id=motion_id, source=self.name, captions=captions, boxes=[])

    @staticmethod
    def merge_intervals_with_names(intervals):
        """Merge overlapping intervals while keeping track of names."""
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = []
        for interval in sorted_intervals:
            if not merged or merged[-1][1] < interval[0] - 1:
                merged.append([interval[0], interval[1], [interval[2]]])
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
                merged[-1][2].append(interval[2])
        return merged

    @staticmethod
    def count_ngrams_for_source_path(merged_intervals_with_names, ngrams: List[int]):
        """Count ngrams for a single source path."""
        counts = {n: 0 for n in ngrams}
        for merged_interval in merged_intervals_with_names:
            num_names = len(merged_interval[2])
            for n in ngrams:
                if num_names >= n:
                    counts[n] += len(list(combinations(merged_interval[2], n)))
        return counts

    @staticmethod
    def count_ngrams_in_dataframe(df: pd.DataFrame, ngrams: List[int]):
        """Count the number of ngrams for the specified lengths in the DataFrame."""
        grouped = df.groupby('source_path')
        total_ngram_counts = {n: 0 for n in ngrams}

        for _, group in grouped:
            intervals_with_names = group[['start_frame', 'end_frame', 'new_name']].to_records(index=False)
            merged_intervals_with_names = HumanML3DNGramLoader.merge_intervals_with_names(intervals_with_names)
            ngram_counts = HumanML3DNGramLoader.count_ngrams_for_source_path(merged_intervals_with_names, ngrams)

            for n in ngrams:
                total_ngram_counts[n] += ngram_counts[n]

        return total_ngram_counts

    @staticmethod
    def generate_ngrams_dataframe(df, ngrams):
        """Generate a DataFrame with all the ngrams for the specified lengths."""
        grouped = df.groupby('source_path')
        results = []

        for source_path, group in grouped:
            intervals_with_names = group[['start_frame', 'end_frame', 'new_name']].to_records(index=False)
            merged_intervals_with_names = HumanML3DNGramLoader.merge_intervals_with_names(intervals_with_names)

            for merged_interval in merged_intervals_with_names:
                for n in ngrams:
                    if len(merged_interval[2]) >= n:
                        for names_combo in combinations(merged_interval[2], n):
                            start_frame, end_frame = merged_interval[0], merged_interval[1]
                            results.append([source_path, start_frame, end_frame, n, list(names_combo)])

        ngrams_df = pd.DataFrame(results, columns=['source_path', 'start_frame', 'end_frame', 'ngram', 'new_names'])
        return ngrams_df