import os
import urllib.request
from zipfile import ZipFile
from pycocotools.coco import COCO
from typing import Iterator
from tqdm import tqdm
from pathlib import Path

from data_definitions import *


class COCOLoader:
    def __init__(self, dataset: str = 'COCO2014', storage_path: str = '.'):
        self.dataset = dataset
        self.data_dir: Path = Path(storage_path) / dataset

        (instance_annotation_file_local_path, captions_annotation_file_local_path) = self.download(dataset)
        self.coco_instances: COCO = COCO(str(instance_annotation_file_local_path))
        self.coco_captions: COCO = COCO(str(captions_annotation_file_local_path))

    @property
    def name(self) -> str:
        return self.dataset

    def download(self, dataset: str) -> Path:
        data_urls = {
            'COCO2014': 'http://images.cocodataset.org/zips/train2014.zip',
            'COCO2017': 'http://images.cocodataset.org/zips/train2017.zip'
        }
        annotation_urls = {
            'COCO2014': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
            'COCO2017': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        }
        instance_annotation_files = {
            'COCO2014': 'instances_train2014.json',
            'COCO2017': 'instances_train2017.json'
        }
        captions_annotation_files = {
            'COCO2014': 'captions_train2014.json',
            'COCO2017': 'captions_train2017.json'
        }

        if dataset not in data_urls:
            raise ValueError(f"{dataset} not found")

        self.data_dir.mkdir(parents=True, exist_ok=True)


        def reporthook(blocknum, blocksize, totalsize):
            if not t.total:
                t.total = totalsize
            t.update(blocknum * blocksize - t.n)

        images_url = data_urls[dataset]
        images_zip_path = self.data_dir / images_url.rsplit('/', 1)[-1]
        images_path = self.data_dir / images_zip_path.stem
        if not images_path.exists():
            if not images_zip_path.exists():
                url = images_url
                print(f"Downloading {url}. This may take a while...")
                with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
                    urllib.request.urlretrieve(url, images_zip_path, reporthook=reporthook)
                print("Download complete.")

            print(f"Extracting files for {images_zip_path}, please wait...")
            with ZipFile(images_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("Extraction complete.")

        annotations_url = annotation_urls[dataset]
        annotations_zip_path = self.data_dir / annotations_url.rsplit('/', 1)[-1]
        annotation_path = self.data_dir / "annotations"
        if not annotation_path.exists():
            if not annotations_zip_path.exists():
                url = annotations_url
                print(f"Downloading {url}. This may take a while...")
                with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
                    urllib.request.urlretrieve(url, annotations_zip_path, reporthook=reporthook)
                print("Download complete.")

            print(f"Extracting files for {annotations_zip_path}, please wait...")
            with ZipFile(annotations_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("Extraction complete.")

        return annotation_path / instance_annotation_files[dataset], annotation_path / captions_annotation_files[dataset]

    def __iter__(self) -> Iterator[Context]:
        def normalize_box(box, sz):
            return [box[0]/sz[0], box[1]/sz[1], box[2]/sz[0], box[3]/sz[1]]

        imgIds = self.coco_captions.getImgIds()
        for imgId in imgIds:
            img = self.coco_captions.loadImgs(imgId)[0]
            img_sz = [img["width"], img["height"]]
            ann_ids_captions = self.coco_captions.getAnnIds(imgIds=img['id'], iscrowd=None)

            anns_captions = self.coco_captions.loadAnns(ann_ids_captions)
            captions = [Caption(caption=ann['caption']) for ann in anns_captions if 'caption' in ann]
            if len(captions) == 0:
                continue

            ann_ids_boxes = self.coco_instances.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns_boxes = self.coco_instances.loadAnns(ann_ids_boxes)
            boxes = [Box(category_name=self.coco_instances.loadCats(ann['category_id'])[0]['name'], bbox=normalize_box(ann['bbox'], img_sz))
                     for ann in anns_boxes if 'bbox' in ann]

            yield Context(sample_id=img['id'], source=self.dataset, captions=captions, boxes=boxes)

    def __len__(self) -> int:
        return len(self.coco_captions.getImgIds())
