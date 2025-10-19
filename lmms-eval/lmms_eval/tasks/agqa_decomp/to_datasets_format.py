import shutil

import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset as torchDataset
import datasets
import json
import os
import yaml
from tqdm import tqdm


class AGQADataset(torchDataset):
    def __init__(self, agqa_root, split, subset_size=None):
        self.agqa_root = Path(agqa_root)
        self.split = split
        self.question_file = {
            'train': self.agqa_root / 'AGQA_balanced/train_balanced.txt',
            'test': self.agqa_root / 'AGQA_balanced/test_balanced.txt'
        }[split]
        self.hierarchy_file = {
            'train': self.agqa_root / 'AGQA-decomp-hierarchies/train_hierarchies',
            'test': self.agqa_root / 'AGQA-decomp-hierarchies/test_hierarchies'
        }[split]
        self.load_questions()
        self.load_hierarchies()
        self.idx_to_keys = {
            i: k for i, k in enumerate(self.hierarchies.keys())
        }

        full_size = len(self.questions)
        if subset_size is not None:
            if not isinstance(subset_size, list):
                subset_size = [subset_size]
            self.subset_size = [int(len(self.questions) * s) if 0 <= s <= 1 else int(s) for s in subset_size]
            self.subset_size = [s for s in self.subset_size if s <= full_size]
        else:
            self.subset_size = None
        print(f'Selected subset_size: {self.subset_size}')

    def load_questions(self):
        self.questions = json.load(self.question_file.open('r'))

    def load_hierarchies(self):
        self.hierarchies = {
            k: {**v, "video_id": vid.stem, 'ans_type': self.questions[k]['ans_type'], 'main_key': k}
            for vid in tqdm(self.hierarchy_file.iterdir(), f'Loading {self.split} hierarchies')
            if vid.stem != 'isolated'
            for k, v in json.load(vid.open('r')).items()
        }

    def __len__(self):
        return 100

    def __getitem__(self, item):
        data_sample = self.hierarchies[self.idx_to_keys[item]]
        return data_sample

    def to_hf_dataset(self):
        print('Translating to hf dataset')
        hier_list = [{**v, 'question_id': k} for k, v in self.hierarchies.items()]
        hier_df = pd.DataFrame(hier_list)
        print('Serializing hierarchy and subquestions')
        hier_df['hierarchy'] = hier_df['hierarchy'].map(lambda x: json.dumps(x))
        hier_df['subquestions'] = hier_df['subquestion'].map(lambda x: json.dumps(x))
        del hier_df['subquestion']
        print('Constructing hf dataset')
        hf_ds = datasets.Dataset.from_pandas(hier_df)

        dataset_dicts = {self.split: hf_ds}

        if self.subset_size is not None:
            for size in self.subset_size:
                size_str = str(size) if size < 1000 else f'{size // 1000}k'
                print(f'Constructing {size_str} subset')
                subset = hf_ds.shuffle(seed=42).select(range(size))
                tag = f'{self.split}_{size_str}'
                dataset_dicts[tag] = subset

        return dataset_dicts

    @staticmethod
    def save_load_script(save_dir):
        print('Saving load script')
        load_script_path = os.path.join(os.path.dirname(__file__), '_load_script.py')
        save_fname = os.path.basename(os.path.dirname(os.path.join(save_dir, 'xxx.xx'))) + '.py'
        save_path = os.path.join(save_dir, save_fname)
        shutil.copy(load_script_path, save_path)



if __name__ == "__main__":
    save_dir = '/mnt/workspace/liaozhaohe/datasets/VideoDatasets/AGQA-Decomp/hf_hierarchies/AGQA-Decomp'
    AGQADataset.save_load_script(save_dir)

    test_dataset = AGQADataset('/mnt/workspace/liaozhaohe/datasets/VideoDatasets/AGQA-Decomp', 'test',
                               subset_size=[5, 100, 1000, 2000, 4000, 8000, 10000, 50000, 100_000, 300_000, 1_000_000])
    test_hf_ds = test_dataset.to_hf_dataset()
    print('Saving test hf datasets')
    for split, ds in test_hf_ds.items():
        print(f'Split: {split}, len(ds): {len(ds)}')
        ds.save_to_disk(f'/mnt/workspace/liaozhaohe/datasets/VideoDatasets/AGQA-Decomp/hf_hierarchies/backups/{split}')

    train_dataset = AGQADataset('/mnt/workspace/liaozhaohe/datasets/VideoDatasets/AGQA-Decomp', 'train')
    train_hf_ds = train_dataset.to_hf_dataset()
    print('Saving train hf datasets')
    for split, ds in train_hf_ds.items():
        print(f'Split: {split}, len(ds): {len(ds)}')
        ds.save_to_disk(f'/mnt/workspace/liaozhaohe/datasets/VideoDatasets/AGQA-Decomp/hf_hierarchies/backups/{split}')

    full_datasets = {**test_hf_ds, **train_hf_ds}
    print('constructing data dict')
    data_dict = datasets.DatasetDict(full_datasets)
    print('saving data dict')
    save_dir = '/mnt/workspace/liaozhaohe/datasets/VideoDatasets/AGQA-Decomp/hf_hierarchies/AGQA-Decomp'
    data_dict.save_to_disk(save_dir)
    AGQADataset.save_load_script(save_dir)

