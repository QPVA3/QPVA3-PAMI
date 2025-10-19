"""AGQA Decomp dataset loading script."""

import json
import datasets
from pathlib import Path
import os


class AGQADecompDataset(datasets.GeneratorBasedBuilder):
    """AGQA Decomp dataset loader."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        # Define features based on the AGQA decomp dataset structure
        features = datasets.Features({
            "question": datasets.Value("string"),
            "type": datasets.Value("string"),
            "old_program": datasets.Value("string"),
            "new_program": datasets.Value("string"),
            "hierarchy": datasets.Value("string"),  # JSON string
            "answer": datasets.Value("string"),
            "video_id": datasets.Value("string"),
            "ans_type": datasets.Value("string"),
            "main_key": datasets.Value("string"),
            "question_id": datasets.Value("string"),
            "subquestions": datasets.Value("string"),  # JSON string
        })

        return datasets.DatasetInfo(
            features=features,
            supervised_keys=None,
            description="AGQA Decomp dataset with question hierarchies",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # When load_dataset is called with a local path, the actual data stays in the original location
        # We need to find that location, not the cache directory where the script is copied
        
        # The path passed to load_dataset() is stored in self.config.data_files or self.config.path
        data_dir = None
        
        # Try to get the original dataset path from config
        if hasattr(self.config, 'path') and self.config.path:
            data_dir = Path(self.config.path)
        elif hasattr(self.config, 'data_dir') and self.config.data_dir:
            data_dir = Path(self.config.data_dir)
        
        # If not found in config, try to extract from the name
        # The name often contains the path when loaded locally
        if data_dir is None and hasattr(self.config, 'name'):
            # The config name might be the full path for local datasets
            potential_path = Path(self.config.name)
            if potential_path.exists() and potential_path.is_dir():
                data_dir = potential_path
        
        # Last resort: look for common dataset paths
        if data_dir is None:
            # Try the standard AGQA path
            standard_path = Path('/mnt/workspace/liaozhaohe/datasets/VideoDatasets/AGQA-Decomp/hf_hierarchies/AGQA-Decomp')
            if standard_path.exists():
                data_dir = standard_path
        
        if data_dir is None:
            raise ValueError(
                "Could not determine dataset directory. "
                "Please use datasets.load_from_disk() instead of datasets.load_dataset() "
                "for datasets saved with DatasetDict.save_to_disk()"
            )
        
        # Read the dataset_dict.json to get all split names
        dataset_dict_path = data_dir / "dataset_dict.json"
        
        splits = []
        
        if dataset_dict_path.exists():
            # Load from DatasetDict format
            with open(dataset_dict_path, "r") as f:
                dataset_info = json.load(f)
            
            # DatasetDict saves splits as a list in the JSON
            for split_name in dataset_info.get("splits", []):
                print(f"Split: {split_name}")
                # Each split is saved in its own directory
                split_dir = data_dir / split_name
                if split_dir.exists():
                    splits.append(
                        datasets.SplitGenerator(
                            name=split_name,
                            gen_kwargs={
                                "data_dir": str(split_dir),
                            },
                        )
                    )
        else:
            # Fallback: look for directories with arrow/parquet files
            for item in data_dir.iterdir():
                if item.is_dir():
                    # Check if this directory contains dataset files
                    arrow_files = list(item.glob("*.arrow"))
                    parquet_files = list(item.glob("*.parquet"))
                    if arrow_files or parquet_files:
                        splits.append(
                            datasets.SplitGenerator(
                                name=item.name,
                                gen_kwargs={
                                    "data_dir": str(item),
                                },
                            )
                        )
        
        return splits

    def _generate_examples(self, data_dir):
        """Yields examples from the saved dataset split."""
        import pyarrow as pa
        import pandas as pd
        
        data_dir = Path(data_dir)
        
        # Look for arrow files (default format for save_to_disk)
        arrow_files = sorted(data_dir.glob("*.arrow"))
        
        if arrow_files:
            # Read from arrow files
            idx_offset = 0
            for arrow_file in arrow_files:
                try:
                    # Read the arrow table - these files use streaming format
                    with open(arrow_file, 'rb') as f:
                        reader = pa.ipc.open_stream(f)
                        table = reader.read_all()
                    
                    # Convert to pandas for easier iteration
                    df = table.to_pandas()
                    
                    for idx, row in df.iterrows():
                        # Convert row to dict and yield
                        example = row.to_dict()
                        # Use a unique key for each example
                        yield idx_offset + idx, example
                    
                    idx_offset += len(df)
                except Exception as e:
                    print(f"Warning: Could not read {arrow_file}: {e}")
                    continue
        else:
            # Fallback to parquet files
            parquet_files = sorted(data_dir.glob("*.parquet"))
            
            if parquet_files:
                idx_offset = 0
                for parquet_file in parquet_files:
                    try:
                        df = pd.read_parquet(parquet_file)
                        for idx, row in df.iterrows():
                            example = row.to_dict()
                            yield idx_offset + idx, example
                        idx_offset += len(df)
                    except Exception as e:
                        print(f"Warning: Could not read {parquet_file}: {e}")
                        continue
            else:
                print(f"Warning: No data files found in {data_dir}")


# Also define aliases that datasets might look for
Builder = AGQADecompDataset