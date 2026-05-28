import indexed_gzip
import pickle
import lz4.frame
import json
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent.parent
REPLAY_DATASET_PATH = ROOT_PATH / "metamon" / "replays" / "parsed-replays" / "gen9ou.tar.gz"
GZ_INDEX_PATH = REPLAY_DATASET_PATH.parent / "gen9ou.tar.gz.igz"
METADATA_INDEX_PATH = REPLAY_DATASET_PATH.parent / "gen9ou_meta.pkl"

class ReplayLoader:
    def __init__(self, num_workers=6):
        self.num_workers = num_workers
        self._file_index = None
        self._pbar_lock = threading.Lock()

    @property
    def file_index(self):
        if self._file_index is None:
            print("Loading metadata index...")
            with open(METADATA_INDEX_PATH, "rb") as meta:
                self._file_index = pickle.load(meta)
        return self._file_index
    
    def process_chunk(self, path_chunk, pbar):
        local_data = []
        with indexed_gzip.IndexedGzipFile(str(REPLAY_DATASET_PATH)) as f:
            f.import_index(str(GZ_INDEX_PATH))
            
            for path in path_chunk:
                offset, size = self.file_index[path]
                f.seek(offset)
                
                raw_lz4_bytes = f.read(size)  
                decompressed_bytes = lz4.frame.decompress(raw_lz4_bytes)
                json_data = json.loads(decompressed_bytes.decode('utf-8'))
                local_data.append(json_data)

                with self._pbar_lock:
                    pbar.update(1)
                
        return local_data
    
    def load(self, sample_size: int, target_prefix: str = "") -> list:
        target_paths = [p for p in self.file_index.keys() if p.startswith(target_prefix) and p.endswith('.lz4')]
        print(f"Found {len(target_paths)} files matching target prefix.")

        sample_size = min(sample_size, len(target_paths))
        sampled_paths = random.sample(target_paths, sample_size)
        sampled_paths.sort(key=lambda path: self.file_index[path][0])  # Sort by offset for faster processing
        
        chunk_size = (len(sampled_paths) + self.num_workers - 1) // self.num_workers
        chunks = [sampled_paths[i:i + chunk_size] for i in range(0, len(sampled_paths), chunk_size)]

        print(f"Using {self.num_workers} threads")

        sampled_data = []
        with tqdm(total=len(sampled_paths), desc="Processing replays", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(lambda c: self.process_chunk(c, pbar), chunks))
                
                for chunk_result in results:
                    sampled_data.extend(chunk_result)

        print(f"Successfully loaded {len(sampled_data)} matches.")
        return sampled_data

# Testing
if __name__ == "__main__":
    loader = ReplayLoader()
    data = loader.load(5000, "gen9ou/2024/02/")

    if data:
        with open(ROOT_PATH / "train_imitation" / "sample.json", "w") as f:
            json.dump(data[0], f, indent=4)
    else:
        print("No target files matching prefix found.")