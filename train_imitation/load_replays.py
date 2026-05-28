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

target_prefix = "gen9ou/2024/02/"
sample_size = 5000

with open(METADATA_INDEX_PATH, "rb") as meta:
    file_index = pickle.load(meta)

target_paths = [p for p in file_index.keys() if p.startswith(target_prefix) and p.endswith('.lz4')]
print(f"Found {len(target_paths)} files matching target prefix.")

sampled_paths = random.sample(target_paths, sample_size)
sampled_paths.sort(key=lambda path: file_index[path][0]) # Sort by offset for faster processing
sampled_data = []

NUM_WORKERS = 6
chunk_size = (len(sampled_paths) + NUM_WORKERS - 1) // NUM_WORKERS
chunks = [sampled_paths[i:i + chunk_size] for i in range(0, len(sampled_paths), chunk_size)]

pbar_lock = threading.Lock()

def process_chunk(path_chunk, pbar):
    local_data = []
    with indexed_gzip.IndexedGzipFile(str(REPLAY_DATASET_PATH)) as f:
        f.import_index(str(GZ_INDEX_PATH))
        
        for path in path_chunk:
            offset, size = file_index[path]
            f.seek(offset)
            
            raw_lz4_bytes = f.read(size)  
            decompressed_bytes = lz4.frame.decompress(raw_lz4_bytes)
            json_data = json.loads(decompressed_bytes.decode('utf-8'))
            local_data.append(json_data)

            with pbar_lock:
                pbar.update(1)
            
    return local_data

print(f"Using {NUM_WORKERS} threads")

with tqdm(total=len(sampled_paths), desc="Processing replays", unit="file") as pbar:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(lambda c: process_chunk(c, pbar), chunks))
        
        for chunk_result in results:
            sampled_data.extend(chunk_result)

print(f"Successfully loaded {len(sampled_data)} matches.")

# Testing
with open(ROOT_PATH / "train_imitation" / "sample.json", "w") as f:
    json.dump(sampled_data[0], f, indent=4)