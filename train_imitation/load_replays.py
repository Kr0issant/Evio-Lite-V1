import indexed_gzip
import pickle
import lz4.frame
import json
import random

from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent.parent
REPLAY_DATASET_PATH = ROOT_PATH / "metamon" / "replays" / "parsed-replays" / "gen9ou.tar.gz"
GZ_INDEX_PATH = REPLAY_DATASET_PATH.parent / "gen9ou.tar.gz.igz"
METADATA_INDEX_PATH = REPLAY_DATASET_PATH.parent / "gen9ou_meta.pkl"

target_prefix = "gen9ou/2024/02/"
sample_size = 5

with open(METADATA_INDEX_PATH, "rb") as meta:
    file_index = pickle.load(meta)

target_paths = [p for p in file_index.keys() if p.startswith(target_prefix) and p.endswith('.lz4')]
print(f"Found {len(target_paths)} files matching target prefix.")

sampled_paths = random.sample(target_paths, sample_size)
sampled_data = []

with indexed_gzip.IndexedGzipFile(str(REPLAY_DATASET_PATH)) as f:
    f.import_index(str(GZ_INDEX_PATH))
    
    for path in sampled_paths:
        offset, size = file_index[path]
        f.seek(offset)
        
        raw_lz4_bytes = f.read(size)  
        decompressed_bytes = lz4.frame.decompress(raw_lz4_bytes)
        json_data = json.loads(decompressed_bytes.decode('utf-8'))
        sampled_data.append(json_data)

print(f"Successfully loaded {len(sampled_data)} matches.")

# Testing
with open("data.json", "w") as f:
    json.dump(sampled_data[0], f, indent=4)