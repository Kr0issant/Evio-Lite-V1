import tarfile
import indexed_gzip
import pickle

from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent.parent
REPLAY_DATASET_PATH = ROOT_PATH / "metamon" / "replays" / "parsed-replays" / "gen9ou.tar.gz"
GZ_INDEX_PATH = REPLAY_DATASET_PATH.parent / "gen9ou.tar.gz.igz"
METADATA_INDEX_PATH = REPLAY_DATASET_PATH.parent / "gen9ou_meta.pkl"

print("Building gzip random-access index (this takes time)...")
f = indexed_gzip.IndexedGzipFile(REPLAY_DATASET_PATH)
f.build_full_index()
f.export_index(GZ_INDEX_PATH)

print("Mapping tar byte offsets...")
tar = tarfile.open(fileobj=f, mode="r:")
file_index = {}

for member in tar:
    if member.isfile():
        file_index[member.name] = (member.offset_data, member.size)

print(f"Indexed {len(file_index)} files. Saving metadata dictionary...")
with open(METADATA_INDEX_PATH, "wb") as out:
    pickle.dump(file_index, out)

print("Done. You can now use these index files for instant fetching.")
f.close()
