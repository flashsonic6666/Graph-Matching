import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"  # Adjust as needed

import json
import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from molscribe import MolScribe
from dataset_utils import molfile_to_bonds_manual, preprocess_smiles
from concurrent.futures import ProcessPoolExecutor

csv_file = "pubchem/train_80k.csv"
image_folder = "images"
batch_size = 256

# Load data
data = pd.read_csv(csv_file, usecols=['pubchem_cid', 'SMILES'])
entries = []
for _, row in data.iterrows():
    cid = row['pubchem_cid']
    gt_smiles = row['SMILES']
    image_path = os.path.join(image_folder, f"{cid}.png")
    if os.path.exists(image_path):
        entries.append((cid, gt_smiles, image_path))

num_gpus = 3
total_entries = len(entries)
entries_per_gpu = (total_entries + num_gpus - 1) // num_gpus

subsets = [entries[i:i+entries_per_gpu] for i in range(0, total_entries, entries_per_gpu)]

ckpt_path = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth')

def process_subset(subset, gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    model = MolScribe(ckpt_path, device=device)

    records = []
    processed_entries_0 = 0
    processed_entries_1 = 0

    for i in range(0, len(subset), batch_size):
        batch = subset[i:i+batch_size]
        batch_images = [e[2] for e in batch]
        try:
            predictions = model.predict_image_files(batch_images)
        except:
            continue
        for (cid, gt_smiles, image_path), pred in zip(batch, predictions):
            pred_smiles = pred.get('smiles', '')
            molfile = pred.get('molfile', '')
            bonds = molfile_to_bonds_manual(molfile)
            if bonds is None or len(bonds) == 0:
                continue
            label = 1 if preprocess_smiles(pred_smiles) == preprocess_smiles(gt_smiles) else 0

            bonds_json = json.dumps(bonds)
            records.append({
                'cid': cid,
                'image_path': image_path,
                'ground_truth_smiles': gt_smiles,
                'predicted_smiles': pred_smiles,
                'bonds': bonds_json,
                'label': label
            })

            if label == 0:
                processed_entries_0 += 1
            else:
                processed_entries_1 += 1

    return records, processed_entries_0, processed_entries_1


if __name__ == "__main__":
    results = []
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_id, subset in enumerate(subsets):
            futures.append(executor.submit(process_subset, subset, gpu_id))

        for fut in tqdm(futures, desc="Processing"):
            res = fut.result()
            results.append(res)

    # Combine all results
    combined_records = []
    total_0 = 0
    total_1 = 0
    for recs, c0, c1 in results:
        combined_records.extend(recs)
        total_0 += c0
        total_1 += c1

    # Save combined results
    df_out = pd.DataFrame(combined_records)
    df_out.to_csv("dataset_with_labels.csv", index=False)

    with open('dataset_with_labels.json', 'w') as f:
        json.dump(combined_records, f, indent=4)

    with open('dataset_stats.txt', 'w') as f:
        f.write(f"Processed {total_0} entries with label 0 and {total_1} entries with label 1\n")

    print(f"Total processed entries with label 0: {total_0}, label 1: {total_1}")
    print("All done!")
