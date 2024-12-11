from torch.utils.data import Dataset
import os
from PIL import Image
import json
from transformers import AutoFeatureExtractor
import torch

model_name = "microsoft/swin-base-patch4-window12-384-in22k"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    identifiers = [item['cid'] for item in data]
    graph_edges = [item['bonds'] for item in data]
    labels = [item['label'] for item in data]
    return identifiers, graph_edges, labels

class CustomDataset(Dataset):
    def __init__(self, args):
        self.images_folder = args.images_folder
        self.feature_extractor = feature_extractor
        self.identifiers, self.graph_edges, self.labels = self.load_and_filter_data(args.train_file)
        
    def load_and_filter_data(self, train_file):
        identifiers, graph_edges, labels = load_data(train_file)
        
        graph_edges = [json.loads(edge) if isinstance(edge, str) else edge for edge in graph_edges]
        for graph in graph_edges:
            for edge in graph:
                edge.pop()
        labels = [int(label) for label in labels]
        return identifiers, graph_edges, labels

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, idx):
        identifier = self.identifiers[idx]
        graph_edges = self.graph_edges[idx]
        label = self.labels[idx]

        # Load and process image
        image = Image.open(os.path.join(self.images_folder, f'{identifier}.png')).convert('RGB')
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze()

        return {'image': image, 'graph_edges': graph_edges, 'label': label}

import torch

def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)

    graph_edges = [torch.tensor(item['graph_edges'], dtype=torch.long) for item in batch]

    max_edges = max(edge.size(0) for edge in graph_edges)

    # Pad graph edges to the maximum size
    batched_graph_edges = torch.zeros((len(batch), max_edges, 2), dtype=torch.long)
    for i, edge in enumerate(graph_edges):
        batched_graph_edges[i, :edge.size(0), :] = edge

    return {
        'images': images,
        'labels': labels,
        'graph_edges': batched_graph_edges,
    }
