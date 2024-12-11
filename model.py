import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

def compute_node_degrees(edge_index, num_nodes):
    # Count the degree of each node
    degree = torch.zeros(num_nodes, dtype=torch.float).to(edge_index.device)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)).to(edge_index.device))
    
    # Reshape to (num_nodes, 1) for compatibility with GNNs
    return degree.unsqueeze(1)

class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[256, 512, 1024], output_dim=1024):
        super(GraphEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Apply Xavier initialization
        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):  # For fully connected layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, GCNConv):  # For GCN layers
            nn.init.xavier_uniform_(module.lin.weight)  # GCNConv has a linear layer (lin)
            if module.lin.bias is not None:
                nn.init.zeros_(module.lin.bias)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        x = global_mean_pool(x, batch)  # Pool to get graph embedding
        x = self.output_layer(x)  # Final projection
        return x

class ContrastiveModel(nn.Module):
    def __init__(self, image_encoder, graph_encoder, temperature=0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.graph_encoder = graph_encoder
        self.temperature = temperature
        self.img_projector = nn.Linear(self.image_encoder.config.hidden_size, 1024)

        # Apply Xavier initialization
        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def contrastive_loss(self, img_emb, g_emb, labels, temperature=0.07):
        # Normalize embeddings
        img_emb = F.normalize(img_emb, dim=-1)
        g_emb = F.normalize(g_emb, dim=-1)

        # Compute similarity matrix (B, B)
        similarity = torch.matmul(img_emb, g_emb.T) / temperature  # Scale by temperature

        # Create targets: all zeros except for diagonal positions indicated by labels
        batch_size = similarity.size(0)
        targets = torch.zeros_like(similarity, device=similarity.device)  # Initialize targets to zeros
        for i in range(batch_size):
            if labels[i] == 1:  # If the label is 1, set the diagonal element to 1
                targets[i, i] = 1

        # Apply softmax over similarity rows
        similarity_softmax = F.log_softmax(similarity, dim=1)  # Log-softmax for probabilities

        # Compute loss: negative log-likelihood for positive pairs
        loss = -torch.sum(targets * similarity_softmax) / batch_size  # Average over batch

        return loss
    '''
    def contrastive_loss(self, img_emb, g_emb, labels, temperature=0.07):
        # Normalize embeddings
        img_emb = F.normalize(img_emb, dim=-1)
        g_emb = F.normalize(g_emb, dim=-1)

        # Compute similarity matrix diagonal (self-similarities)
        similarity_diag = torch.sum(img_emb * g_emb, dim=-1) / temperature  # (batch_size,)

        # Use binary_cross_entropy_with_logits for stability
        loss = F.binary_cross_entropy_with_logits(similarity_diag, labels.float())

        return loss
    '''
    def forward(self, images, adjacency_lists, labels):
        swin_output = self.image_encoder(images)
        img_emb = swin_output.last_hidden_state.mean(dim=1)  # Pooling to (B, D)
        img_emb = self.img_projector(img_emb)
        #img_emb = F.normalize(img_emb, dim=-1)

        # Process each graph independently
        graph_embeddings = []
        for adjacency_list in adjacency_lists:
            num_nodes = max(max(u, v) for u, v in adjacency_list) + 1
            edge_index = adjacency_list.to(dtype=torch.long, device=img_emb.device).t().contiguous()
            node_feats = compute_node_degrees(edge_index, num_nodes).to(img_emb.device)
            batch = torch.zeros(num_nodes, dtype=torch.long, device=img_emb.device) # Same batch for all nodes

            # Compute graph embedding
            g_emb = self.graph_encoder(node_feats, edge_index, batch)
            graph_embeddings.append(g_emb.squeeze(0))

        # Stack graph embeddings to match the image embedding batch dimension
        g_emb = torch.stack(graph_embeddings, dim=0)  # (B, D)
        #g_emb = F.normalize(g_emb, dim=-1)

        # Contrastive Loss
        loss = self.contrastive_loss(img_emb, g_emb, labels)

        return loss, img_emb, g_emb


    def inference(self, images, adjacency_lists):
        # Image Encoder
        swin_output = self.image_encoder(images)
        img_emb = swin_output.last_hidden_state.mean(dim=1)  # Pooling to (B, D)
        img_emb = self.img_projector(img_emb)
        img_emb = F.normalize(img_emb, dim=-1)
        
        # Process each graph independently
        graph_embeddings = []
        for adjacency_list in adjacency_lists:
            num_nodes = max(max(u, v) for u, v in adjacency_list) + 1
            edge_index = adjacency_list.to(dtype=torch.long, device=img_emb.device).t().contiguous()
            node_feats = compute_node_degrees(edge_index, num_nodes).to(img_emb.device)
            batch = torch.zeros(num_nodes, dtype=torch.long, device=img_emb.device)

            # Compute graph embedding
            g_emb = self.graph_encoder(node_feats, edge_index, batch)
            graph_embeddings.append(g_emb.squeeze(0))
            
        # Stack graph embeddings to match the image embedding batch dimension
        g_emb = torch.stack(graph_embeddings, dim=0)  # (B, D)
        g_emb = F.normalize(g_emb, dim=-1)
        # Cosine Similarity
        similarity = torch.sum(img_emb * g_emb, dim=-1)  # (B,)

        print(img_emb)
        print(g_emb)

        # Binary Classification (threshold=0.5)
        is_match = (similarity > 0.5).long()

        print(similarity)

        return similarity, is_match

