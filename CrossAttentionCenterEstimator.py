import torch
import torch.nn as nn
import torch.nn.functional as F

class KeypointEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # Embedding layer for keypoints (x, y) -> higher-dimensional space
        self.embedding = nn.Linear(2, embedding_dim)  # Map 2D keypoints to embedding_dim
        #self.batchnorm = nn.BatchNorm1d(embedding_dim)
        
    def forward(self, keypoints):
        return self.embedding(keypoints)

class CrossAttentionTransformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super().__init__()
        
        # Multi-head attention layer with batch_first=True
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        
        # Feed-forward layer after attention
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Number of transformer layers (stacked attention blocks)
        self.num_layers = num_layers
        
    def forward(self, keypoints_rgb_embedded, keypoints_thermal_embedded, mask):
        for _ in range(self.num_layers):
            # query, key, value, mask
            attended_rgb, attention_weights = self.attention(keypoints_rgb_embedded, keypoints_thermal_embedded, keypoints_thermal_embedded, key_padding_mask=mask)
    
            # Add residual connection and layer normalization
            attended_rgb = self.layer_norm(attended_rgb + keypoints_rgb_embedded)
            
            # Feed-forward layer
            attended_rgb = self.ffn(attended_rgb)

            # layer norm again.
            attended_rgb = self.layer_norm(attended_rgb + keypoints_rgb_embedded)

        return attended_rgb, attention_weights


class CenterPredictor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        
        # A simple fully connected layer to predict center coordinates
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 2)  # we're predicting 2D coordinates (x, y)
        )
    
    def forward(self, attended_keypoints):
        # Predict center by passing through the fully connected layers
        estimated_centers = self.fc(attended_keypoints)
        # get rid of negative centers.
        predicted_centers = estimated_centers #F.relu(estimated_centers)
        return predicted_centers


class CrossAttentionCenterEstimator(nn.Module):
    def __init__(self, embedding_dim = 128, num_heads = 4, num_layers = 2):
        super().__init__()
        # combine each block in final architecture.
        self.keypoint_embeddings = KeypointEmbedding(embedding_dim)
        self.cross_attention_transformer = CrossAttentionTransformer(embedding_dim, num_heads, num_layers)
        self.center_predictor = CenterPredictor(embedding_dim)

    def forward(self, keypoints_rgb_crop, keypoints_thermal_patch, key_padding_mask):
        embedding_rgb = self.keypoint_embeddings(keypoints_rgb_crop)
        embedding_thermal = self.keypoint_embeddings(keypoints_thermal_patch)
        attended_rgb, _ = self.cross_attention_transformer(embedding_rgb, embedding_thermal, mask = key_padding_mask)
        pooled_attended_rgb = attended_rgb.mean(dim=1)
        return self.center_predictor(pooled_attended_rgb) 