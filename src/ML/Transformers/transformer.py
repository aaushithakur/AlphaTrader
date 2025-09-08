import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os, sys
base_path = os.getcwd() + '/src'
sys.path.append(base_path)
import numpy as np
from Shared.configs.Transformer.train_config import OUTPUT_SIZE, FP_THRESH

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Linear transformations
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Transpose for matrix multiplication
        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 3, 1)
        queries = queries.permute(0, 2, 1, 3)

        # Calculate the energy without using einsum
        energy = torch.matmul(queries, keys)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Attention
        attention = F.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        # Apply attention to values
        out = torch.matmul(attention, values)

        # Reshape and concatenate heads
        # -1 indicates that the size of that dimension should be inferred based on the 
        # total number of elements in the tensor and the sizes of other dimensions.
        out = out.permute(0, 2, 1, 3).contiguous().view(N, query_len, -1)

        # Linear transformation for output
        out = self.fc_out(out)
        
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 10*embed_size),
            nn.ReLU(),
            nn.Linear(10*embed_size, embed_size),
            nn.Dropout(0.05),
        )

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask=None)
        # Add skip connection and run layer norm
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_size):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(0.05)
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float()
            * (-math.log(10000.0) / embed_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        # return self.dropout(x)
        return x


class TransformerForecast(nn.Module):
    def __init__(self, embed_size, heads, num_layers, max_len, output_size, batch_size):
        super(TransformerForecast, self).__init__()
        self.embed_size = embed_size
        self.pos_encoder = PositionalEncoding(max_len, embed_size)
        self.learnable_embedding = nn.Parameter(torch.ones(1, 1, embed_size), requires_grad=True) # make sure the embedding is learnable
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        learnable_embedding = self.learnable_embedding.expand(batch_size, 1, -1)
        x = torch.cat((x, learnable_embedding),dim=1) # concat on first dimension because 0 is batch size
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, x, x, mask=None)
        x = self.fc_out(x[:, -1, :])  # considering only the last day
        x = self.tanh(x)
        x = (x+1)/2
        return x

# Simple Neural Network model for benchmarking
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 80)  # First hidden layer
        self.fc2 = nn.Linear(80, 30)  # Output layer
        self.fc3 = nn.Linear(30,OUTPUT_SIZE)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = torch.relu(x)  # Apply ReLU activation to the output of the first hidden layer
        x = self.fc2(x)  # Output layer
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = (x+1)/2
        return x

class CustomLoss(nn.Module):
    def __init__(self, weight_fp=1.0):
        super(CustomLoss, self).__init__()
        self.weight_fp = weight_fp

    def forward(self, predicted, target):
        # Binary cross-entropy loss
        bce_loss = nn.BCELoss()(predicted, target)
        # Calculate the number of false positives
        # fp = torch.sum((predicted > FP_THRESH) & (target == 0))
        fn = torch.sum((predicted < FP_THRESH) & (target == 1))
        # Add penalty for false positives
        penalty = self.weight_fp * fn
        # Combine the losses
        total_loss = bce_loss + penalty
        return total_loss