# import torch
# import torch.nn as nn

# d_model = 64
# head = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

# # Two input vectors (batch size = 1)
# v1 = torch.rand(1, 1, d_model)
# v2 = torch.rand(1, 1, d_model)

# # Stack them as a 2-token sequence: shape [1, 2, d_model]
# x = torch.cat([v1, v2], dim=1)

# # Apply self-attention
# attn_output, attn_weights = head(x, x, x)

import torch
import torch.nn as nn

class SignatureSelfAttentionFusion(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, dropout=0.1, use_cls_token=False):
        super(SignatureSelfAttentionFusion, self).__init__()
        self.use_cls = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, D] = batch of N signature embeddings
        B, N, D = x.shape

        if self.use_cls:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
            x = torch.cat([cls_tokens, x], dim=1)          # [B, N+1, D]

        # Self-attention
        attn_output, attn_weights = self.attn(x, x, x)     # [B, N(+1), D]

        x = self.dropout(self.norm(attn_output + x))       # Add & Norm

        if self.use_cls:
            return x[:, 0], attn_weights                   # [B, D]
        else:
            return x.mean(dim=1), attn_weights             # [B, D]


# Signature embeddings from encoder
sig1 = torch.rand(32, 128)  # [B, D]
sig2 = torch.rand(32, 128)  # [B, D]

# Stack to [B, 2, D]
sig_pair = torch.stack([sig1, sig2], dim=1)
# samples, labels = "/Users/balaji.raok/Documents/online_signature_work/Hindi_sign_db/Genuine_subset/"
# print(len(samples), labels)

# Fuse
fuser = SignatureSelfAttentionFusion(embedding_dim=128, use_cls_token=True)
print(fuser)
fused_vector, attn_weights = fuser(sig_pair)  # [B, 128]
print(fused_vector)



sig1 = torch.rand(1, 128)  # signature 1
sig2 = torch.rand(1, 128)  # signature 2
sig_pair = torch.stack([sig1, sig2], dim=1)  # [1, 2, 128]

fuser = SignatureSelfAttentionFusion(embedding_dim=128)
fused_vector, attn_weights = fuser(sig_pair)  # attn_weights: [1, 2, 2]


import matplotlib.pyplot as plt
import seaborn as sns
import numpy 
# Convert to numpy
attn = attn_weights[0].detach().cpu().numpy()  # [2, 2]

# Labels (optional)
labels = ['Signature 1', 'Signature 2']

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(attn, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Attention Weight'}, square=True)

plt.title('Self-Attention Between Signatures')
plt.xlabel('Attended To')
plt.ylabel('Query')
plt.tight_layout()
plt.show()


