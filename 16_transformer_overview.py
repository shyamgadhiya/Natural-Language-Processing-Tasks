# Transformer encoder block (order matters)
# 1. Input embeddings + positional encoding
# 2. Multi-head self-attention
# 3. Add & LayerNorm (residual connection around step 2)
# 4. Position-wise feed-forward network
# 5. Add & LayerNorm (residual connection around step 4)
# Repeat steps 2-5 for N stacked encoder layers
