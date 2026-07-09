class ButterflyQuantFFN(nn.Module):
    def __init__(self, d_model, d_ff, num_butterfly_layers=None):
        super().__init__()
        self.rotation = ButterflyRotation(d_model, num_butterfly_layers)
        self.w1 = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
        self.w2 = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)
 
    def forward(self, x):
        x_rot = self.rotation(x)
        w1_q = BitNetQuantize.apply(self.w1)
        w2_q = BitNetQuantize.apply(self.w2)
        return F.linear(F.gelu(F.linear(x_rot, w1_q)), w2_q)


class HOMoE_TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=512, num_experts=8, top_k=2, dropout=0.1, num_butterfly_layers=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.moe = HOMoE_Layer(d_model, d_ff, num_experts, top_k, num_butterfly_layers)
        self.proj = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, training=True):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        moe_out, aux_loss = self.moe(x, training=training)
        moe_out = self.proj(moe_out)
        x = x + self.dropout(moe_out)
        x = self.norm2(x)
        return x, aux_loss


class StandardMoE_TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=512, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.moe = StandardMoE_Layer(d_model, d_ff, num_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, training=True):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        moe_out, aux_loss = self.moe(x, training=training)
        x = x + self.dropout(moe_out)
        x = self.norm2(x)
        return x, aux_loss


class StandardMoEQuant_TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=512, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.moe = StandardMoEQuant_Layer(d_model, d_ff, num_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x, mask=None, training=True):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        moe_out, aux_loss = self.moe(x, training=training)
        x = x + self.dropout(moe_out)
        x = self.norm2(x)
        return x, aux_loss

class DenseTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,mask=None, training=True):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x, 0
