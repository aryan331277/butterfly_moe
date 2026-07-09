class ButterflyRotation(nn.Module):
    def __init__(self, dim: int, num_layers: int = None):
        super().__init__()
        self.dim = dim
        # Pad dim to next power of two for butterfly ops
        self.padded_dim = 2 ** math.ceil(math.log2(dim)) if dim & (dim - 1) != 0 else dim
        self.num_layers = num_layers or max(int(math.log2(self.padded_dim)), 2)
        self.angles = nn.ParameterList([
            nn.Parameter(torch.randn(self.padded_dim // 2) * 0.02)
            for _ in range(self.num_layers)
        ])
        self.needs_pad = (dim != self.padded_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch, seq_len, dim = x.shape

        # Pad to power of two if needed (handles non-power-of-two dims)
        if self.needs_pad:
            pad_size = self.padded_dim - dim
            x = F.pad(x, (0, pad_size))

        for layer_idx in range(self.num_layers):
            stride = 2 ** (layer_idx + 1)
            x = self._apply_butterfly_layer(x, self.angles[layer_idx], stride)

        # Unpad
        if self.needs_pad:
            x = x[..., :dim]

        if len(original_shape) == 2:
            x = x.squeeze(1)
        return x

    def _apply_butterfly_layer(self, x: torch.Tensor, angles: torch.Tensor, stride: int) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        x_reshaped = x.reshape(batch, seq_len, dim // stride, stride)
        half_stride = stride // 2
        num_groups = dim // stride
        cos_angles = torch.cos(angles[:dim // 2]).reshape(num_groups, half_stride)
        sin_angles = torch.sin(angles[:dim // 2]).reshape(num_groups, half_stride)
        x1 = x_reshaped[..., :half_stride]
        x2 = x_reshaped[..., half_stride:]
        y1 = cos_angles * x1 - sin_angles * x2
        y2 = sin_angles * x1 + cos_angles * x2
        y = torch.cat([y1, y2], dim=-1)
        return y.reshape(batch, seq_len, dim)


class BitNetQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        gamma = weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        weight_scaled = weight / gamma
        weight_quant = weight_scaled.round().clamp(-1, 1)
        return weight_quant * gamma

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class HO_Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_butterfly_layers: int = None):
        super().__init__()
        self.phi = ButterflyRotation(d_model, num_butterfly_layers)
        self.theta = ButterflyRotation(d_ff, num_butterfly_layers)

    def forward(self, x: torch.Tensor, w_base_quant: torch.Tensor) -> torch.Tensor:
        x_rot = self.phi(x)
        y_base = F.linear(x_rot, w_base_quant)
        y = self.theta(y_base)
        return y
