import torch
import torch.nn as nn
from mamba_ssm import Mamba


def _reverse_seq(x: torch.Tensor) -> torch.Tensor:
    # x: [B, L, D]
    return torch.flip(x, dims=[1])


class BiMamba(nn.Module):
    """
    简单的 Bi-directional Mamba：前向 Mamba + 反向 Mamba（共享参数或不共享都可）
    这里默认共享同一个 Mamba（更省参数），效果也通常够用。
    """
    def __init__(self, mamba: nn.Module):
        super().__init__()
        self.mamba = mamba

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向
        y_fwd = self.mamba(x)
        # 反向：翻转 -> mamba -> 翻转回来
        y_bwd = _reverse_seq(self.mamba(_reverse_seq(x)))
        return 0.5 * (y_fwd + y_bwd)


class CrossMamba(nn.Module):
    """
    Cross-Mamba: 让 query 序列“跨模态”吸收 context 信息。
    实现方式：concat([context, query]) -> Mamba -> 取 query 部分输出

    输入:
      query   : [B, Lq, Dq]   (例如文本特征)
      context : [B, Lc, Dc]   (例如视觉/音频特征)
    输出:
      out     : [B, Lq, out_dim] (默认 out_dim = Dq)

    可选:
      context_padding_mask: [B, Lc]，True 表示 padding（会把对应 context token 置 0）
    """
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        d_model: int = None,
        out_dim: int = None,
        # mamba hyperparams
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        # block options
        dropout: float = 0.1,
        bidirectional: bool = True,
        add_segment_embed: bool = True,
    ):
        super().__init__()
        d_model = d_model if d_model is not None else query_dim
        out_dim = out_dim if out_dim is not None else query_dim

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.d_model = d_model
        self.out_dim = out_dim
        self.bidirectional = bidirectional
        self.add_segment_embed = add_segment_embed

        # 1) 输入投影到 d_model
        self.q_in = nn.Identity() if query_dim == d_model else nn.Linear(query_dim, d_model, bias=False)
        self.c_in = nn.Identity() if context_dim == d_model else nn.Linear(context_dim, d_model, bias=False)

        # 2) segment embedding（区分 context / query），可选
        if add_segment_embed:
            self.seg_context = nn.Parameter(torch.zeros(1, 1, d_model))
            self.seg_query = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.seg_context, std=0.02)
            nn.init.normal_(self.seg_query, std=0.02)

        # 3) Mamba 主体（要求安装 mamba-ssm）
        try:
            from mamba_ssm import Mamba
        except Exception as e:
            raise ImportError(
                "无法导入 mamba_ssm.Mamba。请先安装：pip install mamba-ssm\n"
                f"原始错误：{repr(e)}"
            )

        mamba_core = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.mamba = BiMamba(mamba_core) if bidirectional else mamba_core

        # 4) 输出投影到 out_dim
        self.out_proj = nn.Identity() if d_model == out_dim else nn.Linear(d_model, out_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,                      # [B, Lq, Dq]
        context: torch.Tensor,                    # [B, Lc, Dc]
        context_padding_mask: torch.Tensor = None # [B, Lc], True=pad
    ) -> torch.Tensor:
        B, Lq, _ = query.shape
        B2, Lc, _ = context.shape
        assert B == B2, "batch size mismatch"

        q = self.q_in(query)       # [B, Lq, d_model]
        c = self.c_in(context)     # [B, Lc, d_model]

        # padding mask：把 padding token 置 0（Mamba 不像 attention 那样天然支持 mask）
        if context_padding_mask is not None:
            # True=pad -> 0
            mask = (~context_padding_mask).to(c.dtype)  # valid=1, pad=0
            c = c * mask[:, :, None]

        if self.add_segment_embed:
            c = c + self.seg_context
            q = q + self.seg_query

        # 拼接：context 在前，query 在后 => query 看到 context（Cross）
        x = torch.cat([c, q], dim=1)  # [B, Lc+Lq, d_model]
        y = self.mamba(x)             # [B, Lc+Lq, d_model]

        # 取 query 段
        yq = y[:, Lc:, :]             # [B, Lq, d_model]
        out = self.out_proj(yq)       # [B, Lq, out_dim]
        out = self.drop(out)
        return out


class ModalityEnhancer(nn.Module):
    """
    单模态特征增强模块: Pre-LN + BiMamba + Residual + FFN
    用于在proj_v/proj_a/proj_l后进一步细化单模态特征。
    
    输入: [B, L, D]
    输出: [B, L, D]
    """
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.dim = dim
        
        # Pre-LN for Mamba
        self.ln1 = nn.LayerNorm(dim)
        
        # BiMamba for sequence modeling
        mamba_core = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba = BiMamba(mamba_core) if bidirectional else mamba_core
        self.drop1 = nn.Dropout(dropout)
        
        # Pre-LN for FFN
        self.ln2 = nn.LayerNorm(dim)
        hidden = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        # Mamba with residual
        x = x + self.drop1(self.mamba(self.ln1(x)))
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class CrossMambaBlock(nn.Module):
    """
    Pre-LN + CrossMamba + Residual + FFN 的标准 block（即插即用）
    输出维度默认保持为 query_dim。
    """
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        d_model: int = None,
        out_dim: int = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        bidirectional: bool = True,
    ):
        super().__init__()
        out_dim = out_dim if out_dim is not None else query_dim

        self.ln1 = nn.LayerNorm(query_dim)
        self.cross = CrossMamba(
            query_dim=query_dim,
            context_dim=context_dim,
            d_model=d_model if d_model is not None else query_dim,
            out_dim=out_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            bidirectional=bidirectional,
            add_segment_embed=True,
        )

        # 若 out_dim != query_dim，残差要对齐
        self.skip = nn.Identity() if out_dim == query_dim else nn.Linear(query_dim, out_dim, bias=False)

        self.ln2 = nn.LayerNorm(out_dim)
        hidden = out_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,                      # [B, Lq, Dq]
        context: torch.Tensor,                    # [B, Lc, Dc]
        context_padding_mask: torch.Tensor = None # [B, Lc]
    ) -> torch.Tensor:
        q0 = query
        q = self.ln1(query)
        cross_out = self.cross(q, context, context_padding_mask=context_padding_mask)  # [B, Lq, out_dim]
        x = self.skip(q0) + cross_out
        x = x + self.ffn(self.ln2(x))
        return x


if __name__ == "__main__":
    # 演示：文本 (64,50,768) 与 视觉 (64,50,768) 做一次 Cross Mamba；
    #      文本 与 音频 再做一次 Cross Mamba；
    #      最后 concat。

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    B, L, D = 64, 50, 768
    text = torch.randn(B, L, D, device=device)
    vision = torch.randn(B, L, D, device=device)
    audio = torch.randn(B, L, D, device=device)

    # text <- vision
    text_x_v = CrossMambaBlock(
        query_dim=D,
        context_dim=D,
        d_model=D,        # 可以改小，比如 512
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        bidirectional=True
    ).to(device)

    # text <- audio
    text_x_a = CrossMambaBlock(
        query_dim=D,
        context_dim=D,
        d_model=D,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        bidirectional=True
    ).to(device)

    out_tv = text_x_v(text, vision)  # [64,50,768]
    out_ta = text_x_a(text, audio)   # [64,50,768]

    fused = torch.cat([out_tv, out_ta], dim=-1)  # [64,50,1536]

    print("out_tv:", out_tv.shape)
    print("out_ta:", out_ta.shape)
    print("fused :", fused.shape)
