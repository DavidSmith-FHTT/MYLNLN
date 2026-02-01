import torch
import torch.nn as nn


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
    Cross-Mamba: 让 query 序列"跨模态"吸收 context 信息。
    实现方式：concat([context, query]) -> Mamba -> 取 query 部分输出

    输入:
      query   : [B, Lq, Dq]   (例如文本特征)
      context : [B, Lc, Dc]   (例如视觉/音频特征)
    输出:
      out     : [B, Lq, out_dim] (默认 out_dim = Dq)
    """
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        d_model: int = None,
        # mamba hyperparams
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        # block options
        bidirectional: bool = True,
    ):
        super().__init__()
        d_model = d_model if d_model is not None else query_dim

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.d_model = d_model
        self.bidirectional = bidirectional

        # 1) 输入投影到 d_model
        self.q_in = nn.Identity() if query_dim == d_model else nn.Linear(query_dim, d_model, bias=False)
        self.c_in = nn.Identity() if context_dim == d_model else nn.Linear(context_dim, d_model, bias=False)

        # 2) segment embedding（区分 context / query）
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

        # 4) 输出投影到 query_dim
        self.out_proj = nn.Identity() if d_model == query_dim else nn.Linear(d_model, query_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,                      # [B, Lq, Dq]
        context: torch.Tensor,                    # [B, Lc, Dc]
    ) -> torch.Tensor:
        B, Lq, _ = query.shape
        B2, Lc, _ = context.shape
        assert B == B2, "batch size mismatch"

        q = self.q_in(query)       # [B, Lq, d_model]
        c = self.c_in(context)     # [B, Lc, d_model]

        # 添加 segment embedding
        c = c + self.seg_context
        q = q + self.seg_query

        # 拼接：context 在前，query 在后 => query 看到 context（Cross）
        x = torch.cat([c, q], dim=1)  # [B, Lc+Lq, d_model]
        y = self.mamba(x)             # [B, Lc+Lq, d_model]

        # 取 query 段
        yq = y[:, Lc:, :]             # [B, Lq, d_model]
        out = self.out_proj(yq)       # [B, Lq, query_dim]
        return out


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
        # mamba hyperparams
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        # block options
        bidirectional: bool = True,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(query_dim)
        self.cross = CrossMamba(
            query_dim=query_dim,
            context_dim=context_dim,
            d_model=d_model if d_model is not None else query_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bidirectional=bidirectional,
        )

        self.ln2 = nn.LayerNorm(query_dim)
        hidden = query_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, query_dim),
        )

    def forward(
        self,
        query: torch.Tensor,                      # [B, Lq, Dq]
        context: torch.Tensor,                    # [B, Lc, Dc]
    ) -> torch.Tensor:
        q0 = query
        q = self.ln1(query)
        cross_out = self.cross(q, context)  # [B, Lq, query_dim]
        x = q0 + cross_out
        x = x + self.ffn(self.ln2(x))
        return x

