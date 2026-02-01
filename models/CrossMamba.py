import torch
import torch.nn as nn
from mamba_ssm import Mamba


def _reverse_seq(x: torch.Tensor) -> torch.Tensor:
    """
    将序列维度反转（用于构造反向序列的分支）。

    Args:
        x: [B, L, D]  输入序列特征（batch, length, dim）

    Returns:
        [B, L, D]  序列维度 L 上翻转后的张量
    """
    # x: [B, L, D]
    return torch.flip(x, dims=[1])


class BiMamba(nn.Module):
    """
    双向 Mamba 封装：前向 Mamba + 反向 Mamba（通过“翻转序列”实现）。

    设计意图：
    - 前向分支捕获从左到右的依赖；
    - 反向分支捕获从右到左的依赖；
    - 最终将两者平均融合（0.5 * (y_fwd + y_bwd)），以获得双向上下文信息。
    """

    def __init__(self, mamba: nn.Module):
        """
        Args:
            mamba: 任意可作用于 [B, L, D] -> [B, L, D] 的序列模块（这里传入 Mamba 核心）
        """
        super().__init__()

        # 核心序列建模模块（前向与反向共享同一个 mamba 实例：更省参数）
        self.mamba = mamba

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]  输入序列特征

        Returns:
            [B, L, D]  双向融合后的序列特征
        """
        # 前向分支：直接建模原始序列
        y_fwd = self.mamba(x)

        # 反向分支：翻转序列 -> 过 mamba -> 再翻转回来对齐原顺序
        # y_bwd = _reverse_seq(self.mamba(_reverse_seq(x)))
        y_bwd = self.mamba(_reverse_seq(x))

        # 双向输出平均融合（保持尺度稳定）
        return 0.5 * (y_fwd + y_bwd)


class CrossMamba(nn.Module):
    """
    Cross-Mamba：让 query 序列“跨模态/跨来源”吸收 context 序列信息。

    实现方式（不改变任何计算逻辑）：
    1) 为 context / query 分别加上可学习的 segment embedding（区分来源）
    2) 拼接：concat([context, query]) 使 query 位置能“看到”前面的 context
    3) 对拼接序列做 Mamba（或 BiMamba）建模
    4) 只取输出中的 query 段作为最终输出

    输入:
      query   : [B, Lq, Dq]   (例如文本特征 / 主模态特征)
      context : [B, Lc, Dc]   (例如视觉/音频特征 / 辅助模态特征)
    输出:
      out     : [B, Lq, out_dim] (默认 out_dim = Dq)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        d_model: int,
        # mamba hyperparams
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        # block options
        bidirectional: bool = True,
    ):
        super().__init__()

        # 维度配置（记录用途：便于调试/阅读）
        self.query_dim = query_dim         # query 特征维度（期望与 d_model 一致）
        self.context_dim = context_dim     # context 特征维度（期望与 d_model 一致）
        self.d_model = d_model             # Mamba 内部工作维度
        self.bidirectional = bidirectional # 是否启用双向 Mamba

        # 1) segment embedding（区分 context / query）——可学习参数
        #    作用：给不同来源的 token 注入“来源标识”，帮助模型区分两段序列
        self.seg_context = nn.Parameter(torch.zeros(1, 1, d_model))
        self.seg_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.seg_context, std=0.02)
        nn.init.normal_(self.seg_query, std=0.02)

        # 2) Mamba 主体（序列建模核心）
        #    - mamba_core：单向 Mamba
        #    - self.mamba：若 bidirectional=True，则外面包一层 BiMamba 做双向融合
        mamba_core = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba = BiMamba(mamba_core) if bidirectional else mamba_core

    def forward(
        self,
        query: torch.Tensor,    # [B, Lq, Dq]
        context: torch.Tensor,  # [B, Lc, Dc]
    ) -> torch.Tensor:
        """
        Args:
            query:   [B, Lq, Dq] 查询序列（希望被增强的那一段）
            context: [B, Lc, Dc] 上下文序列（提供信息的那一段）

        Returns:
            out: [B, Lq, d_model] 取回 query 段对应的输出
        """
        # 读取 batch 与长度信息（用于切分 query 段）
        B, Lq, _ = query.shape
        B2, Lc, _ = context.shape
        assert B == B2, "batch size mismatch"

        # 为了可读性保留局部别名（不改变计算）
        q = query       # [B, Lq, d_model]（假设外部已对齐到 d_model）
        c = context     # [B, Lc, d_model]（假设外部已对齐到 d_model）

        # 添加 segment embedding：分别标记 context / query
        c = c + self.seg_context
        q = q + self.seg_query

        # 拼接：context 在前，query 在后
        # 目的：让 query token 在序列建模时能够利用前面 context 的信息（Cross）
        x = torch.cat([c, q], dim=1)  # [B, Lc+Lq, d_model]

        # 序列建模：Mamba / BiMamba
        y = self.mamba(x)             # [B, Lc+Lq, d_model]

        # 取 query 段（丢弃 context 段输出）
        yq = y[:, Lc:, :]             # [B, Lq, d_model]

        # 输出（维持原逻辑：out 即 query 段输出）
        out = yq                      # [B, Lq, query_dim]
        return out


class CrossMambaBlock(nn.Module):
    """
    标准可插拔 Block：Pre-LN + CrossMamba + Residual + FFN（Transformer 风格的结构）

    结构顺序（不改变任何计算逻辑）：
      1) ln1(query) -> cross(...) -> 残差加回 query
      2) ln2(x) -> ffn(...) -> 残差加回 x

    输出维度默认保持为 query_dim。
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        d_model: int,
        # mamba hyperparams
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        # block options
        bidirectional: bool = True,
    ):
        super().__init__()

        # LayerNorm 1：CrossMamba 前的预归一化（Pre-Norm）
        # 作用：稳定 CrossMamba 输入分布，利于训练
        self.ln1 = nn.LayerNorm(query_dim)

        # CrossMamba：跨序列/跨模态信息注入模块（核心交互层）
        self.cross = CrossMamba(
            query_dim=query_dim,
            context_dim=context_dim,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bidirectional=bidirectional,
        )

        # LayerNorm 2：FFN 前的预归一化（Pre-Norm）
        # 作用：稳定 FFN 输入分布
        self.ln2 = nn.LayerNorm(query_dim)

        # FFN 隐藏层维度（典型设置：4x 扩展）
        hidden = query_dim * 4

        # FFN：位置前馈网络（逐 token 的非线性变换）
        # 作用：增强表示能力、引入非线性、补足 CrossMamba 之外的特征变换
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, hidden),  # 线性升维
            nn.GELU(),                     # 非线性激活
            nn.Linear(hidden, query_dim),  # 线性降维回原维度
        )

    def forward(
        self,
        query: torch.Tensor,    # [B, Lq, Dq]
        context: torch.Tensor,  # [B, Lc, Dc]
    ) -> torch.Tensor:
        """
        Args:
            query:   [B, Lq, Dq] 需要被增强的序列特征
            context: [B, Lc, Dc] 提供补充信息的序列特征

        Returns:
            [B, Lq, Dq] 增强后的 query 特征
        """
        # 第一段残差：保留原 query 作为 residual
        q0 = query

        # CrossMamba 前 Pre-Norm
        q = self.ln1(query)

        # Cross 交互：让 query 吸收 context 信息
        cross_out = self.cross(q, context)  # [B, Lq, query_dim]

        # 残差连接：原 query + cross 输出
        x = q0 + cross_out

        # 第二段：FFN（Pre-Norm + FFN + Residual）
        x = x + self.ffn(self.ln2(x))

        return x
