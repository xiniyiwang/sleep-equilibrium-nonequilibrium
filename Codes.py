import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from collections import defaultdict
from scipy.stats import linregress
import seaborn as sns
import random
from typing import List, Tuple, Optional, Iterable, Union, Sequence
from collections import Counter
from typing import Dict, List, Tuple, Iterable
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import datetime
import pandas as pd

Cycle = Tuple[int, ...]
DictCycles = Dict[int, List[Cycle]]

# pip install pyEDFlib numpy pandas
from pathlib import Path
from pyedflib import EdfReader
from scipy import stats
from typing import Iterable, Tuple, Optional, Union, List, Dict, Literal, Any
from scipy.stats import wilcoxon, rankdata
def explained_variance_via_regression(a, b):
    """
    用线性回归 a = alpha + beta * b 计算 b 对 a 的解释方差
    """
    a_mean = a.mean()
    b_mean = b.mean()

    # 回归系数 beta, alpha
    beta = ((a - a_mean) * (b - b_mean)).sum() / ((b - b_mean) ** 2).sum()
    alpha = a_mean - beta * b_mean

    # 预测 a_hat
    a_hat = alpha + beta * b

    # SST: total sum of squares
    sst = ((a - a_mean) ** 2).sum()
    # SSR: residual sum of squares
    ssr = ((a - a_hat) ** 2).sum()

    R2 = 1 - ssr / (sst + 1e-12)
    return R2
def plot_regression_with_ci(
    xwid: float, ywid: float,
    x_list: Iterable[float],
    y1_list: Iterable[float],
    y2_list: Optional[Iterable[float]] = None,
    *,
    x_label: Optional[str] = "X",
    y1_label: Optional[str] = "Y (left)",
    y2_label: Optional[str] = "Y (right)",
    title: Optional[str] = "Linear Regression (dual y-axes)",
    scatter_size: int = 50,
    show: bool = True,
    left_color: str = "tab:red",
    right_color: str = "tab:blue",
    # 新增参数
    legend_loc: str = "upper left",
    save_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    x 对 y1（左轴，红）必画；x 对 y2（右轴，蓝）可选。
    - 图例固定在 legend_loc（默认左上角）
    - 若提供 save_path，则保存为 PDF（自动补 .pdf）
    """
    x = np.asarray(x_list, dtype=float)
    y1 = np.asarray(y1_list, dtype=float)

    if x.shape[0] != y1.shape[0]:
        raise ValueError("x_list 与 y1_list 长度不一致。")

    fig, ax_left = plt.subplots(figsize=(xwid, ywid))
    results: Dict[str, Dict[str, Any]] = {}

    def _fit_and_plot(ax, x_arr, y_arr, legend_base, color):
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_clean = x_arr[mask]
        y_clean = y_arr[mask]
        if x_clean.size < 2:
            raise ValueError(f"{legend_base or 'Series'} 有效点不足以回归（<2）。")

        reg = linregress(x_clean, y_clean)
        xfit = np.linspace(x_clean.min(), x_clean.max(), 200)
        yfit = reg.slope * xfit + reg.intercept

        ax.scatter(x_clean, y_clean, s=scatter_size, alpha=0.85,
                   facecolor=color, edgecolor=color)
        (ln,) = ax.plot(xfit, yfit, linewidth=2, color=color)

        legend_text = f"r={reg.rvalue:.2f}, p={reg.pvalue:.1e}"
        return reg, ln, legend_text, x_clean.size

    # 左轴（红）
    reg_left, ln_left, lab_left, n_left = _fit_and_plot(ax_left, x, y1, y1_label, left_color)
    if x_label is not None:
        ax_left.set_xlabel(x_label)
    if y1_label is not None:
        ax_left.set_ylabel(y1_label, color=left_color)

    ax_left.tick_params(axis='y', colors=left_color)
    ax_left.spines['left'].set_color(left_color)
    ax_left.grid(True, alpha=0.3)

    results["left"] = {
        "slope": reg_left.slope,
        "intercept": reg_left.intercept,
        "r": reg_left.rvalue,
        "p": reg_left.pvalue,
        "stderr": reg_left.stderr,
        "intercept_stderr": getattr(reg_left, "intercept_stderr", np.nan),
        "n": int(n_left),
    }

    handles = [ln_left]
    labels = [lab_left]

    # 右轴（蓝，可选）
    if y2_list is not None:
        y2 = np.asarray(y2_list, dtype=float)
        if x.shape[0] != y2.shape[0]:
            raise ValueError("x_list 与 y2_list 长度不一致。")

        ax_right = ax_left.twinx()
        reg_right, ln_right, lab_right, n_right = _fit_and_plot(ax_right, x, y2, y2_label, right_color)
        if y2_label is not None:
            ax_right.set_ylabel(y2_label, color=right_color)

        ax_right.tick_params(axis='y', colors=right_color)
        ax_right.spines['right'].set_color(right_color)

        results["right"] = {
            "slope": reg_right.slope,
            "intercept": reg_right.intercept,
            "r": reg_right.rvalue,
            "p": reg_right.pvalue,
            "stderr": reg_right.stderr,
            "intercept_stderr": getattr(reg_right, "intercept_stderr", np.nan),
            "n": int(n_right),
        }

        handles.append(ln_right)
        labels.append(lab_right)

    if title is not None:
        ax_left.set_title(title)

    # 固定图例位置到左上角（或 legend_loc 指定）
    # ax_left.legend(handles, labels, loc=legend_loc, frameon=False)
    
    leg = fig.legend(handles, labels, loc=legend_loc, frameon=True) #
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.6)
    leg.set_zorder(1000)

    plt.tight_layout()

    # 保存 PDF（若提供 save_path）
    if save_path is not None:
        if not save_path.lower().endswith(".pdf"):
            save_path = save_path + ".pdf"
        fig.savefig(save_path, format='pdf', dpi=300)

    if show:
        plt.show()

    return results
    
def test_loop_extremeness(
    loop_key: Tuple[int, ...],
    AllCycle: Dict[int, List[List[int]]],
    AllStrength: Dict[int, List[float]],
    baseline: Literal["mean", "max", "min"] = "mean",   # 与“其他环”比较的基线
    aggregate: Literal["mean", "sum", "max", "min", "first"] = "mean",  # 同一样本内该环多次出现如何聚合
    zero_method: Literal["wilcox", "pratt"] = "pratt", #"wilcox",
    mode: Literal["auto", "approx", "exact"] = "auto",
    tie_tol: float = 1e-12
) -> dict:
    """
    对指定 loop_key，构造每个样本的一条差值：
        diff_s = X_C^(agg) - baseline(others)
    然后对 diffs 做单样本 Wilcoxon 符号秩检验：
        p_greater : H1 中位数 > 0 （显著更大/更接近“最大”）
        p_less    : H1 中位数 < 0 （显著更小/更接近“最小”）

    返回字典包含 p 值、样本数、差值中位数、唯一最大/最小计数等。
    """
    def agg(vals: List[float]) -> Optional[float]:
        if len(vals) == 0: 
            return None
        if aggregate == "mean": return float(np.mean(vals))
        if aggregate == "sum":  return float(np.sum(vals))
        if aggregate == "max":  return float(np.max(vals))
        if aggregate == "min":  return float(np.min(vals))
        if aggregate == "first":return float(vals[0])
        raise ValueError("aggregate not in {'mean','sum','max','min','first'}")

    diffs = []
    per_sample = []  # 保存 (sample_id, xC, max_others, min_others)

    # 逐样本构造一条差值
    for s, cycles in AllCycle.items():
        if not cycles:
            continue
        strengths = AllStrength.get(s, [])
        if not strengths or len(strengths) != len(cycles):
            continue

        # 该环在此样本内的所有出现位置
        idxs_this = [i for i, c in enumerate(cycles) if tuple(c) == loop_key]
        if not idxs_this:
            continue

        # 仅与 “不同环” 比较（排除同环的其它出现）
        idxs_other = [j for j, c in enumerate(cycles) if tuple(c) != loop_key]
        if not idxs_other:
            continue

        vals_this  = [float(strengths[i]) for i in idxs_this]
        vals_other = [float(strengths[j]) for j in idxs_other]

        xC = agg(vals_this)
        if xC is None:
            continue

        # baseline(others)
        if baseline == "mean":
            base = float(np.mean(vals_other))
        elif baseline == "max":
            base = float(np.max(vals_other))
        elif baseline == "min":
            base = float(np.min(vals_other))
        else:
            raise ValueError("baseline must be in {'mean','max','min'}")

        diffs.append(xC - base)
        per_sample.append( (s, xC, float(np.max(vals_other)), float(np.min(vals_other))) )

    # 若无有效样本
    if len(diffs) == 0:
        return {
            "n": 0, "median_diff": None,
            "p_greater": None, "p_less": None,
            "wilcoxon_stat": None,
            "unique_max": 0, "tie_max": 0,
            "unique_min": 0, "tie_min": 0,
            "baseline": baseline, "aggregate": aggregate,
            "zero_method": zero_method, "mode": mode
        }

    diffs = np.asarray(diffs, float)

    # Wilcoxon 单样本（同一 diffs，两个方向各算一次单侧 p）
    stat_g, p_greater = wilcoxon(diffs, alternative="greater",
                                 zero_method=zero_method, mode=mode)
    stat_l, p_less    = wilcoxon(diffs, alternative="less",
                                 zero_method=zero_method, mode=mode)

    # 统计“唯一/并列 最大/最小”次数（基于聚合后的 xC 与 other 的 max/min）
    unique_max = tie_max = unique_min = tie_min = 0
    for _, xC, max_o, min_o in per_sample:
        if xC > max_o + tie_tol:
            unique_max += 1
        elif abs(xC - max_o) <= tie_tol:
            tie_max += 1
        if xC < min_o - tie_tol:
            unique_min += 1
        elif abs(xC - min_o) <= tie_tol:
            tie_min += 1

    # 可选：秩双列相关（rank-biserial）效应量（根据非零差的秩计算）
    nz = np.abs(diffs) > tie_tol
    n_eff = int(np.sum(nz))
    r_rb = None
    if n_eff > 0:
        ranks = rankdata(np.abs(diffs[nz]))
        W_plus = float(np.sum(ranks[diffs[nz] > 0]))
        R = n_eff * (n_eff + 1) / 2.0
        r_rb = 2.0 * W_plus / R - 1.0  # >0 倾向“更大”，<0 倾向“更小”

    return {
        "n": int(len(diffs)),
        "median_diff": float(np.median(diffs)),
        "wilcoxon_stat": float(stat_g),  # 与 SciPy 定义一致：正差秩和
        "p_greater": float(p_greater),   # H1: 该环显著更大
        "p_less": float(p_less),         # H1: 该环显著更小
        "rank_biserial_r": r_rb,         # 效应量，>0 倾向大，<0 倾向小
        "unique_max": int(unique_max), "tie_max": int(tie_max),
        "unique_min": int(unique_min), "tie_min": int(tie_min),
        "baseline": baseline, "aggregate": aggregate,
        "zero_method": zero_method, "mode": mode
    }
    
def test_magnitude_against_mean(loop_key, AllCycle, AllStrength, use_gap=False):
    """use_gap=False: d_s = xC - mean(others)
       use_gap=True : g_s = xC - max(others)"""
    diffs = []
    for s, cycles in AllCycle.items():
        if not cycles: 
            continue
        strengths = AllStrength.get(s, [])
        # 找到该loop在此样本中的所有出现位置（通常1个）
        idxs = [i for i, c in enumerate(cycles) if tuple(c) == loop_key]
        for i in idxs:
            xC = strengths[i]
            others = [strengths[j] for j in range(len(strengths)) if j != i]
            if not others:
                continue
            diff = xC - (max(others) if use_gap else np.mean(others))
            diffs.append(diff)

    if len(diffs) == 0:
        return {"n": 0, "stat": None, "p": None, "median_diff": None}

    # 单侧：大于0（更大）
    stat, p = wilcoxon(diffs, alternative='greater', zero_method='wilcox', mode='approx')
    median_diff = float(np.median(diffs))

    result = {
        "n": len(diffs),
        "stat": stat,
        "p": p,
        "median_diff": median_diff
    }

    # 只打印显著的结果
    if p < 0.05:
        print(
            f"loop {loop_key}, use_gap={use_gap}: "
            f"n={len(diffs)}, stat={stat:.3f}, p={p:.3g}, median_diff={median_diff:.3g}"
        )

    return result
    
def fmt_sig2_html_threshold(x, threshold=1e-3, sigfigs=2):
    """
    两位有效数字；|x| <= threshold 用 HTML 上标的 ×10^k 形式（如 2×10<sup>−5</sup>）；
    否则用十进制小数（不使用科学计数法），保留 sigfigs 位有效数字并去掉多余的 0。
    """
    import math
    import pandas as pd
    if pd.isna(x):
        return ""
    if x == 0:
        return "0"

    ax = abs(float(x))

    # 小数阈值内：a × 10^k（k 为上标，使用 HTML <sup>）
    if ax <= threshold:
        # 计算指数 k 和系数 a，使得 ax = a * 10^k，a ∈ [1,10)
        k = 0 if ax == 0 else math.floor(math.log10(ax))
        a = ax / (10 ** k)
        # 两位有效数字四舍五入
        a_rounded = float(f"{a:.{sigfigs}g}")
        # 处理四舍五入后 a=10 的进位：e.g. 9.95 -> 10.0
        if a_rounded >= 10:
            a_rounded /= 10.0
            k += 1
        # 号（正负）
        sign = "−" if x < 0 else ""   # 使用 Unicode 负号更美观
        k_sign = "−" if k < 0 else ""
        return f"{sign}{a_rounded:g}×10<sup>{k_sign}{abs(k)}</sup>"

    # 否则：十进制显示（不使用科学计数法），保留 sigfigs 有效数字
    # 计算需要的小数位数，以确保 sigfigs 有效数字
    k = math.floor(math.log10(ax))
    decimals = max(0, sigfigs - k - 1)
    val = round(float(x), decimals)
    s = f"{val:.{decimals}f}"
    # 去掉多余的 0 和末尾的小数点
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s
    
def _map_states_to_idx(states: np.ndarray) -> Tuple[dict, dict]:
    """建立状态标签与矩阵索引的双向映射（保持升序稳定）。"""
    uniq = np.array(sorted(np.unique(states)))
    lbl2idx = {lbl: i for i, lbl in enumerate(uniq)}
    idx2lbl = {i: lbl for i, lbl in enumerate(uniq)}
    return lbl2idx, idx2lbl

def count_transitions(
    seq: Union[np.ndarray, List[int]],
    states: Optional[Iterable[int]] = None,
    include_self: bool = False
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    统计一个序列的状态转移次数矩阵。
    参数
    ----
    seq : 序列（如 [0,0,1,2,1,...]）
    states : 可选，指定状态集合（如 [0,1,2,3,4]）；不指定则用序列中出现过的状态
    include_self : 是否统计自环（i->i）

    返回
    ----
    counts : numpy 数组，shape = (K, K)，K=状态数
    counts_df : pandas DataFrame，行为来源状态，列为去向状态
    """
    seq = np.asarray(seq)
    if seq.size < 2:
        raise ValueError("序列长度需≥2才能产生转移。")

    if states is None:
        states = np.unique(seq)
    else:
        states = np.array(list(states))

    lbl2idx, idx2lbl = _map_states_to_idx(states)

    # 过滤掉不在 states 集合中的点（若有）
    mask_in = np.isin(seq, states)
    seq = seq[mask_in]
    if seq.size < 2:
        raise ValueError("过滤后序列长度不足以统计转移。")

    src = seq[:-1]
    dst = seq[1:]
    if not include_self:
        keep = src != dst
        src, dst = src[keep], dst[keep]

    # 映射到索引
    src_idx = np.vectorize(lbl2idx.get)(src)
    dst_idx = np.vectorize(lbl2idx.get)(dst)

    K = len(states)
    counts = np.zeros((K, K), dtype=int)
    np.add.at(counts, (src_idx, dst_idx), 1)

    counts_df = pd.DataFrame(
        counts,
        index=[idx2lbl[i] for i in range(K)],
        columns=[idx2lbl[i] for i in range(K)],
    )
    counts_df.index.name = "from"
    counts_df.columns.name = "to"
    return counts, counts_df

def batch_count_transitions(
    sequences: List[Union[np.ndarray, List[int]]],
    states: Optional[Iterable[int]] = None,
    include_self: bool = False
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    对多名被试（多条序列）汇总转移次数。
    """
    if states is None:
        # 汇总出现过的状态
        all_vals = np.concatenate([np.asarray(s).ravel() for s in sequences])
        states = np.unique(all_vals)
    states = np.array(list(states))

    total_counts = np.zeros((len(states), len(states)), dtype=int)
    for seq in sequences:
        c, _ = count_transitions(seq, states=states, include_self=include_self)
        total_counts += c

    counts_df = pd.DataFrame(
        total_counts,
        index=states, columns=states
    )
    counts_df.index.name = "from"
    counts_df.columns.name = "to"
    return total_counts, counts_df

def normalize_rows(counts: np.ndarray) -> np.ndarray:
    """
    将转移次数矩阵按行归一化为转移概率矩阵（遇到全0行则保持0）。
    """
    counts = counts.astype(float)
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums>0)
    return probs
    
plt.rcParams.update({
    "font.size": 15,          # 基础字号（影响很多元素的默认值）
    "axes.labelsize": 15,     # x/y 轴标签（xlabel, ylabel）
    "axes.titlesize": 16,     # 标题
    "xtick.labelsize": 15,    # x 轴刻度数字
    "ytick.labelsize": 15,    # y 轴刻度数字
    "legend.fontsize": 15,    # 图例文字
    "legend.title_fontsize": 15, # 图例标题（若有）
})

import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Optional, Union, List

def _map_states_to_idx(states: np.ndarray) -> Tuple[dict, dict]:
    """建立状态标签与矩阵索引的双向映射（保持升序稳定）。"""
    uniq = np.array(sorted(np.unique(states)))
    lbl2idx = {lbl: i for i, lbl in enumerate(uniq)}
    idx2lbl = {i: lbl for i, lbl in enumerate(uniq)}
    return lbl2idx, idx2lbl

def count_transitions(
    seq: Union[np.ndarray, List[int]],
    states: Optional[Iterable[int]] = None,
    include_self: bool = False
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    统计一个序列的状态转移次数矩阵。
    参数
    ----
    seq : 序列（如 [0,0,1,2,1,...]）
    states : 可选，指定状态集合（如 [0,1,2,3,4]）；不指定则用序列中出现过的状态
    include_self : 是否统计自环（i->i）

    返回
    ----
    counts : numpy 数组，shape = (K, K)，K=状态数
    counts_df : pandas DataFrame，行为来源状态，列为去向状态
    """
    seq = np.asarray(seq)
    if seq.size < 2:
        raise ValueError("序列长度需≥2才能产生转移。")

    if states is None:
        states = np.unique(seq)
    else:
        states = np.array(list(states))

    lbl2idx, idx2lbl = _map_states_to_idx(states)

    # 过滤掉不在 states 集合中的点（若有）
    mask_in = np.isin(seq, states)
    seq = seq[mask_in]
    if seq.size < 2:
        raise ValueError("过滤后序列长度不足以统计转移。")

    src = seq[:-1]
    dst = seq[1:]
    if not include_self:
        keep = src != dst
        src, dst = src[keep], dst[keep]

    # 映射到索引
    src_idx = np.vectorize(lbl2idx.get)(src)
    dst_idx = np.vectorize(lbl2idx.get)(dst)

    K = len(states)
    counts = np.zeros((K, K), dtype=int)
    np.add.at(counts, (src_idx, dst_idx), 1)

    counts_df = pd.DataFrame(
        counts,
        index=[idx2lbl[i] for i in range(K)],
        columns=[idx2lbl[i] for i in range(K)],
    )
    counts_df.index.name = "from"
    counts_df.columns.name = "to"
    return counts, counts_df

def batch_count_transitions(
    sequences: List[Union[np.ndarray, List[int]]],
    states: Optional[Iterable[int]] = None,
    include_self: bool = False
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    对多名被试（多条序列）汇总转移次数。
    """
    if states is None:
        # 汇总出现过的状态
        all_vals = np.concatenate([np.asarray(s).ravel() for s in sequences])
        states = np.unique(all_vals)
    states = np.array(list(states))

    total_counts = np.zeros((len(states), len(states)), dtype=int)
    for seq in sequences:
        c, _ = count_transitions(seq, states=states, include_self=include_self)
        total_counts += c

    counts_df = pd.DataFrame(
        total_counts,
        index=states, columns=states
    )
    counts_df.index.name = "from"
    counts_df.columns.name = "to"
    return total_counts, counts_df

def normalize_rows(counts: np.ndarray) -> np.ndarray:
    """
    将转移次数矩阵按行归一化为转移概率矩阵（遇到全0行则保持0）。
    """
    counts = counts.astype(float)
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums>0)
    return probs
    
def epoch_index_from_clock(start_dt: datetime.datetime,
                           clock_str: str,
                           epoch_len_s: int = 60) -> Optional[int]:
    """
    根据记录起始时刻 start_dt（tz-aware）与'HH:MM:SS'格式的 clock_str，
    计算该时刻在 60s/epoch 的 hypnogram 中对应的 epoch 索引（从 0 开始）。
    若 clock_str 为空/无效，返回 None。
    规则：将 clock_str 先拼到 start_dt 的“同一天”，如果早于 start_dt，则顺延到次日；
         若仍早于则继续顺延直到 >= start_dt。
    """
    if clock_str is None or (isinstance(clock_str, float) and pd.isna(clock_str)):
        return None

    clock_str = str(clock_str).strip()
    if not clock_str:
        return None

    # 解析 "HH:MM:SS"
    try:
        hh, mm, ss = map(int, clock_str.split(":"))
        t = datetime.time(hh, mm, ss)
    except Exception:
        return None

    tz = start_dt.tzinfo
    base_date = start_dt.date()
    candidate = datetime.datetime.combine(base_date, t, tzinfo=tz)

    while candidate < start_dt:
        candidate += datetime.timedelta(days=1)

    delta_sec = (candidate - start_dt).total_seconds()
    if delta_sec < 0:
        return None
    return int(delta_sec // epoch_len_s)
    
def hedges_g(x, y):
    # Hedges g（带小样本校正）
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    # 合并标准差（不假设等方差时也常用于效应量的基线）
    sp = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / (nx+ny-2))
    d = (np.mean(x) - np.mean(y)) / sp if sp > 0 else 0.0
    J = 1 - 3/(4*(nx+ny)-9)  # 校正因子
    return J * d

def holm_bonferroni(pvals, labels):
    # 输入 pvals: dict{('a','b'): p_ab, ...}; 返回调整后的p
    items = sorted(pvals.items(), key=lambda kv: kv[1])  # 按原始p从小到大
    m = len(items)
    adjusted = {}
    max_adj = 0
    # Holm step-up（报告非递减）
    for rank, ((i,j), p) in enumerate(items, start=1):
        adj = (m - rank + 1) * p
        max_adj = max(max_adj, adj)
        adjusted[(i,j)] = min(max_adj, 1.0)
    return adjusted

def bootstrap_stat(arr, n_boot=10000, metric='mean', rng=None):
    arr = np.asarray(arr, float)
    rng = np.random.default_rng(rng)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    samp = arr[idx]
    if metric == 'mean':
        return samp.mean(axis=1)
    elif metric == 'median':
        return np.median(samp, axis=1)
    else:
        raise ValueError("metric must be 'mean' or 'median'")

def rank_three_groups(a, b, c, metric='mean', n_boot=20000, alpha=0.05, seed=0):
    groups = {'a': np.asarray(a, float), 'b': np.asarray(b, float), 'c': np.asarray(c, float)}
    # 基本统计
    summary = {}
    for k, v in groups.items():
        summary[k] = dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)),
                          std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)

    # Bootstrap 统计量分布
    boots = {k: bootstrap_stat(v, n_boot=n_boot, metric=metric, rng=seed+idx)
             for idx, (k, v) in enumerate(groups.items())}

    # “胜率”与差值分布
    # 胜率：某组统计量为三组最大（允许平手按等分记分）
    wins = {k: 0.0 for k in groups}
    diffs = {}  # pairwise diff 的置信区间
    pairs = [('a','b'), ('a','c'), ('b','c')]
    for i in range(n_boot):
        vals = {k: boots[k][i] for k in groups}
        maxval = max(vals.values())
        top = [k for k, v in vals.items() if v == maxval]
        for k in top:
            wins[k] += 1.0 / len(top)
    win_prob = {k: wins[k] / n_boot for k in groups}

    # 差值分布与 CI（百分位区间）
    for (g1, g2) in pairs:
        d = boots[g1] - boots[g2]
        lo, hi = np.percentile(d, [100*alpha/2, 100*(1-alpha/2)])
        diffs[(g1, g2)] = dict(diff_mean=float(np.mean(d)), ci=(float(lo), float(hi)))

    # Welch t 检验 + Holm 校正
    pvals = {}
    tests = {}
    for (g1, g2) in pairs:
        tstat, p = stats.ttest_ind(groups[g1], groups[g2], equal_var=False)
        pvals[(g1, g2)] = float(p)
        tests[(g1, g2)] = dict(t=float(tstat), p=float(p), g=float(hedges_g(groups[g1], groups[g2])))
    pvals_adj = holm_bonferroni(pvals, pairs)

    # 排序（按样本统计量 point estimate：选 metric 对应的点估计）
    point = {k: summary[k][metric] for k in groups}
    ranking = sorted(point.items(), key=lambda kv: kv[1], reverse=True)

    out = dict(summary=summary,
               metric=metric,
               point_estimates=point,
               win_prob=win_prob,
               boot_diffs=diffs,
               welch_tests=tests,
               welch_p_holm=pvals_adj,
               ranking=ranking)
    return out
    
from typing import Optional, Iterable, Any
import math

def compute_sleep_period_time(
    stage_seq: Iterable[Any],
    epoch_sec: int = 60,
    wake_values={0, "W", "w", "Wake", "wake"},
    missing_values={"?", None, "NA", "na", "NaN"},
    include_missing_in_spt: bool = False,
    max_wake_gap_sec: Optional[int] = None,   # ✅ 新增：最长允许的清醒间隔
    include_missing_in_gap: bool = False,     # ✅ 新增：缺失是否计入清醒间隔
):
    """
    计算 SPT（Sleep Period Time）：从第一次入睡到“最后一次睡眠结束”的总时段。
    - 中途清醒 (WASO) 计入；
    - 缺失 (missing) 默认不计入（可用 include_missing_in_spt=True 切换）。
    - 若设置 max_wake_gap_sec，则在第一次入睡之后一旦出现“连续清醒 >= 该阈值”，
      SPT 只统计到这段长清醒之前的最后一个睡眠 epoch（等价于只保留第一段睡眠）。
    返回 dict：
      first_sleep_idx, last_sleep_idx, window_epochs, n_missing, missing_seconds,
      n_counted_epochs, spt_seconds, spt_minutes,
      cut_by_long_wake(是否因清醒过长而截断), wake_gap_seconds(触发截断的清醒段长度)
    """
    # --- 兼容 numpy/pandas 标量 ---
    try:
        epoch_sec = int(epoch_sec)
    except Exception:
        epoch_sec = 60

    # --- 统一集合到小写字符串形式（数值保留0用于判定wake） ---
    def _to_lower_str(x):
        return str(x).strip().lower()

    wake_lc = {_to_lower_str(v) for v in wake_values if not (isinstance(v, (int, float)) and v == 0)}
    wake_has_zero = any((isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) and v == 0)
                        for v in wake_values)

    missing_lc = {_to_lower_str(v) for v in missing_values if v is not None}
    missing_has_none = any(v is None for v in missing_values)

    # --- 判定函数 ---
    def _is_nan(x):
        return isinstance(x, float) and math.isnan(x)

    def is_missing(x) -> bool:
        if missing_has_none and x is None:
            return True
        if _is_nan(x):
            return True
        return _to_lower_str(x) in missing_lc

    def is_wake(x) -> bool:
        if (isinstance(x, (int, float)) and not _is_nan(x)) and wake_has_zero and x == 0:
            return True
        return _to_lower_str(x) in wake_lc

    def is_sleep(x) -> bool:
        return (not is_missing(x)) and (not is_wake(x))

    # --- 主体 ---
    stage_list = list(stage_seq)
    n = len(stage_list)
    if n == 0:
        return {
            "first_sleep_idx": None, "last_sleep_idx": None,
            "window_epochs": 0, "n_missing": 0, "missing_seconds": 0,
            "n_counted_epochs": 0, "spt_seconds": 0, "spt_minutes": 0.0,
            "cut_by_long_wake": False, "wake_gap_seconds": 0,
        }

    # 1) 找第一次与最后一次“有测且为睡眠”的 epoch
    first_sleep_idx: Optional[int] = None
    for i, s in enumerate(stage_list):
        if is_sleep(s):
            first_sleep_idx = i
            break
    if first_sleep_idx is None:
        # 全程无睡眠
        return {
            "first_sleep_idx": None, "last_sleep_idx": None,
            "window_epochs": 0, "n_missing": 0, "missing_seconds": 0,
            "n_counted_epochs": 0, "spt_seconds": 0, "spt_minutes": 0.0,
            "cut_by_long_wake": False, "wake_gap_seconds": 0,
        }

    # 2) 从第一次入睡开始向后扫描，跟踪连续清醒段；如超过阈值则截断
    cutoff_last_sleep_idx = None
    cut_by_long_wake = False
    wake_run = 0  # 连续清醒的 epoch 数（可选是否把 missing 算进去）

    # 最近一次的“睡眠”位置（用于在出现长清醒时作为截断点）
    last_sleep_seen = first_sleep_idx

    threshold_epochs = None
    if max_wake_gap_sec is not None and max_wake_gap_sec > 0:
        threshold_epochs = max(1, int(math.ceil(max_wake_gap_sec / float(epoch_sec))))

    for k in range(first_sleep_idx, n):
        x = stage_list[k]
        if is_sleep(x):
            last_sleep_seen = k
            wake_run = 0  # 清零清醒连段
        else:
            # 非睡眠：若是缺失且不计入gap，则不增长 wake_run
            if is_missing(x) and not include_missing_in_gap:
                # 不计入清醒间隔；跳过
                pass
            else:
                # 清醒（或把缺失也算入gap时）
                if is_wake(x) or (is_missing(x) and include_missing_in_gap):
                    wake_run += 1
                    if threshold_epochs is not None and wake_run >= threshold_epochs:
                        cutoff_last_sleep_idx = last_sleep_seen
                        cut_by_long_wake = True
                        break

    # 若未触发截断，则正常的“最后一次睡眠”是全序列末尾的最后睡眠
    if cutoff_last_sleep_idx is None:
        # 正向从末尾找最后睡眠
        last_sleep_idx: Optional[int] = None
        for j in range(n - 1, -1, -1):
            if is_sleep(stage_list[j]):
                last_sleep_idx = j
                break
    else:
        last_sleep_idx = cutoff_last_sleep_idx

    # 3) 统计窗口（first..last）
    window_epochs = last_sleep_idx - first_sleep_idx + 1
    n_missing = 0
    for k in range(first_sleep_idx, last_sleep_idx + 1):
        if is_missing(stage_list[k]):
            n_missing += 1

    if include_missing_in_spt:
        n_counted_epochs = window_epochs
    else:
        n_counted_epochs = max(0, window_epochs - n_missing)

    spt_seconds = n_counted_epochs * epoch_sec

    return {
        "first_sleep_idx": first_sleep_idx,
        "last_sleep_idx": last_sleep_idx,
        "window_epochs": window_epochs,
        "n_missing": n_missing,
        "missing_seconds": n_missing * epoch_sec,
        "n_counted_epochs": n_counted_epochs,
        "spt_seconds": spt_seconds,
        "spt_minutes": spt_seconds / 60.0,
        "cut_by_long_wake": cut_by_long_wake,
        "wake_gap_seconds": (wake_run * epoch_sec) if cut_by_long_wake else 0,
    }
    
def read_sc_subjects(xls_path: Union[str, Path],
                     dtype_backend: str = "numpy_nullable") -> Dict[str, pd.DataFrame]:
    """
    读取 SC-subjects.xls/.xlsx 的所有工作表，并返回:
        {sheet_name: DataFrame}
    """
    xls_path = Path(xls_path)
    if not xls_path.exists():
        raise FileNotFoundError(f"未找到文件：{xls_path}")

    xls = pd.ExcelFile(xls_path)
    sheets_dict: Dict[str, pd.DataFrame] = {}

    for name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=name, dtype_backend=dtype_backend)
        df = df.dropna(how="all")                    # 去掉全空行
        df.columns = [str(c).strip() for c in df.columns]  # 列名去空格
        sheets_dict[name] = df

    return sheets_dict

def plot_sleep_hypnogram(
    LightsOffIdx, FirstIdx, LastIdx, 
    labels: Sequence[str],
    use_aasm: bool = False,
    show_N4: bool = True,   # 新增：True 时采用 R&K (包含 N4)，False 时不单独显示 N4
    epoch_sec: int = 30,
    start_at_sec: float = 0.0,
    title: Optional[str] = None,
    invert_y: bool = True,
    figsize=(12, 3),
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    用 30s（或自定义 epoch_sec）睡眠分期标签绘制阶梯式 hypnogram。

    参数
    ----
    labels : 序列[str]
        R&K 风格标签（'W','1','2','3','4','R','M','?'），或已合并的 AASM（'W','N1','N2','N3','REM'）。
    use_aasm : bool
        True: 将 R&K 自动合并到 AASM（3/4->N3, R->REM, M/? 为空缺）。
        False: 保持 R&K 或近似 R&K 的数轴。
    show_N4 : bool
        仅在 use_aasm=False 时生效：
        True  -> R&K 数轴包含 N4（W, N1, N2, N3, N4, REM）。
        False -> 不单独显示 N4，将 4 合并到 3（W, N1, N2, N3, REM）。
    epoch_sec : int
        每个 epoch 的秒数（默认 30）。
    start_at_sec : float
        第一个 epoch 的起始时间（秒），用于横轴偏移。
    title : str | None
        图标题。
    invert_y : bool
        是否将 Y 轴倒置（顶端为 W，底部为 REM/N3）。
    figsize : tuple
        画布大小。
    save_path : str | None
        若提供则保存图片到该路径。
    ax : matplotlib.axes.Axes | None
        若提供则在该坐标轴上作图；否则新建图像。

    返回
    ----
    ax : matplotlib.axes.Axes
    """
    # 两套映射：AASM 或 R&K (+ 是否显示 N4)
    if use_aasm:
        # AASM：W, N1, N2, N3, REM
        stage2num = {
            'W':0, 'N1':1, 'N2':2, 'N3':3, 'REM':4,
            # 兼容直接喂 R&K 的情况
            '1':1, '2':2, '3':3, '4':3, 'R':4, 'M':np.nan, '?':np.nan
        }
        yticks    = [0,1,2,3,4]
        yticklabs = ['W','N1','N2','N3','REM']
    else:
        if show_N4:
            # 经典 R&K：W, 1,2,3,4,R  -> y 轴 W, N1, N2, N3, N4, REM
            stage2num = {
                'W':0, '1':1, '2':2, '3':3, '4':4, 'R':5, 'M':np.nan, '?':np.nan,
                # 兼容 AASM 输入
                'N1':1, 'N2':2, 'N3':3, 'REM':5
            }
            yticks    = [0,1,2,3,4,5]
            yticklabs = ['W','N1','N2','N3','N4','REM']  # R&K 语义
        else:
            # 不单独显示 N4：4 合并到 3，y 轴不画 N4
            stage2num = {
                'W':0, '1':1, '2':2, '3':3, '4':3, 'R':4, 'M':np.nan, '?':np.nan,
                # 兼容 AASM 输入
                'N1':1, 'N2':2, 'N3':3, 'REM':4
            }
            yticks    = [0,1,2,3,4]
            yticklabs = ['W','N1','N2','N3','REM']

    # 数字化并构造时间轴（小时）
    labels = np.asarray(labels)

    if np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.floating):
        # 已经是数字标签了，直接用
        y = labels.astype(float)
    else:
        # 字符串标签才用映射
        y = np.array([stage2num.get(str(s), np.nan) for s in labels], dtype=float)

    # y = np.array([stage2num.get(str(s), np.nan) for s in labels], dtype=float)
    t_hours = (start_at_sec + np.arange(len(y)) * epoch_sec) / 3600.0

    # 作图
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    ax.step(t_hours, y, where='post')
    # Lights off / first / last vertical lines
    ax.axvline(x=LightsOffIdx*(epoch_sec/60)/60, linestyle='--', linewidth=1.2, color='blue')
    ax.axvline(x=(LightsOffIdx+FirstIdx)*(epoch_sec/60)/60, linestyle='--', linewidth=1.2, color='red')
    ax.axvline(x=(LightsOffIdx+LastIdx+1)*(epoch_sec/60)/60, linestyle='--', linewidth=1.2, color='black')

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabs)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Stage')
    if title:
        ax.set_title(title)

    if invert_y:
        ax.invert_yaxis()
    ax.grid(True, axis='y', linestyle=':', linewidth=0.8)
    plt.tight_layout()

    if save_path:
        ax.figure.savefig(save_path, dpi=300, format='pdf')

    if created_fig:
        plt.show()

    return ax

def _to_text(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "ignore")
    return str(x)

def _parse_single_hypnogram(edf_path: Path,
                            default_epoch_sec: int = 30) -> Optional[pd.DataFrame]:
    """
    解析单个 Hypnogram EDF，离散成 epoch 标签，返回 DataFrame。
    未评分/未知（R&K='?' 或注释未覆盖到）对应的 epoch 会被直接删除。
    若文件没有‘Sleep stage’注释则返回 None。
    """
    with EdfReader(str(edf_path)) as f:
        onsets, durations, desc = f.readAnnotations()
        onsets    = np.asarray(onsets, dtype=float)
        durations = np.asarray(durations, dtype=float)
        desc      = np.array([_to_text(d).replace("\x00", "").strip() for d in desc], dtype=object)

    # 仅保留睡眠分期注释，并抽取 R&K 阶段码
    mask = np.array([d.startswith("Sleep stage") for d in desc])
    if not mask.any():
        return None

    onset = onsets[mask]
    dur   = durations[mask]
    rk    = np.array([d.split()[-1] for d in desc[mask]], dtype=object)  # W/1/2/3/4/R/M/?

    # 估计 epoch 长度（多数是 30s）；若 dur 异常则回退到默认值
    med = np.median(dur) if dur.size else np.nan
    epoch_sec = int(np.round(med)) if np.isfinite(med) and med > 0 else int(default_epoch_sec)

    # 时间轴离散
    t0   = float(onset.min()) if onset.size else 0.0
    tend = float((onset + dur).max()) if onset.size else 0.0
    n_epochs = int(np.ceil((tend - t0) / epoch_sec)) if tend > t0 else 0
    if n_epochs <= 0:
        return None

    # 默认设为未知：注释未覆盖到的区间也会被视为未知并删除
    labels_rk = np.array(['?'] * n_epochs, dtype=object)

    for s, d, lab in zip(onset - t0, dur, rk):
        i0 = max(0, int(np.floor(s / epoch_sec)))
        i1 = min(n_epochs, int(np.ceil((s + d) / epoch_sec)))
        labels_rk[i0:i1] = lab

    # 删除未知/未评分 epoch（R&K='?'）
    keep = labels_rk != '?'
    if not np.any(keep):
        return None

    # R&K -> AASM 合并（这里不再处理 '?'，因为已经删掉）
    rk2aasm = {'W':'W','R':'REM','1':'N1','2':'N2','3':'N3','4':'N3','M':'W'}
    labels_aasm = np.array([rk2aasm.get(x, x) for x in labels_rk], dtype=object)

    epoch_idx = np.arange(n_epochs, dtype=int)
    t_start = t0 + epoch_idx * epoch_sec
    t_end   = t0 + (epoch_idx + 1) * epoch_sec

    df = pd.DataFrame({
        "epoch_idx": epoch_idx[keep],
        "t_start_s": t_start[keep],
        "t_end_s":   t_end[keep],
        "stage_rk":  labels_rk[keep],
        "stage":     labels_aasm[keep],
        "epoch_sec": epoch_sec,
    })
    return df

def _natural_sort_key(p: Path):
    """
    自然排序键：先按父目录、再按文件名；数字顺序友好。
    """
    import re
    def split_key(s: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]
    return (*split_key(str(p.parent)), *split_key(p.name))

from pathlib import Path
from typing import Union, List
import pandas as pd

def load_sleepstageage(root_dir: Union[str, Path],
                       pattern: str = "*-Hypnogram.edf") -> List[pd.DataFrame]:
    """
    只在 SleepStageAge_HypnogramData 顶层（不递归子目录）查找匹配 pattern 的 EDF，
    解析为 epoch 标签，返回一个 DataFrame 列表。
    """
    root = Path(root_dir).expanduser().resolve()
    ssa_dir = root / "SleepStageAge_HypnogramData"
    if not ssa_dir.exists():
        raise FileNotFoundError(f"未找到目录：{ssa_dir}")

    # 关键修改：glob 不递归子目录；rglob 会递归
    files: List[Path] = sorted(ssa_dir.glob(pattern), key=_natural_sort_key)
    if not files:
        raise FileNotFoundError(f"在 {ssa_dir} 下未匹配到 EDF 文件（pattern='{pattern}'）")

    dfs: List[pd.DataFrame] = []
    for p in files:
        df = _parse_single_hypnogram(p)
        if df is None:
            continue

        base = p.stem  # e.g. SC4001EC-Hypnogram
        subj = base.replace("-Hypnogram", "").replace("_Hypnogram", "")
        df.insert(0, "subject_id", subj)
        df.insert(1, "file_name", p.name)
        df.insert(2, "file_path", str(p.resolve()))
        dfs.append(df)

    if not dfs:
        raise RuntimeError("找到文件但没有包含 'Sleep stage' 注释的数据。")

    return dfs

def count_transitions_to_zero(sleep_sequence, target):
    count = 0
    for i in range(1, len(sleep_sequence)):
        if sleep_sequence[i] == target and sleep_sequence[i-1] != target:
            count += 1
    return count
    
def flatten_all_strength(AllStrength):
    """
    将 AllStrength 中每个 key 对应的数字列表按 key 顺序展开成一个整体 list
    """
    flattened = []
    for key in sorted(AllStrength.keys()):
        flattened.extend(AllStrength[key])
    return flattened

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import linregress
# from typing import Iterable, Optional, Dict, Any

# def plot_regression_with_ci(
#     x_list: Iterable[float],
#     y1_list: Iterable[float],
#     y2_list: Optional[Iterable[float]] = None,
#     *,
#     x_label: Optional[str] = "X",
#     y1_label: Optional[str] = "Y (left)",
#     y2_label: Optional[str] = "Y (right)",
#     title: Optional[str] = "Linear Regression (dual y-axes)",
#     scatter_size: int = 50,
#     show: bool = True
# ) -> Dict[str, Dict[str, Any]]:
#     """
#     x 对 y1（左轴）必画；x 对 y2（右轴）可选。
#     - 使用 scipy.stats.linregress 做线性回归
#     - 图例显示 r 与 p
#     - 返回回归结果字典
#     """
#     x = np.asarray(x_list, dtype=float)
#     y1 = np.asarray(y1_list, dtype=float)

#     if x.shape[0] != y1.shape[0]:
#         raise ValueError("x_list 与 y1_list 长度不一致。")

#     fig, ax_left = plt.subplots(figsize=(8, 5))
#     results: Dict[str, Dict[str, Any]] = {}

#     def _fit_and_plot(ax, x_arr, y_arr, legend_base):
#         mask = np.isfinite(x_arr) & np.isfinite(y_arr)
#         x_clean = x_arr[mask]
#         y_clean = y_arr[mask]
#         if x_clean.size < 2:
#             raise ValueError(f"{legend_base or 'Series'} 有效点不足以回归（<2）。")

#         reg = linregress(x_clean, y_clean)
#         xfit = np.linspace(x_clean.min(), x_clean.max(), 200)
#         yfit = reg.slope * xfit + reg.intercept

#         ax.scatter(x_clean, y_clean, s=scatter_size, alpha=0.85)
#         (ln,) = ax.plot(xfit, yfit, linewidth=2)

#         legend_text = f"{(legend_base or 'Series')}: r={reg.rvalue:.3f}, p={reg.pvalue:.3e}"
#         return reg, ln, legend_text, x_clean.size

#     # 左轴
#     reg_left, ln_left, lab_left, n_left = _fit_and_plot(ax_left, x, y1, y1_label)
#     if x_label is not None:
#         ax_left.set_xlabel(x_label)
#     if y1_label is not None:
#         ax_left.set_ylabel(y1_label)
#     ax_left.grid(True, alpha=0.3)

#     results["left"] = {
#         "slope": reg_left.slope,
#         "intercept": reg_left.intercept,
#         "r": reg_left.rvalue,
#         "p": reg_left.pvalue,
#         "stderr": reg_left.stderr,
#         "intercept_stderr": getattr(reg_left, "intercept_stderr", np.nan),
#         "n": int(n_left),
#     }

#     handles = [ln_left]
#     labels = [lab_left]

#     # 右轴（可选）
#     if y2_list is not None:
#         if x.shape[0] != np.asarray(y2_list).shape[0]:
#             raise ValueError("x_list 与 y2_list 长度不一致。")
#         ax_right = ax_left.twinx()
#         reg_right, ln_right, lab_right, n_right = _fit_and_plot(ax_right, x, np.asarray(y2_list, dtype=float), y2_label)
#         if y2_label is not None:
#             ax_right.set_ylabel(y2_label)

#         results["right"] = {
#             "slope": reg_right.slope,
#             "intercept": reg_right.intercept,
#             "r": reg_right.rvalue,
#             "p": reg_right.pvalue,
#             "stderr": reg_right.stderr,
#             "intercept_stderr": getattr(reg_right, "intercept_stderr", np.nan),
#             "n": int(n_right),
#         }

#         handles.append(ln_right)
#         labels.append(lab_right)

#     if title is not None:
#         ax_left.set_title(title)

#     ax_left.legend(handles, labels, loc="best", frameon=False)
#     plt.tight_layout()
#     if show:
#         plt.show()

#     return results

def canonicalize_loop(loop):
    """
    将loop标准化为最小字典序旋转形式，例如：
    (1,2,3) -> (1,2,3)
    (2,3,1) -> (1,2,3)
    """
    n = len(loop)
    loop = list(loop)
    candidates = [tuple(loop[i:] + loop[:i]) for i in range(n)]
    return min(candidates)
def count_loops(AllCycle):
    """
    输入：AllCycle是一个dict，key为0~21，每个value是一个list of loops
    输出：loop出现次数的dict
    """
    loop_counter = defaultdict(int)
    
    for subject_loops in AllCycle.values():
        for loop in subject_loops:
            canon_loop = tuple(loop) #canonicalize_loop(loop)
            loop_counter[canon_loop] += 1
    
    return loop_counter
def gather_loop_strengths(AllCycle, AllStrength):
    """
    遍历每个subject，将loop标准化，并把对应的strength收集到dict中
    返回：
        loop_strength_dict: {loop_name: [strength1, strength2, ...]}
    """
    loop_strength_dict = defaultdict(list)

    for subj in AllCycle:
        cycles = AllCycle[subj]
        strengths = AllStrength[subj]
        for i, loop in enumerate(cycles):
            canon_loop = tuple(loop) #canonicalize_loop(loop)
            loop_name = '-'.join(map(str, canon_loop))
            strength = strengths[i]
            loop_strength_dict[loop_name].append(strength)

    return loop_strength_dict

def plot_loop_strength_distribution(cycle_dict, epr_dict, ylabel, plotline=False, savepath=None):
    """
    参数
    ----
    cycle_dict : dict
        {k: [loop1, loop2, ...]}，每个 loop 是一个列表，比如 [0,1,2,4,0]
    epr_dict : dict
        {k: [epr1, epr2, ...]}，与 cycle_dict[k] 一一对应
    """
    # 1. 收集所有不同的 loop（以 tuple 形式作为 key）
    state_name_map = {
        0: "Wake",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM",
    }
    loop_set = set()
    for k, loops in cycle_dict.items():
        for loop in loops:
            loop_set.add(tuple(loop))

    # 排序后的 loop 列表（每个元素是 tuple，例如 (0,1,2,4,0)）
    loop_list = sorted(loop_set)
    # 给每个 loop 一个 x 轴 index
    loop_to_x = {loop: i for i, loop in enumerate(loop_list)}

    # 2. 为每个 loop 收集所有 key 的 EPR，用于之后算 mean/std
    loop_to_strengths = {loop: [] for loop in loop_list}

    # 3. 为每个 key 自动分配一个颜色（同一个 key 各点同色）
    keys = sorted(cycle_dict.keys())
    cmap = plt.get_cmap('tab10')
    key_to_color = {k: cmap(i % cmap.N) for i, k in enumerate(keys)}

    # 用来存每个 key 对应的所有 (x, y)，方便之后画连线
    key_to_xs = {k: [] for k in keys}
    key_to_ys = {k: [] for k in keys}

    plt.figure(figsize=(7,5))

    # 为了避免图例重复，只给每个 key 打一次 label
    used_key_labels = set()

    # 4. 画散点：按 key 迭代，key 决定颜色；按 loop 决定 x 位置
    for k in keys:
        loops_k = cycle_dict[k]
        eprs_k = epr_dict[k]
        color = key_to_color[k]

        for loop, val in zip(loops_k, eprs_k):
            loop_t = tuple(loop)
            x = loop_to_x[loop_t]

            # 收集该 loop 的所有 EPR（之后算 mean/std）
            loop_to_strengths[loop_t].append(val)

            # 收集该 key 的所有 (x, y)（之后画连线）
            key_to_xs[k].append(x)
            key_to_ys[k].append(val)

            label = f'ID {k}' if k not in used_key_labels else ""
            if label:
                used_key_labels.add(k)

            plt.scatter(
                x,
                val,
                color=color,
                alpha=0.7,
                edgecolor='black',
                s=200,
                label=label
            )

    # 4.5 把同一个 key 的点用线连起来
    if plotline:
        for k in keys:
            xs = key_to_xs[k]
            ys = key_to_ys[k]
            color = key_to_color[k]
            # 为了保证是按 x 顺序连线，按 x 排一下
            if len(xs) > 1:
                xs, ys = zip(*sorted(zip(xs, ys)))
            plt.plot(xs, ys, color=color, alpha=0.6, linewidth=5)

    # 5. 对每个 loop 计算 mean 和 std，并画黑色误差棒
    x_positions = []
    means = []
    stds = []

    for loop in loop_list:
        vals = np.array(loop_to_strengths[loop])
        x = loop_to_x[loop]
        x_positions.append(x)
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    # plt.errorbar(
    #     x_positions,
    #     means,
    #     yerr=stds,
    #     fmt='o',
    #     color='black',
    #     capsize=5,
    #     markersize=20,
    #     label='Mean ± STD'
    # )

    # 6. x 轴刻度显示为 loop name，例如 "0-1-2-4-0"
    xtick_labels = [
        '-'.join(state_name_map.get(s, str(s)) for s in loop)
        for loop in loop_list
    ]
    plt.xticks(x_positions, xtick_labels, rotation=30, ha='right')

    plt.ylabel(ylabel)
    plt.xlabel('Loop')
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, format='pdf', dpi=300)

    plt.show()

def plot_loop_histogram(loop_counter, savepath=None):
    """
    根据 loop 统计结果绘制直方图
    横坐标用 Wake, N1, N2, N3, REM 组成的 loop 名称
    并按照出现次数从多到少排序
    """
    import matplotlib.pyplot as plt

    # 状态编号到名称的映射
    state_name_map = {
        0: "Wake",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM",
    }

    # 按 count 从大到小排序
    sorted_items = sorted(loop_counter.items(), key=lambda x: x[1], reverse=True)

    loops = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    # 把如 (0,1,2,0) 转成 "Wake-N1-N2-Wake"
    loop_labels = [
        '-'.join(state_name_map.get(s, str(s)) for s in loop)
        for loop in loops
    ]

    plt.figure(figsize=(10, 7))
    bars = plt.bar(loop_labels, counts, color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count')
    plt.title('Loop Occurrence Counts Across 22 Subjects')
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, format='pdf', dpi=300)

    plt.show()

def cycle_edges(cycle):
    c = tuple(cycle)
    return list(zip(c, c[1:] + (c[0],)))

def compute_cycle_epr_from_J(F: np.ndarray, A: np.ndarray, cycle, eps: float = 1e-100) -> float:
    """
    用净流 F(=J, 反对称) 和亲和 A 计算某个环的 EPR：
        sigma_c = psi_c * sum_{edges in c} A_ij
      其中 psi_c = min_{edges in c} J^+_ij, J^+ = max(J, 0).
    要求：cycle 的方向与正流方向一致（即每条边 J^+_ij > 0）。
    """
    edges = cycle_edges(cycle)[:-1]
    Jpos = F #np.maximum(F, 0.0)
    # print(edges, Jpos)
    flows = [Jpos[i, j] for i, j in edges]
    if any(f < 0 for f in flows): #eps
        raise ValueError("该环包含非正流边；请在 J^+ 的支持图上选取环，或调整方向。")
    psi_c = min(flows)
    A_loop = sum(A[i, j] for i, j in edges)
    # print(flows, cycle)
    return psi_c * A_loop
# def compute_cycle_sum(F: np.ndarray, A: np.ndarray, cycle: tuple) -> float:
#     """
#     对给定的转移环路，计算 sum_{(i,j) in cycle} F[i,j] * A[i,j]

#     参数：
#         F: ndarray, 转移频率或通量矩阵，形状 (n, n)
#         A: ndarray, 权重矩阵，形状 (n, n)
#         cycle: tuple/list，例如 (1, 2, 3)，表示环路 1→2→3→1

#     返回：
#         该环路上 Fij * Aij 的累加值（float）
#     """
#     total = 0.0
#     # for i, j in zip(cycle, cycle[1:] + (cycle[0],)):
#     for i, j in zip(cycle, cycle[1:] + [cycle[0]]):
#         f, a = F[i, j], A[i, j]
#         if not (np.isnan(f) or np.isnan(a)):
#             total += f * a
#     return total
    
import numpy as np


def compute_state_probabilities_list(data_dict, num_states=None, state_order=None):
    """
    输入:
        data_dict: dict
            每个键是样本名，每个值是一个状态序列
            状态既可以是整数，也可以是字符串（如 "WAKE", "N1", "N2", "N3", "REM"）

        num_states: int, 可选
            仅在 state_order=None 且输入为整数标签时有意义。
            - 若不给，则自动按实际出现状态数生成长度
            - 若给出，则要求 num_states >= 实际出现状态种类数
              多余位置补 0

        state_order: list, 可选
            指定状态顺序，例如:
                ["WAKE", "N1", "N2", "N3", "REM"]
            若给出，则输出概率向量长度固定为 len(state_order)，
            每个位置严格对应 state_order 中的状态；
            未出现的状态概率为 0。

    输出:
        probs_list: list
            每个元素是一个 ndarray，表示对应序列的状态概率
    """
    probs_list = []

    for seq in data_dict.values():
        seq = np.asarray(seq)

        if seq.size == 0:
            if state_order is not None:
                state_probs = np.zeros(len(state_order), dtype=float)
            else:
                if num_states is None:
                    state_probs = np.array([], dtype=float)
                else:
                    state_probs = np.zeros(num_states, dtype=float)
            probs_list.append(state_probs)
            continue

        # -------- 去掉最后一段连续为 0 / "0" 的部分 --------
        # 对字符串睡眠标签（WAKE/N1/N2...）不会误伤
        mask_nonzero = ~((seq == 0) | (seq == "0"))
        nonzero_idx = np.where(mask_nonzero)[0]
        if nonzero_idx.size > 0:
            last_nonzero = nonzero_idx[-1]
            seq = seq[:last_nonzero + 1]
        # 如果全是 0 / "0"，保持原样
        # ----------------------------------------

        total = len(seq)

        if total == 0:
            if state_order is not None:
                state_probs = np.zeros(len(state_order), dtype=float)
            else:
                if num_states is None:
                    state_probs = np.array([], dtype=float)
                else:
                    state_probs = np.zeros(num_states, dtype=float)

        else:
            if state_order is not None:
                # 严格按用户给定顺序统计
                label_to_idx = {s: i for i, s in enumerate(state_order)}
                unknown_states = set(np.unique(seq)) - set(state_order)
                if len(unknown_states) > 0:
                    raise ValueError(
                        f"序列中存在未包含在 state_order 里的状态: {unknown_states}"
                    )

                state_counts = np.zeros(len(state_order), dtype=int)
                for s in seq:
                    state_counts[label_to_idx[s]] += 1

                state_probs = state_counts / total

            else:
                # 未指定 state_order：自动按出现状态压缩
                unique_states = np.unique(seq)
                K = unique_states.size

                if num_states is None:
                    n_out = K
                else:
                    if num_states < K:
                        raise ValueError(
                            f"num_states={num_states} 小于该序列实际出现的状态种类数 {K}，"
                            "请增大 num_states 或使用 state_order。"
                        )
                    n_out = num_states

                label_to_idx = {s: i for i, s in enumerate(unique_states)}

                state_counts = np.zeros(n_out, dtype=int)
                for s in seq:
                    state_counts[label_to_idx[s]] += 1

                state_probs = state_counts / total

        probs_list.append(state_probs)

    return probs_list

import numpy as np
import matplotlib.pyplot as plt


def fit_markov_chain(states, n_states=None, state_order=None, plot=False, return_mapping=False):
    """
    给定一串离散状态序列，拟合马尔可夫链转移矩阵。
    支持整数标签，也支持字符串标签（如 "Wake", "N1", "N2", "N3", "REM"）。

    参数：
        states: array-like, shape=(T,)
            状态序列，可为整数或字符串标签

        n_states: int, 可选
            最终输出矩阵的状态总数。
            - 当 states 为整数标签且 state_order=None 时：
                若未给出，则自动=实际出现的不同状态数（并压缩重排标签）
                若给出，则认为整数标签本身就在 [0, n_states-1] 中，不再重排
            - 当 state_order 给出时，n_states 会被忽略，矩阵大小由 state_order 决定

        state_order: list, 可选
            指定状态顺序，例如：
                ["Wake", "N1", "N2", "N3", "REM"]
            若给出，则输出矩阵严格按这个顺序排列；
            缺失的状态对应全 0 行/列。
            这也是推荐的方式，尤其适合字符串标签。

        plot: bool
            是否绘制转移矩阵

        return_mapping: bool
            是否同时返回状态到矩阵索引的映射字典

    返回：
        trans_mat: shape=(n_states, n_states) 的转移概率矩阵
        label_to_idx: dict, 可选
            仅当 return_mapping=True 时返回
    """
    states = np.asarray(states)

    if states.size == 0:
        raise ValueError("states 序列为空")

    # ---------- 去除末尾连续为 0 或 '0' 的序列 ----------
    # 仅对整数0或字符串'0'做裁剪；像 "Wake" 不会受影响
    mask_nonzero = ~((states == 0) | (states == "0"))
    nonzero_idx = np.where(mask_nonzero)[0]
    if nonzero_idx.size > 0:
        last_nonzero = nonzero_idx[-1]
        states = states[:last_nonzero + 1]
    # 如果全是 0 / "0"，就不裁剪，保持原样
    # -------------------------------------------

    unique_states = np.unique(states)

    # =============== 决定状态映射 ===============
    if state_order is not None:
        # 用户显式指定状态顺序
        state_order = list(state_order)
        n_states_final = len(state_order)
        label_to_idx = {s: i for i, s in enumerate(state_order)}

        unknown_states = set(unique_states) - set(state_order)
        if len(unknown_states) > 0:
            raise ValueError(
                f"states 中存在未包含在 state_order 里的状态: {unknown_states}"
            )

        states_mapped = np.array([label_to_idx[s] for s in states], dtype=int)

    else:
        # 未指定 state_order
        if np.issubdtype(states.dtype, np.integer):
            # -------- 整数标签逻辑：保持你原来的设计 --------
            unique_states = np.unique(states.astype(int))
            states = states.astype(int)

            if n_states is None:
                # 自动压缩到 0..k-1
                n_states_final = unique_states.size
                label_to_idx = {s: i for i, s in enumerate(unique_states)}
                states_mapped = np.array([label_to_idx[s] for s in states], dtype=int)
            else:
                # 用户指定：认为标签本身就是 0..n_states-1
                if states.min() < 0 or states.max() >= n_states:
                    raise ValueError(
                        f"指定 n_states={n_states} 时，状态标签必须在 [0, {n_states-1}] 区间内，"
                        f"但实际出现的最小/最大标签为 {states.min()}/{states.max()}。"
                    )
                n_states_final = n_states
                label_to_idx = {i: i for i in range(n_states_final)}
                states_mapped = states.copy()
        else:
            # -------- 字符串/混合标签逻辑 --------
            if n_states is not None:
                raise ValueError(
                    "当 states 为字符串标签时，建议使用 state_order 指定状态顺序；"
                    "若不指定 state_order，则会自动按出现状态压缩映射，此时不需要传 n_states。"
                )

            n_states_final = unique_states.size
            label_to_idx = {s: i for i, s in enumerate(unique_states)}
            states_mapped = np.array([label_to_idx[s] for s in states], dtype=int)
    # =======================================================

    # 如果不足两步，就返回全零矩阵
    if states_mapped.size < 2:
        trans_mat = np.zeros((n_states_final, n_states_final), dtype=float)
        if plot:
            plt.imshow(trans_mat, cmap="Blues")
            plt.title("Markov Transition Matrix")
            plt.xlabel("To State")
            plt.ylabel("From State")
            plt.colorbar(label="P(j | i)")
            plt.show()
        return (trans_mat, label_to_idx) if return_mapping else trans_mat

    # 计数矩阵
    counts = np.zeros((n_states_final, n_states_final), dtype=int)
    for i, j in zip(states_mapped[:-1], states_mapped[1:]):
        counts[i, j] += 1

    # 转换为概率矩阵
    counts_sum = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        trans_mat = counts / counts_sum
    trans_mat[np.isnan(trans_mat)] = 0

    if plot:
        plt.imshow(trans_mat, cmap="Blues")
        plt.title("Markov Transition Matrix")
        plt.xlabel("To State")
        plt.ylabel("From State")
        plt.colorbar(label="P(j | i)")
        plt.show()

    return (trans_mat, label_to_idx) if return_mapping else trans_mat
    
def compute_steady_state_and_entropy(T, tol=1e-12):
    """
    给定转移率矩阵 T（T[i, j] 表示从 i -> j 的速率），
    求解稳态分布 P 及其每个状态的熵贡献。
    若存在某个状态 i 使得第 i 行和第 i 列全为 0，则先将对应行列删去，
    在剩余子矩阵上计算稳态分布，最后在对应位置插入 0。
    """
    T = np.array(T, dtype=float)
    n = T.shape[0]

    # 找到“第 i 行第 i 列全为 0”的孤立状态
    row_zero = np.all(T == 0, axis=1)
    col_zero = np.all(T == 0, axis=0)
    remove_mask = row_zero & col_zero       # 要丢弃的状态
    keep_mask = ~remove_mask                # 保留的状态

    # 特殊情况：如果所有状态都被移除，则直接返回全 0
    if not np.any(keep_mask):
        P = np.zeros(n)
        S = np.zeros(n)
        total_entropy = 0.0
        return P, S, total_entropy

    # 构造子矩阵（只包含非孤立状态）
    T_reduced = T[keep_mask][:, keep_mask]
    m = T_reduced.shape[0]

    # ===== 在子矩阵上按原有逻辑计算稳态 =====
    row_sums = T_reduced.sum(axis=1)
    print(row_sums)

    # 你原来的实现：M = T^T - I
    M = T_reduced.T - np.eye(m)

    # 添加归一化条件
    A = np.vstack([M, np.ones(m)])
    b = np.zeros(m + 1)
    b[-1] = 1

    # 解 M P_reduced = 0 with sum(P_reduced) = 1
    P_reduced, *_ = np.linalg.lstsq(A, b, rcond=None)

    # ===== 处理数值误差：去掉微小负值并重新归一化 =====
    # 1. 把绝对值非常小的数直接视为 0（包括 -1e-16 这种）
    P_reduced[np.abs(P_reduced) < tol] = 0.0
    # 2. 对真正小于 0 的部分全部截断为 0
    P_reduced[P_reduced < 0] = 0.0
    # 3. 再归一化一次，保证和为 1
    s = P_reduced.sum()
    if s <= 0:
        raise RuntimeError(
            "求得的稳态分布和为非正值，可能 T 不可约或不是合理的转移矩阵，请检查 T。"
        )
    P_reduced /= s

    # ===== 把结果插回到完整维度 =====
    P = np.zeros(n, dtype=float)
    P[keep_mask] = P_reduced
    # 被移除的状态对应概率为 0（P[remove_mask] 默认就是 0）

    # 计算每个状态的熵贡献
    S = np.zeros_like(P)
    for i, p in enumerate(P):
        S[i] = -p * np.log(p) if p > 0 else 0.0  # 避免 log(0)

    total_entropy = np.sum(S)

    return P, S, total_entropy

def compute_flux_matrix(T, P):
    """
    输入:
        T: n x n 转移率矩阵，T[i, j] 表示从 i → j 的速率
        P: 稳态概率分布向量，长度 n

    输出:
        F: n x n 概率流矩阵，F[i, j] = T[i, j]*P[i] - T[j, i]*P[j]
    """
    T = np.array(T, dtype=float)
    P = np.array(P, dtype=float)
    n = T.shape[0]

    F = np.zeros((n, n))
    fs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            F[i, j] = -T[i, j] * P[i] + T[j, i] * P[j]
            fs[i, j] = T[i, j] * P[i]
    
    return F, fs
    
def compute_affinity_matrix(T, p, eps=1e-10):
    T = np.asarray(T, float)
    p = np.asarray(p, float).reshape(-1)
    Phi = p[:, None] * T                 # Phi_ij = p_i * T_ij
    A = np.log((Phi + eps) / (Phi.T + eps))
    np.fill_diagonal(A, 0.0)
    # 数值对称化以确保反对称
    A = 0.5 * (A - A.T)
    return A

def compute_J_matrix(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    输入：
    - T: 转移概率矩阵 (n x n)，T[i, j] 是从 i 到 j 的转移概率
    - P: 稳态分布 (n,) 向量，P[i] 是状态 i 的概率

    输出：
    - J: (n x n) 概率流矩阵，J[i, j] = P[j] T[j, i] - min(P[j] T[j, i], P[i] T[i, j])
    """
    n = T.shape[0]
    J = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            flow_ji = P[j] * T[j, i]
            flow_ij = P[i] * T[i, j]
            J[i, j] = flow_ij - min(flow_ji, flow_ij)

    return J

def filter_positive_flux_cycles(cycles, J, tol=0.0):
    """
    假设：J[i, j] 表示从 i -> j 的净流（不调整索引顺序）。
    过程：依次检查 cycles；一旦某个环在当前 J 上所有边的净流都 > tol，
         取该环上的 min_flux，并把该值从该环每条边上减去（就地修改 J），
         然后继续在已更新的 J 上检查后续环。

    输入：
    - cycles: list[list[int]]，每个环如 [1, 3, 2]
    - J: (n x n) 概率流矩阵，J[i, j] = i -> j 的净流
    - tol: 数值容差，默认 0（可设 1e-12 等避免浮点误差）

    输出：
    - valid_cycles: list[tuple[list[int], float]]，每项为 (cycle, min_flux)
    """
    valid_cycles = []

    for cycle in cycles:
        n = len(cycle)
        is_valid = True
        flux_values = []

        # 检查该环是否在当前 J 上所有边都为正
        for k in range(n):
            src = cycle[k]
            dst = cycle[(k + 1) % n]  # 闭环
            flux = J[src, dst]        # 不改 ij 顺序：边 src->dst 取 J[src, dst]
            if flux <= tol:
                is_valid = False
                break
            flux_values.append(flux)

        # 若有效，则减去该环的最小边流量
        if is_valid:
            min_flux = min(flux_values)
            for k in range(n):
                src = cycle[k]
                dst = cycle[(k + 1) % n]
                J[src, dst] -= min_flux
                if tol > 0.0 and abs(J[src, dst]) < tol:
                    J[src, dst] = 0.0  # 可选：去掉数值噪声
            valid_cycles.append((cycle, float(min_flux)))

    return valid_cycles


# def filter_positive_flux_cycles(cycles, J):
#     """
#     输入：
#     - cycles: list of cycles，每个 cycle 是一组节点，如 [1, 3, 2]
#     - J: 概率流矩阵 (n x n)，J[i, j] 表示从 j -> i 的净流

#     输出：
#     - valid_cycles: list of (cycle, min_flux) 的元组，表示所有满足条件的环及其强度
#     """
#     valid_cycles = []

#     for cycle in cycles:
#         n = len(cycle)
#         is_valid = True
#         flux_values = []

#         for i in range(n):
#             src = cycle[i]
#             dst = cycle[(i + 1) % n]  # 确保形成闭环
#             flux = J[src, dst]  # 注意：J_{ij} 表示 j -> i 的流，所以是 J[dst, src]
#             if flux <= 0:
#                 is_valid = False
#                 break
#             flux_values.append(flux)

#         if is_valid:
#             min_flux = min(flux_values)
#             valid_cycles.append((cycle, min_flux))

#     return valid_cycles
    
def unique_cycles(n_nodes, cycle_length):
    nodes = list(range(n_nodes))  # 改为 0-based：0 到 n_nodes-1
    all_cycles = set()

    for combo in combinations(nodes, cycle_length):
        for perm in permutations(combo):
            # 将最小节点放在第一个位置，标准化（消除旋转冗余）
            min_idx = perm.index(min(perm))
            rotated = perm[min_idx:] + perm[:min_idx]
            # 排除方向对称：只保留一种方向
            if rotated[::-1] not in all_cycles:
                all_cycles.add(rotated)

    return all_cycles

def analyze_transition_matrices(Tmat, plot=True, probs=None, eps=1e-10):
    if isinstance(Tmat, np.ndarray) and Tmat.ndim == 2:
        Tmat = [Tmat]

    def _normalize_prob(p, name="probs"):
        p = np.asarray(p, dtype=float).ravel()
        if np.any(p < 0):
            raise ValueError(f"{name} 含有负值。")
        s = p.sum()
        if s <= 0:
            raise ValueError(f"{name} 求和<=0，无法归一化。")
        if not np.isclose(s, 1.0):
            p = p / s
        return p

    # === 这里开始是新的 probs 处理逻辑 ===
    probs_list = [None] * len(Tmat)  # 默认用各自稳态
    if probs is None:
        pass
    elif isinstance(probs, (list, tuple)):
        if len(probs) != len(Tmat):
            raise ValueError("probs 为列表/元组时，长度必须与 Tmat 一致。")
        for i, (T, p_raw) in enumerate(zip(Tmat, probs)):
            if p_raw is not None:
                p = _normalize_prob(p_raw, name=f"probs[{i}]")
                if p.shape[0] != T.shape[0]:
                    raise ValueError(f"[Matrix {i}] probs[{i}] 长度与矩阵状态数不符。")
                probs_list[i] = p
    elif isinstance(probs, np.ndarray) and probs.ndim == 2:
        # 支持 (len(Tmat), K) 或 (K, len(Tmat))
        nmat = len(Tmat)
        nstate = Tmat[0].shape[0]
        if not all(T.shape[0] == nstate for T in Tmat):
            raise ValueError("Tmat 中矩阵的状态数不一致，无法用 2D probs 对齐。")
        if probs.shape == (nmat, nstate):
            for i in range(nmat):
                probs_list[i] = _normalize_prob(probs[i], name=f"probs[{i},:]")
        elif probs.shape == (nstate, nmat):
            for i in range(nmat):
                probs_list[i] = _normalize_prob(probs[:, i], name=f"probs[:,{i}]")
        else:
            raise ValueError(f"2D probs 形状不匹配，应为 ({nmat},{nstate}) 或 ({nstate},{nmat})，当前为 {probs.shape}。")
    elif isinstance(probs, dict):
        # 键为状态数
        for i, T in enumerate(Tmat):
            n = T.shape[0]
            if n in probs and probs[n] is not None:
                p = _normalize_prob(probs[n], name=f"probs[{n}]")
                if p.shape[0] != n:
                    raise ValueError(f"[Matrix {i}] probs[{n}] 长度与矩阵状态数不符。")
                probs_list[i] = p
    else:
        # 当作单个向量广播（老逻辑）
        p_single = _normalize_prob(probs, name="probs")
        L = p_single.shape[0]
        for i, T in enumerate(Tmat):
            if T.shape[0] == L:
                probs_list[i] = p_single
            # 不匹配则保留 None -> 用稳态
    # === 新的 probs 处理逻辑结束 ===

    Pss, Sss, Fss, FS, Ass, Jss, EPRss = [], [], [], [], [], [], []

    for n, (T, P_override) in enumerate(zip(Tmat, probs_list)):
        P_comp, S, _ = compute_steady_state_and_entropy(T)
        print(f"[Matrix {n}] P (steady-state) =", [f"{p:.2e}" for p in P_comp])
        print(f"[Matrix {n}] S =", [f"{s:.2e}" for s in S])

        P_eff = P_override if P_override is not None else P_comp
        if P_override is not None:
            print(f"[Matrix {n}] 使用传入 probs 进行后续计算。")

        F, fs = compute_flux_matrix(T, P_eff)
        A = compute_affinity_matrix(T, P_eff, eps=eps)
        J = compute_J_matrix(T, P_eff)

        fs_no_diag = fs.copy(); A_no_diag = A.copy()
        np.fill_diagonal(fs_no_diag, np.nan)
        np.fill_diagonal(A_no_diag, np.nan)
        EPR = np.nansum(fs_no_diag * A_no_diag) #0.5 * 
        # EPR = 0.5 * np.sum(F * A)
        print(f"[Matrix {n}] EPR = {EPR:.2e}")

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            im0 = axes[0].imshow(F, cmap="Blues"); axes[0].set_title("$F_{ij}$"); plt.colorbar(im0, ax=axes[0])
            im1 = axes[1].imshow(A, cmap="Blues"); axes[1].set_title("$A_{ij}$"); plt.colorbar(im1, ax=axes[1])
            im2 = axes[2].imshow(J, cmap="Blues"); axes[2].set_title("$J_{ij}$"); plt.colorbar(im2, ax=axes[2])
            for ax in axes: ax.set_xlabel("j"); ax.set_ylabel("i")
            plt.tight_layout(); plt.show()

        Pss.append(P_comp); Sss.append(S); Fss.append(F); FS.append(fs); Ass.append(A); Jss.append(J); EPRss.append(EPR)
    return Pss, Sss, Fss, FS, Ass, Jss, EPRss
    
def find_valid_flux_cycles(F, A, J_input, if_print=False, eps=1e-12): #cycle_sets, 
    """
    输入：
        J_input: 单个 J 矩阵 (ndarray) 或包含多个 J 的 list
        cycle_sets: list of cycle lists，例如 [cycles_3, cycles_4, cycles_5]

    输出：
        AllCycle: dict，每个样本的有效环列表
        AllStrength: dict，每个样本中每个环的最小流强度
    """
    # 如果输入是单个矩阵，转换为列表
    if isinstance(J_input, np.ndarray) and J_input.ndim == 2:
        J_input = [J_input]
    if isinstance(F, np.ndarray) and F.ndim == 2:
        F = [F]
    if isinstance(A, np.ndarray) and A.ndim == 2:
        A = [A]

    AllCycle = {}
    AllStrength = {}
    AllEPR = {}

    for n, J in enumerate(J_input):
        AllCycle[n] = []
        AllStrength[n] = []
        AllEPR[n] = []
        if if_print:
            print(f"\n[Sample {n}]")

        # for cycles in cycle_sets:
        # 去除首尾重复（闭环形式）
        # clean_cycles = [c[:-1] if c[0] == c[-1] else c for c in cycles]

        # 过滤有效环
        # valid_cycles = filter_positive_flux_cycles(clean_cycles, J.copy())
        valid_cycles = decompose_cycles_greedy(J, order="natural")

        for cycle, strength in valid_cycles:
            epr = compute_cycle_epr_from_J(J, A[n], cycle, eps=eps)
            AllCycle[n].append(cycle)
            AllStrength[n].append(strength)
            AllEPR[n].append(epr)
            if if_print:
                print(f"  Cycle: {cycle}, Strength: {strength:.2e}, EPR: {epr:.2e}")

    return AllCycle, AllStrength, AllEPR
    
def fit_shared_markov_chain(state_dict, n_states=None, plot=False):
    """
    给定多个时序状态序列，拟合一个共享的马尔可夫转移矩阵。

    参数：
        state_dict: dict，每个 key 对应一个状态序列（整数数组）
        n_states: 状态总数（自动根据数据最大状态值推断）
        plot: 是否绘制转移矩阵图

    返回：
        trans_mat: shape=(n_states, n_states)，转移概率矩阵
    """
    all_sequences = list(state_dict.values())

    if n_states is None:
        n_states = max(np.max(seq) for seq in all_sequences) + 1

    counts = np.zeros((n_states, n_states), dtype=int)

    for seq in all_sequences:
        for i, j in zip(seq[:-1], seq[1:]):
            counts[i, j] += 1

    # 转移概率矩阵
    with np.errstate(divide='ignore', invalid='ignore'):
        trans_mat = counts / counts.sum(axis=1, keepdims=True)
        trans_mat[np.isnan(trans_mat)] = 0

    if plot:
        plt.imshow(trans_mat, cmap='Blues')
        plt.title("Shared Markov Transition Matrix")
        plt.xlabel("To State")
        plt.ylabel("From State")
        plt.colorbar(label="P(j | i)")
        plt.show()

    return trans_mat

from sklearn.manifold import SpectralEmbedding
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_potential_surface(plt_idx, Tmat, Pss, grid_size=50, background_U=5.0, alpha=0.3, cycles=None):
    """
    可视化状态概率对应的潜在势能景观，并在旁边绘制2D嵌入+循环路径箭头。

    参数：
        Tmat: np.ndarray or list[np.ndarray] — 转移矩阵或矩阵列表
        Pss: np.ndarray or list[np.ndarray] — 稳态概率分布或列表
        grid_size: int — 网格分辨率
        background_U: float — 非状态点默认U值（高势能）
        alpha: float — surface透明度
        cycles: list[tuple[int]] — 可选，表示在2D嵌入图中画出的状态循环路径
    """
    
    if isinstance(Tmat, np.ndarray):
        Tmat = [Tmat]
    if isinstance(Pss, np.ndarray):
        Pss = [Pss]

    state_labels = ['Wake', 'N1', 'N2', 'N3', 'REM']

    for n in range(len(Tmat)):
        T = Tmat[n]
        P = Pss[n]
        affinity = (T + T.T) / 2

        embedder = SpectralEmbedding(n_components=2, affinity='precomputed')
        X_2d = embedder.fit_transform(affinity)
        U_points = -np.log(P + 1e-12)

        # --- 创建网格并填充势能 ---
        x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
        y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                             np.linspace(y_min, y_max, grid_size))
        zz = np.full_like(xx, background_U)

        for i in range(5):
            xi = np.argmin(np.abs(xx[0, :] - X_2d[i, 0]))
            yi = np.argmin(np.abs(yy[:, 0] - X_2d[i, 1]))
            zz[yi, xi] = U_points[i]

        # --- 创建图形 ---
        fig = plt.figure(figsize=(16, 6))

        # 左侧：3D势能图
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax3d.plot_surface(xx, yy, zz, cmap='viridis', edgecolor='none', alpha=alpha)
        ax3d.scatter(X_2d[:, 0], X_2d[:, 1], U_points, c='red', s=80)

        for i, label in enumerate(state_labels):
            text = f"{label}\nU={U_points[i]:.2f}"
            ax3d.text(X_2d[i, 0], X_2d[i, 1], U_points[i] + 0.5, text,
                      ha='center', fontsize=15, color='black')

        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel(r'U')
        # ax3d.set_title(f'3D Potential Surface (Sample {n})')
        fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)

        # 右侧：2D嵌入图 + Cycle
        ax2d = fig.add_subplot(1, 2, 2)
        ax2d.scatter(X_2d[:, 0], X_2d[:, 1], c='red', s=80)

        for i, label in enumerate(state_labels):
            ax2d.text(X_2d[i, 0], X_2d[i, 1], label, fontsize=20, ha='center', va='center', color='black')

            if cycles is not None:
                # 使用 tab20 中的前10个深色（偶数索引）
                full_cmap = cm.get_cmap('tab20')
                color_list = [full_cmap(i) for i in range(0, 20, 2)]  # 深色部分
    
                num_cycles = len(cycles)
                if num_cycles > len(color_list):
                    # 循环使用颜色，但避免相邻颜色太接近
                    color_list = color_list * ((num_cycles // len(color_list)) + 1)
    
                offset_scale = 0.003 * (X_2d[:, 0].max() - X_2d[:, 0].min())
    
                for idx, cycle in enumerate(cycles):
                    color = color_list[idx]
                    for i in range(len(cycle)):
                        start = cycle[i]
                        end = cycle[(i + 1) % len(cycle)]
    
                        x0, y0 = X_2d[start]
                        x1, y1 = X_2d[end]
                        dx = x1 - x0
                        dy = y1 - y0
                        length = np.sqrt(dx**2 + dy**2)
    
                        if length == 0:
                            norm_vec = np.array([0, 0])
                        else:
                            norm_vec = np.array([-dy, dx]) / length
    
                        offset = norm_vec * offset_scale * idx
                        start_shifted = np.array([x0, y0]) + offset
                        end_shifted = np.array([x1, y1]) + offset

                        ax2d.annotate("",
                            xy=end_shifted,
                            xytext=start_shifted,
                            arrowprops=dict(
                                arrowstyle='-|>',
                                color=color,
                                lw=2,
                                mutation_scale=30
                            )
                        )


        ax2d.set_xlabel('X')
        ax2d.set_ylabel('Y')
        ax2d.set_title(f'Cycles of Sample {plt_idx}')
        ax2d.grid(True)

        plt.tight_layout()
        plt.savefig(f"Figures/TemporalLoops/Cycles{plt_idx}.pdf", format='pdf', dpi=300)
        plt.show()

def jplus_from_TP(T: np.ndarray, P: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """
    给定转移矩阵 T（行 i 到列 j 的率/概率）与稳态分布 P，返回 J^+ = max(T_ij P_i - T_ji P_j, 0).
    """
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    # 流矩阵：Net = outward - inward（按边向量化）
    Net = T * P[:, None] - (T * P[:, None]).T
    Jplus = np.maximum(Net, 0.0)
    if tol > 0:
        Jplus[Jplus < tol] = 0.0
    return Jplus

def _build_adj(J: np.ndarray, tol: float) -> List[List[int]]:
    """从正流矩阵构建邻接表。"""
    n = J.shape[0]
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if J[i, j] > tol:
                adj[i].append(j)
    return adj

def _find_any_directed_cycle(J: np.ndarray, node_order: List[int], tol: float) -> Optional[List[int]]:
    """
    在当前正流图中找一条有向环，返回节点序列（包含闭合：... -> start）。
    若不存在，返回 None。
    """
    n = J.shape[0]
    adj = _build_adj(J, tol)

    visited = [0] * n
    onstack = [False] * n
    stack: List[int] = []
    pos = {}  # node -> index in stack (仅在 onstack=True 时有效)

    def dfs(u: int) -> Optional[List[int]]:
        visited[u] = 1
        onstack[u] = True
        pos[u] = len(stack)
        stack.append(u)

        for v in adj[u]:
            if J[u, v] <= tol:
                continue
            if visited[v] == 0:
                cyc = dfs(v)
                if cyc is not None:
                    return cyc
            elif onstack[v]:
                # 在栈中找到 v -> 构造环：stack[pos[v]:] + [v]
                start_idx = pos[v]
                cycle_nodes = stack[start_idx:] + [v]
                return cycle_nodes

        # 回溯
        onstack[u] = False
        stack.pop()
        pos.pop(u, None)
        return None

    for s in node_order:
        if visited[s] == 0:
            cyc = dfs(s)
            if cyc is not None:
                return cyc

    return None

def _cycle_bottleneck(J: np.ndarray, cycle_nodes: List[int]) -> float:
    """计算环上瓶颈流量（最小边流）。cycle_nodes 末尾等于起点。"""
    b = float('inf')
    for a, bnode in zip(cycle_nodes[:-1], cycle_nodes[1:]):
        b = min(b, J[a, bnode])
    return b

def _subtract_cycle(J: np.ndarray, cycle_nodes: List[int], w: float, tol: float):
    """从 J 上沿环扣除强度 w，并做容差截断。"""
    for a, b in zip(cycle_nodes[:-1], cycle_nodes[1:]):
        J[a, b] -= w
        if J[a, b] < tol:
            J[a, b] = 0.0

def decompose_cycles_greedy(J_plus: np.ndarray,
                            order: str = "natural",
                            seed: Optional[int] = None,
                            tol: float = 1e-12,
                            max_iters: int = 10**6) -> List[Tuple[List[int], float]]:
    """
    将非负守恒流 J_plus 做“贪心环分解”：
      - 每次找任意一条有向环（受 node order 影响），
      - 取环瓶颈作为权重，沿环扣减，
      - 记录 (cycle_nodes, weight)，直到无环。
    参数
    ----
    order : "natural" | "random"
        - "natural": 节点顺序为 0..n-1（确定性，依赖输入顺序）
        - "random":  每步随机打乱节点扫描顺序（不唯一）
    seed : 随机种子（当 order="random" 时用于复现）
    tol :  数值容差，小于该值的流量视为 0
    返回
    ----
    cycles : 列表 [(cycle_nodes, weight), ...]，
             其中 cycle_nodes 如 [n0, n1, ..., n0]（闭合），weight > 0.
    """
    J = np.array(J_plus, dtype=float)  # 复制，避免就地修改原矩阵
    n = J.shape[0]
    cycles: List[Tuple[List[int], float]] = []

    if order not in ("natural", "random"):
        raise ValueError("order must be 'natural' or 'random'")

    rng = random.Random(seed)

    iters = 0
    while iters < max_iters:
        iters += 1
        if order == "natural":
            node_order = list(range(n))
        else:
            node_order = list(range(n))
            rng.shuffle(node_order)

        cycle_nodes = _find_any_directed_cycle(J, node_order, tol)
        if cycle_nodes is None:
            break  # 没有环了

        w = _cycle_bottleneck(J, cycle_nodes)
        if w <= tol:
            # 保险起见，防止极小值导致死循环：将极小边归零并继续
            _subtract_cycle(J, cycle_nodes, w, tol)
            continue

        cycles.append((cycle_nodes, w))
        _subtract_cycle(J, cycle_nodes, w, tol)

    return cycles

def reconstruct_from_cycles(n: int, cycles: List[Tuple[List[int], float]]) -> np.ndarray:
    """由 (cycle, weight) 列表重构流矩阵，用于核验分解结果。"""
    Jrec = np.zeros((n, n), dtype=float)
    for nodes, w in cycles:
        for a, b in zip(nodes[:-1], nodes[1:]):
            Jrec[a, b] += w
    return Jrec

def flow_conservation_error(J: np.ndarray) -> float:
    """
    返回 max_i |sum_j J_ij - sum_j J_ji|，用来检查结点守恒。
    若输入 J 为 J^+（只保留正流），且系统稳态，则该误差应接近 0。
    """
    outflow = J.sum(axis=1)
    inflow = J.sum(axis=0)
    return float(np.max(np.abs(outflow - inflow)))
def canon_cycle(t: Cycle, ignore_rotation: bool = True, ignore_direction: bool = False) -> Cycle:
    """
    规范化一个有向环（元组）。
    - ignore_rotation=True: 忽略起点（把所有旋转看成同一环），取字典序最小旋转。
    - ignore_direction=True: 方向不区分（反向也视为同一环），取正向/反向两边最小的那个表示。
    说明：若 t 是闭合的 (0,1,2,0)，会自动去掉最后一个重复点。
    """
    # 去闭合尾巴
    if len(t) > 1 and t[0] == t[-1]:
        t = t[:-1]
    if not ignore_rotation and not ignore_direction:
        return t

    def min_rotation(seq: Tuple[int, ...]) -> Tuple[int, ...]:
        n = len(seq)
        if n == 0:
            return seq
        rots = [seq[i:] + seq[:i] for i in range(n)]
        return min(rots)

    t_fwd = min_rotation(t) if ignore_rotation else t
    if not ignore_direction:
        return t_fwd

    # 方向不区分时，同时考虑反向的最小旋转
    t_rev = tuple(reversed(t))
    t_rev = min_rotation(t_rev) if ignore_rotation else t_rev
    return t_fwd if t_fwd <= t_rev else t_rev
def normalize_cycles(
    cycles: Iterable[Cycle],
    ignore_rotation: bool = True,
    ignore_direction: bool = False,
    keep_multiplicity: bool = False,
):
    """
    把一个环列表规范化为可比较的对象：
    - keep_multiplicity=False: 返回 set（忽略重复次数）
    - keep_multiplicity=True: 返回 Counter（保留重复次数）
    """
    canoned = [canon_cycle(c, ignore_rotation, ignore_direction) for c in cycles]
    return Counter(canoned) if keep_multiplicity else set(canoned)
def diff_keys_between_dicts(
    A: DictCycles,
    B: DictCycles,
    ignore_rotation: bool = True,
    ignore_direction: bool = False,
    keep_multiplicity: bool = False,
):
    """
    比较两个 {key: [cycle-tuples]}，返回“内部环集合不同”的 key 列表。
    - ignore_rotation=True  -> (0,1,2) 与 (1,2,0) 视为相同
    - ignore_direction=True -> (0,1,2) 与 (0,2,1) 也视为相同
    - keep_multiplicity=True -> 比较时保留重复次数（用 Counter）
    """
    keys = set(A) | set(B)
    diff = []
    for k in sorted(keys):
        cycles_A = A.get(k, [])
        cycles_B = B.get(k, [])
        norm_A = normalize_cycles(cycles_A, ignore_rotation, ignore_direction, keep_multiplicity)
        norm_B = normalize_cycles(cycles_B, ignore_rotation, ignore_direction, keep_multiplicity)
        if norm_A != norm_B:
            diff.append(k)
    return diff

def leading_zeros_and_first_idx_np(data, tol=0.0, treat_nan_as_zero=False):
    """
    对 data 里的每个一维数组，找前面连续为“wake”的长度，
    并给出第一个“非 wake”元素的索引。

    这里把这些都当成 wake:
    - 数值：0，或 |x| <= tol
    - 字符串：'W', 'w', 'Wake', 'wake'
    - 可选：NaN 当作 wake（treat_nan_as_zero=True 时）

    返回：
        { key: {"leading_zeros": n, "first_nonzero_index": idx}, ... }
        这里变量名沿用原来的，你可以理解成
        leading_zeros 其实是 leading_wake_count
        first_nonzero_index 其实是 first_non_wake_index
    """
    out = {}
    wake_strs = {"W", "w", "Wake", "wake"}

    for k, arr in data.items():
        a = np.asarray(arr)

        if a.ndim != 1:
            raise ValueError(f"{k} 不是一维数组")
        if a.size == 0:
            out[k] = {"leading_zeros": 0, "first_nonzero_index": None}
            continue

        # 分两种情况：数值型 vs 字符串/混合型
        if np.issubdtype(a.dtype, np.number):
            # 数值：0 或 |x|<=tol 视为 wake
            wake_mask = (np.abs(a) <= tol)
            if treat_nan_as_zero:
                wake_mask = wake_mask | np.isnan(a)
        else:
            # 非纯数值（比如 'W'、object 数组）
            # 先做一个逐元素判断
            wake_mask = []
            for x in a:
                is_wake = False

                # 1) 字符串类的 wake
                if isinstance(x, str):
                    if x in wake_strs:
                        is_wake = True

                # 2) 数值 0 也当成 wake
                elif isinstance(x, (int, float, np.number)):
                    if abs(x) <= tol:
                        is_wake = True

                # 3) NaN 是否当成 wake
                if treat_nan_as_zero:
                    try:
                        if np.isnan(x):
                            is_wake = True
                    except TypeError:
                        pass

                wake_mask.append(is_wake)

            wake_mask = np.array(wake_mask, dtype=bool)

        # 非 wake 的位置
        non_wake_mask = ~wake_mask

        if non_wake_mask.any():
            first_idx = int(np.argmax(non_wake_mask))
            out[k] = {
                "leading_zeros": first_idx,
                "first_nonzero_index": first_idx,
            }
        else:
            # 全都是 wake
            out[k] = {
                "leading_zeros": int(a.size),
                "first_nonzero_index": None,
            }

    return out

def first_passage_counts_array(
    data,
    target_state,
    mode="leave",
    tol=0.0,
    treat_nan_as_target=False,
    target_aliases=None,
):
    """
    data: dict[str, sequence]
        每个 key 对应一个状态序列（list/array）

    target_state:
        目标状态，比如 0 或 'W' 等。
        - 如果是数值：|x - target_state| <= tol 视为在目标状态
        - 如果是字符串：x == target_state 视为在目标状态
    mode: {"leave", "arrive"}
        - "leave":  计算“首次离开目标状态”需要多少步
            => 从头开始数连续的“在目标状态”的长度
        - "arrive": 计算“首次到达目标状态”需要多少步
            => 从头开始数连续的“非目标状态”的长度
    tol: float
        对数值目标状态的容差
    treat_nan_as_target: bool
        若为 True，则 NaN 也当成目标状态
    target_aliases: set 或 list，可选
        附加的一组“也算作目标状态”的值（比如 {"W", "w", "Wake", "wake", 0}）

    返回:
        list[int]，长度与 data 中 key 的个数相同，
        第 i 个元素对应 data 中第 i 个序列的步数。
    """

    if mode not in {"leave", "arrive"}:
        raise ValueError("mode 必须是 'leave' 或 'arrive'")

    if target_aliases is None:
        target_aliases = set()
    else:
        target_aliases = set(target_aliases)

    counts = []

    def is_target(x):
        """判断单个元素是否处于目标状态"""
        # 1) 字符串类
        if isinstance(x, str):
            if x == target_state or x in target_aliases:
                return True

        # 2) 数值类
        elif isinstance(x, (int, float, np.number)):
            # 如果 target_state 也是数值，就按差值判断
            if isinstance(target_state, (int, float, np.number)):
                if abs(x - target_state) <= tol:
                    return True
            # 否则可以把 x 放进 alias 里对比（可选）
            if x in target_aliases:
                return True

        # 3) NaN 单独处理（可选）
        if treat_nan_as_target:
            try:
                if np.isnan(x):
                    return True
            except TypeError:
                pass

        return False

    # 按字典插入顺序遍历
    for seq in data.values():
        arr = np.asarray(seq)
        n = 0

        if mode == "leave":
            # 从序列开头开始数连续的“在目标状态”的步数，直到第一次不在目标
            for x in arr:
                if is_target(x):
                    n += 1
                else:
                    break

        elif mode == "arrive":
            # 从序列开头开始数连续的“非目标状态”的步数，直到第一次进入目标
            for x in arr:
                if not is_target(x):
                    n += 1
                else:
                    break

        counts.append(n)

    return counts


def plot_sleep_length_gmm(
    sleep_length, xwid, ywid, title, xtitle, ytitle, save_path=None, show=True,
    hist_bin_width=None,      # 新增：分箱间距（如 1.0）
    hist_density=True,        # 新增：True=密度；False=个数
    bin_anchor=0.0            # 新增：锚点；None 表示以最小值对齐
):
    """
    根据 sleep_length（1D 序列）：
      1) 拟合 1 成分 GMM
      2) 绘制红色直方图（密度或个数）、绿色 PDF、蓝色 μ±2σ 阴影带
      3) 在底部用更大的散点展示样本位置

    参数
    ----
    sleep_length : array-like
        一维睡眠时长序列（分钟/小时均可）
    hist_bin_width : float or None, default None
        直方图分箱间距。None 表示使用 bins='auto'。
        例如设为 1.0 时，将按 …,-1–0, 0–1, 1–2,… 分组统计。
    hist_density : bool, default True
        True：直方图按密度绘制；False：按“个数”绘制。
        当为 False 时，绿色 PDF 会自动乘以 N*Δx 与柱高可比。
    bin_anchor : float or None, default 0.0
        分箱锚点，决定箱边界起点的对齐方式。
        - 0.0：在 0 上对齐（常用于“0–1, 1–2, …”）
        - None：以数据最小值对齐

    返回
    ----
    fig, ax, stats : (matplotlib.figure.Figure, matplotlib.axes.Axes, dict)
        stats = {"mu": float, "std": float, "ci_low": float, "ci_high": float, "gmm": GaussianMixture}
    """
    # -------- 数据准备与校验 --------
    arr = np.asarray(sleep_length, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("sleep_length 为空：请提供至少一个数值。")
    samples = arr.reshape(-1, 1)
    N = len(samples)

    # -------- GMM 拟合（1 个高斯）--------
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=0)
    gmm.fit(samples)
    mu = float(gmm.means_[0][0])
    std = float(np.sqrt(gmm.covariances_[0][0][0]))

    # -------- x 轴范围与 PDF --------
    span = samples.max() - samples.min()
    pad = 0.05 * (span if span > 0 else 1.0)
    x_min = samples.min() - pad
    x_max = samples.max() + pad
    x_vals = np.linspace(x_min, x_max, 400)
    pdf_vals = norm.pdf(x_vals, loc=mu, scale=std)

    # -------- 配色 --------
    COLORS = {
        "hist": "#e41a1c",     # 红（直方图）
        "pdf_line": "#1b9e77", # 绿（PDF线）
        "pdf_fill": "#66c2a5", # 浅绿（PDF填充）
        "ci_band": "#377eb8",  # 蓝（μ±2σ 区间 & 均值线）
        "points": "#333333"    # 深灰（数据点）
    }

    # -------- 画图风格 --------
    # plt.rcParams.update({
    #     "axes.labelsize": 12,
    #     "axes.titlesize": 13,
    #     "xtick.labelsize": 11,
    #     "ytick.labelsize": 11,
    #     "legend.fontsize": 11
    # })

    fig, ax = plt.subplots(figsize=(xwid, ywid))

    # -------- 构造分箱边界 --------
    if hist_bin_width is not None:
        if hist_bin_width <= 0:
            raise ValueError("hist_bin_width 必须为正数。")
        if bin_anchor is None:
            # 以最小值为锚
            anchor = float(samples.min())
        else:
            anchor = float(bin_anchor)

        # 左右边界向下/向上取整到间距的整数倍
        left = anchor + hist_bin_width * np.floor((samples.min() - anchor) / hist_bin_width)
        right = anchor + hist_bin_width * np.ceil((samples.max() - anchor) / hist_bin_width)
        # 为避免浮点误差，右侧多加一个极小值
        bins = np.arange(left, right + hist_bin_width * 1.0000001, hist_bin_width)
    else:
        bins = 'auto'

    # -------- 直方图（密度或个数） --------
    h = ax.hist(
        samples.ravel(),
        bins=bins,
        density=hist_density,
        alpha=0.30,
        edgecolor='none',
        color=COLORS["hist"]
    )

    # -------- 高斯 PDF（必要时缩放到“个数”量纲） --------
    if hist_density:
        pdf_plot = pdf_vals
        pdf_label = f'Gaussian PDF  (μ={mu:.2f}, σ={std:.2f})'
    else:
        # 直方图是“个数”，需将 PDF 乘以 N*Δx
        if isinstance(bins, str):
            # auto 时近似用平均箱宽
            # （更严谨可从 h[1] 读取 bin edges，计算 mean width）
            edges = h[1]
        else:
            edges = bins
        # 平均箱宽
        dx = np.mean(np.diff(edges))
        pdf_plot = pdf_vals * (N * dx)
        pdf_label = f'Gaussian PDF × N·Δx  (μ={mu:.2f}, σ={std:.2f})'

    ax.fill_between(x_vals, pdf_plot, alpha=0.22, color=COLORS["pdf_fill"])
    # ax.plot(x_vals, pdf_plot, linewidth=2.2, color=COLORS["pdf_line"], label=pdf_label)

    # -------- μ±2σ 阴影带 与 μ 线 --------
    ci_low, ci_high = mu - 2*std, mu + 2*std
    # ax.axvspan(ci_low, ci_high, alpha=0.16, color=COLORS["ci_band"], label='≈95% interval')
    ax.axvline(mu, linestyle='--', linewidth=1.8, color=COLORS["ci_band"], label='Mean μ')

    # -------- 底部散点（展示样本位置） --------
    y0, y1 = ax.get_ylim()
    y_pts = y0 + 0.03 * (y1 - y0)  # 距离底部 3% 的高度
    ax.scatter(samples.ravel(), np.full_like(samples.ravel(), y_pts), s=50,
               color=COLORS["points"], edgecolors="white", linewidths=0.5, alpha=0.95,
               label="Samples")

    # -------- 细节美化 --------
    ax.set_title(title, pad=10)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    sf = ScalarFormatter(useMathText=False)
    sf.set_scientific(False)
    ax.xaxis.set_major_formatter(sf)

    ax.legend(loc='upper left', frameon=False)
    ax.grid(True, linewidth=0.5, alpha=0.35)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format='pdf', dpi=300)
    if show:
        plt.show()

    stats = {
        "mu": mu,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "gmm": gmm
    }
    return fig, ax, stats

# def plot_hist_discrete(sleep_length, xwid, ywid, title, xtitle, ytitle, save_path=None, show=True):
#     """
#     仅对输入数据绘制直方图（离散整数 x 轴，y 为计数）。
    
#     参数
#     ----
#     sleep_length : array-like
#         一维序列（将按整数处理并在整数刻度上分箱）
#     xwid, ywid : float
#         图像宽、高（英寸）
#     title : str
#         图标题
#     xtitle : str
#         x 轴标题（如 "Nap Duration (min)"）
#     ytitle : str
#         y 轴标题（如 "Count" 或 "Count Number"）
#     show : bool, default True
#         是否在函数内部调用 plt.show()
    
#     返回
#     ----
#     fig, ax, stats : (Figure, Axes, dict)
#         stats = {
#             "counts": ndarray,       # 每个整数桶的计数
#             "bin_edges": ndarray,    # 边界：i 到 i+1（以 i±0.5 定义）
#             "bin_centers": ndarray   # 整数中心
#         }
#     """
#     # -------- 数据准备与校验 --------
#     arr = np.asarray(sleep_length, dtype=float).ravel()
#     if arr.size == 0:
#         raise ValueError("sleep_length 为空：请提供至少一个数值。")

#     # 计算整数范围，并以 i±0.5 作为边界，使柱心对齐到整数刻度
#     x_min_int = int(np.floor(np.nanmin(arr)))
#     x_max_int = int(np.ceil(np.nanmax(arr)))
#     # 确保至少有一个桶
#     if x_max_int < x_min_int:
#         x_max_int = x_min_int
#     # 构造边界：从 (min-0.5) 到 (max+0.5)，步长 1
#     bin_edges = np.arange(x_min_int - 0.5, x_max_int + 1.5, 1.0)
#     bin_centers = np.arange(x_min_int, x_max_int + 1, 1)

#     # -------- 画图 --------
#     plt.rcParams.update({
#         "axes.labelsize": 12,
#         "axes.titlesize": 13,
#         "xtick.labelsize": 11,
#         "ytick.labelsize": 11,
#         "legend.fontsize": 11
#     })

#     fig, ax = plt.subplots(figsize=(xwid, ywid))

#     # 频数直方图（计数）
#     counts, edges, patches = ax.hist(
#         arr,
#         bins=bin_edges,
#         density=False,          # 计数而非密度
#         color="#e41a1c",        # 红色（可改）
#         alpha=0.75,
#         edgecolor="white",
#         linewidth=0.8
#     )

#     # x 轴刻度对齐到整数
#     ax.set_xticks(bin_centers)

#     # 细节美化
#     ax.set_title(title, pad=10)
#     ax.set_xlabel(xtitle)
#     ax.set_ylabel(ytitle)

#     # 去除上/右边框
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     # y 轴为整型步进（更易读）
#     ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

#     # 网格仅 y 方向，弱化
#     ax.grid(axis='y', linewidth=0.5, alpha=0.35)

#     fig.tight_layout()
#     if save_path is not None:
#         fig.savefig(save_path, format='pdf', dpi=300)
#     if show:
#         plt.show()

#     stats = {
#         "counts": counts,
#         "bin_edges": edges,
#         "bin_centers": bin_centers
#     }
#     return fig, ax, stats

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MaxNLocator

def plot_hist_discrete(sleep_length, xwid, ywid, title, xtitle, ytitle, save_path=None, show=True):
    """
    支持两种模式：
    1) 数值直方图（输入可转为 float）：在整数刻度上分箱，绘制频数直方图
    2) 类别柱状图（输入为非数值的可哈希对象，如元组环）：按类别计数，绘制柱状图
    """
    # -------- 公共：数据准备与简单校验 --------
    arr_obj = np.asarray(sleep_length, dtype=object).ravel()
    if arr_obj.size == 0:
        raise ValueError("sleep_length 为空：请提供至少一个元素。")

    # 判断是否可当作数值处理（全部元素都能被 float() 成功转换）
    def _is_number(x):
        try:
            float(x)  # np.nan, np.inf 也允许
            return True
        except Exception:
            return False

    all_numeric = all(_is_number(x) for x in arr_obj)

    # 状态标签映射
    stage_labels = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    fig, ax = plt.subplots(figsize=(xwid, ywid))

    if all_numeric:
        # ================= 数值直方图模式 =================
        arr = np.asarray(arr_obj, dtype=float)

        # 计算整数范围，并以 i±0.5 作为边界，使柱心对齐到整数刻度
        x_min_int = int(np.floor(np.nanmin(arr)))
        x_max_int = int(np.ceil(np.nanmax(arr)))
        if x_max_int < x_min_int:
            x_max_int = x_min_int
        bin_edges = np.arange(x_min_int - 0.5, x_max_int + 1.5, 1.0)
        bin_centers = np.arange(x_min_int, x_max_int + 1, 1)

        # 频数直方图（计数）
        counts, edges, patches = ax.hist(
            arr,
            bins=bin_edges,
            density=False,
            color="#e41a1c",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.8
        )

        # x 轴刻度对齐到整数
        ax.set_xticks(bin_centers)

        # 如果横坐标是 0–4 的整数，改成 Wake/N1/N2/N3/REM
        if set(bin_centers).issubset(set(stage_labels.keys())):
            labels = [stage_labels[int(x)] for x in bin_centers]
            ax.set_xticklabels(labels)

        stats = {
            "counts": counts,
            "bin_edges": edges,
            "bin_centers": bin_centers
        }

    else:
        # ================= 类别柱状图模式 =================
        cnt = Counter(arr_obj)

        # 将类别对象转成可读标签
        def _format_label(cat):
            # 如果是 0–4 的整数，用睡眠阶段标签
            if (isinstance(cat, (int, np.integer, float)) 
                and int(cat) in stage_labels 
                and float(cat).is_integer()):
                return stage_labels[int(cat)]
            # 元组环：(0,1,2,0) → "0→1→2→0"
            if isinstance(cat, tuple):
                try:
                    return "→".join(map(str, cat))
                except Exception:
                    return str(cat)
            # 其他类型用 str
            return str(cat)

        # 排序：先按频数降序，再按标签字典序升序
        items = list(cnt.items())
        items.sort(key=lambda kv: (-kv[1], _format_label(kv[0])))

        categories = [k for k, v in items]
        counts = np.array([v for k, v in items], dtype=int)
        labels = [_format_label(k) for k in categories]
        xs = np.arange(len(categories))

        ax.bar(xs, counts, color="#e41a1c", alpha=0.75, edgecolor="white", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha='right')

        stats = {
            "counts": counts,
            "categories": categories,
            "labels": labels
        }

    # -------- 细节美化（两种模式通用） --------
    ax.set_title(title, pad=10)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.grid(axis='y', linewidth=0.5, alpha=0.35)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()

    return fig, ax, stats
