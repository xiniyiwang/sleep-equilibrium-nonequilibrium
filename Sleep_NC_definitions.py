"""Calculation and plotting definitions for AllFiguresNapNoc_NC_run.ipynb.

Keep this file beside the notebook in the server working directory.
No data loading, simulations, plotting or file exports run on import.
Private helper factories initialize only function definitions and constants.
The original Code module is supplied explicitly by the notebook setup.
"""

# ======================================================================
# Definitions from cleaned notebook cell 3 (zero-based)
# ======================================================================

def _build_loading_helpers():
    # Original cell 3, line 629.
    import pandas as pd

    # Original cell 3, line 630.
    import datetime

    # Original cell 3, line 707.
    from pathlib import Path

    # Original cell 3, line 1695.
    import numpy as np

    # Original cell 3, line 7.
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

    # Original cell 3, line 631.
    def _parse_lights_off_to_datetime(lights_off_val, start_dt):
        """
        把 lights_off_val（可能是 time / datetime / Timestamp / 字符串）转成一个 datetime。
        若 lights_off_val 只有“时分秒”，就以 start_dt 的日期为基准；如果早于 start_dt，则默认是“跨天”加 1 天。
        """
        if start_dt is None or pd.isna(lights_off_val):
            return None

        # 统一成 python datetime/time
        if isinstance(lights_off_val, pd.Timestamp):
            lights_off_val = lights_off_val.to_pydatetime()

        if isinstance(lights_off_val, datetime.datetime):
            target_dt = lights_off_val
            # 如果 target_dt 没 tz 而 start_dt 有 tz，就把 tz 对齐（不做时区换算，仅附着 tzinfo）
            if (target_dt.tzinfo is None) and (getattr(start_dt, "tzinfo", None) is not None):
                target_dt = target_dt.replace(tzinfo=start_dt.tzinfo)
            return target_dt

        if isinstance(lights_off_val, datetime.time):
            t = lights_off_val
        else:
            # 字符串 / 数值等：尽量用 pandas 解析成时间
            t_parsed = pd.to_datetime(str(lights_off_val), errors="coerce")
            if pd.isna(t_parsed):
                return None
            t = t_parsed.to_pydatetime().time()

        target_dt = start_dt.replace(hour=t.hour, minute=t.minute, second=t.second, microsecond=0)
        if target_dt < start_dt:
            target_dt = target_dt + datetime.timedelta(days=1)
        return target_dt

    # Original cell 3, line 664.
    def lights_off_epoch_idx_by_tstart_end(
        df_i,
        start_dt,
        lights_off_val,
        t_start_col="t_start_s",
        t_end_col="t_end_s",
        epoch_idx_col="epoch_idx",
    ):
        """
        用 df_i 的 [t_start_s, t_end_s) 直接定位 lights_off 对应的 epoch_idx。
        规则：找到满足 t_start_s <= delta_s < t_end_s 的那一行；用 searchsorted 实现。
        若超出范围：夹到 [0, last]。
        """
        if start_dt is None or pd.isna(lights_off_val) or df_i is None or len(df_i) == 0:
            return None

        target_dt = _parse_lights_off_to_datetime(lights_off_val, start_dt)
        if target_dt is None:
            return None

        delta_s = (target_dt - start_dt).total_seconds()

        # 确保按时间升序（一般本来就升序）
        df = df_i.sort_values(t_end_col, kind="mergesort").reset_index(drop=True)

        t_end = df[t_end_col].to_numpy(dtype=float)
        # 找到第一个 t_end > delta_s 的位置
        pos = int(np.searchsorted(t_end, delta_s, side="right"))

        # 夹到有效范围
        pos = max(0, min(pos, len(df) - 1))

        # 可选：严谨校验一下是否落在窗内；不在就按边界处理
        t0 = float(df.loc[pos, t_start_col])
        t1 = float(df.loc[pos, t_end_col])
        if not (t0 <= delta_s < t1):
            if delta_s < float(df.loc[0, t_start_col]):
                pos = 0
            elif delta_s >= float(df.loc[len(df)-1, t_end_col]):
                pos = len(df) - 1
            # 中间极少数异常（比如 t_start/t_end 不连续）就保持 pos

        return int(df.loc[pos, epoch_idx_col])

    # Original cell 3, line 710.
    def load_psg_y_only(
        psg_dir,
        target_name,
        stage_seg,            # (n_seg,) 每段一个label（int/str都行）
        epoch_sec,            # (n_seg,) 每段持续时间（秒）
        base_sec=30.0,        # 固定epoch长度（秒）
        include_tail=False,   # True: 不足base_sec的尾巴也算一个epoch（一般不建议）
    ):
        """
        只按方案B把 (stage_seg, epoch_sec) 展开成固定长度 base_sec 的 epoch 标签 y。

        Returns
        -------
        y : np.ndarray, shape (n_epochs,)
            每个固定epoch的标签（与传入stage_seg同类型）
        """

        def _explode_to_fixed_epochs(stage, dur, base_sec=30.0, include_tail=False):
            stage = np.asarray(stage)
            dur = np.asarray(dur, dtype=float)

            stage_fixed = []
            for s, d in zip(stage, dur):
                n_full = int(np.floor(d / base_sec))
                stage_fixed.extend([s] * n_full)

                if include_tail:
                    rem = d - n_full * base_sec
                    if rem > 1e-6:
                        stage_fixed.append(s)

            return np.asarray(stage_fixed)

        # 保留参数接口一致性：确认文件存在（可选）
        psg_dir = Path(psg_dir)
        matches = list(psg_dir.rglob(target_name))
        if len(matches) == 0:
            raise FileNotFoundError(f"Not found: {target_name} under {psg_dir}")

        # 只算 y
        y = _explode_to_fixed_epochs(stage_seg, epoch_sec, base_sec=base_sec, include_tail=include_tail)
        return y

    return {'first_passage_counts_array': first_passage_counts_array, 'lights_off_epoch_idx_by_tstart_end': lights_off_epoch_idx_by_tstart_end, 'load_psg_y_only': load_psg_y_only}

_loading_helpers = _build_loading_helpers()
first_passage_counts_array = _loading_helpers['first_passage_counts_array']
lights_off_epoch_idx_by_tstart_end = _loading_helpers['lights_off_epoch_idx_by_tstart_end']
load_psg_y_only = _loading_helpers['load_psg_y_only']


# ======================================================================
# Definitions from cleaned notebook cell 5 (zero-based)
# ======================================================================

def parse_sc_filename(filename):
    """
    SC4001EC-Hypnogram.edf -> (0, 1)
    SC4012EC-Hypnogram.edf -> (1, 2)
    """
    match = re.match(
        r"^SC4(\d{2})([12])",
        Path(filename).name,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(f"无法解析文件名：{filename}")

    subject = int(match.group(1))
    night = int(match.group(2))

    return subject, night


# ======================================================================
# Definitions from cleaned notebook cell 17 (zero-based)
# ======================================================================

def _build_temporal_helpers():
    # Original cell 20, line 1.
    import numpy as np

    # Original cell 20, line 850.
    from typing import Dict, List, Sequence, Any, Tuple, Optional

    # Original cell 20, line 851.
    from collections import Counter

    # Original cell 20, line 852.
    import pandas as pd

    # Original cell 20, line 23.
    def compress_hypnogram_with_indices(hypnogram: Sequence[Any]):
        """
        去掉连续重复状态，同时返回压缩后每个状态
        在原始 hypnogram 中第一次出现的位置。

        例如:
        hyp = [0,0,1,1,2,2,0,0]
        返回:
        seq = [0,1,2,0]
        idx = [0,2,4,6]
        """
        hypnogram = list(hypnogram)
        if len(hypnogram) == 0:
            return [], []

        seq = [hypnogram[0]]
        idx = [0]

        for k, x in enumerate(hypnogram[1:], start=1):
            if x != seq[-1]:
                seq.append(x)
                idx.append(k)

        return seq, idx

    # Original cell 20, line 91.
    def find_all_loops_one_subject(
        hypnogram: Sequence[Any],
        compress_consecutive: bool = True,
        min_distinct_states: int = 3,
        max_distinct_states: Optional[int] = 5,
        epoch_sec: Optional[float] = None,
    ):
        """
        从单个 subject 的 hypnogram 中找出所有真实出现过的 simple loop occurrence。

        返回中新增：
        - 每个 occurrence 的 duration_steps
        - 每个 subject 的 mean_cycle_duration
        - 若给了 epoch_sec，则额外返回秒单位 duration
        """
        hypnogram = list(hypnogram)

        if compress_consecutive:
            seq, seq_start_idx = compress_hypnogram_with_indices(hypnogram)
        else:
            seq = hypnogram
            seq_start_idx = list(range(len(hypnogram)))

        n = len(seq)

        occurrences = []
        family_counts = Counter()
        duration_list = []

        for i in range(n):
            start_state = seq[i]

            for j in range(i + 1, n):
                if seq[j] != start_state:
                    continue

                path = seq[i:j + 1]

                if not is_simple_cycle(path, min_distinct_states=min_distinct_states):
                    continue

                n_distinct = len(set(path[:-1]))
                if max_distinct_states is not None and n_distinct > max_distinct_states:
                    continue

                cycle_raw = tuple(path)
                cycle_family = canonicalize_cycle_directional(path)

                # duration 定义为：从第一次进入起点，到再次回到起点，
                # 在原始 hypnogram 上跨过的步数
                duration_steps = seq_start_idx[j] - seq_start_idx[i]

                occ = {
                    "start": i,
                    "end": j,
                    "cycle_raw": cycle_raw,
                    "cycle_family": cycle_family,
                    "n_distinct": n_distinct,
                    "length": len(path),                 # 压缩序列中的长度（保留原字段）
                    "duration_steps": duration_steps,   # 原始 hypnogram 中的 duration
                }

                if epoch_sec is not None:
                    occ["duration_sec"] = duration_steps * epoch_sec

                occurrences.append(occ)
                family_counts[cycle_family] += 1
                duration_list.append(duration_steps)

        mean_cycle_duration = float(np.mean(duration_list)) if len(duration_list) > 0 else np.nan

        out = {
            "hypnogram_used": seq,
            "occurrences": occurrences,
            "family_counts": dict(family_counts),
            "total_loop_occurrences": len(occurrences),
            "n_unique_loop_families": len(family_counts),
            "mean_cycle_duration": mean_cycle_duration,
        }

        if epoch_sec is not None:
            out["mean_cycle_duration_sec"] = mean_cycle_duration * epoch_sec if np.isfinite(mean_cycle_duration) else np.nan

        return out

    # Original cell 20, line 177.
    def count_total_loops_all_subjects(
        data: Dict[Any, Sequence[Any]],
        compress_consecutive: bool = True,
        min_distinct_states: int = 3,
        max_distinct_states: Optional[int] = 5,
        epoch_sec: Optional[float] = None,
    ):
        """
        统计每个 subject 的 hypnogram 总共经历了多少个 loop。

        返回：
        - results: 每个 subject 的详细结果
        - df_summary: 每个 subject 的汇总表

        df_summary 新增一列：
        - mean_cycle_duration

        若给 epoch_sec，还新增：
        - mean_cycle_duration_sec
        """
        results = {}
        summary_rows = []

        for subj, hyp in data.items():
            res = find_all_loops_one_subject(
                hypnogram=hyp,
                compress_consecutive=compress_consecutive,
                min_distinct_states=min_distinct_states,
                max_distinct_states=max_distinct_states,
                epoch_sec=epoch_sec,
            )
            results[subj] = res

            row = {
                "subject": subj,
                "total_loop_occurrences": res["total_loop_occurrences"],
                "n_unique_loop_families": res["n_unique_loop_families"],
                "compressed_length": len(res["hypnogram_used"]),
                "mean_cycle_duration": res["mean_cycle_duration"],
            }

            if epoch_sec is not None:
                row["mean_cycle_duration_sec"] = res.get("mean_cycle_duration_sec", np.nan)

            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows).sort_values("subject").reset_index(drop=True)
        return results, df_summary

    # Original cell 20, line 855.
    def compress_hypnogram_with_runs(hypnogram: Sequence[Any]) -> Dict[str, List[Any]]:
        """
        压缩连续重复状态，同时保留每个压缩段在原始序列中的起止信息。

        返回
        ----
        {
            "seq": [0,1,2,0,...],              # 压缩后的状态序列
            "run_starts": [0, 5, 9, 13, ...], # 每段在原始 hypnogram 中的起始 index
            "run_ends":   [5, 9, 13, 20, ...],# 每段在原始 hypnogram 中的结束 index（右开区间）
            "run_lengths":[5, 4, 4, 7, ...],  # 每段持续多少个 epoch / sample
        }
        """
        hypnogram = list(hypnogram)
        if len(hypnogram) == 0:
            return {
                "seq": [],
                "run_starts": [],
                "run_ends": [],
                "run_lengths": [],
            }

        seq = []
        run_starts = []
        run_ends = []
        run_lengths = []

        start = 0
        current = hypnogram[0]

        for idx in range(1, len(hypnogram)):
            if hypnogram[idx] != current:
                seq.append(current)
                run_starts.append(start)
                run_ends.append(idx)              # 右开区间
                run_lengths.append(idx - start)

                current = hypnogram[idx]
                start = idx

        # 最后一段
        seq.append(current)
        run_starts.append(start)
        run_ends.append(len(hypnogram))
        run_lengths.append(len(hypnogram) - start)

        return {
            "seq": seq,
            "run_starts": run_starts,
            "run_ends": run_ends,
            "run_lengths": run_lengths,
        }

    # Original cell 20, line 916.
    def is_simple_cycle(path: Sequence[Any], min_distinct_states: int = 3) -> bool:
        """
        判断 path 是否是 simple cycle。

        要求：
        1. 首尾相同
        2. 至少包含 min_distinct_states 个不同状态（默认>=3）
        3. 除首尾外，中间状态不能重复
        """
        path = list(path)
        if len(path) < min_distinct_states + 1:
            return False
        if path[0] != path[-1]:
            return False

        core = path[:-1]
        if len(set(core)) < min_distinct_states:
            return False

        if len(core) != len(set(core)):
            return False

        return True

    # Original cell 20, line 941.
    def canonicalize_cycle_directional(cycle: Sequence[Any]) -> Tuple[Any, ...]:
        """
        对闭合 cycle 做旋转归一化，但保留方向。
        """
        cycle = list(cycle)
        if len(cycle) < 2 or cycle[0] != cycle[-1]:
            raise ValueError(f"Not a closed cycle: {cycle}")

        core = cycle[:-1]
        rotations = [tuple(core[k:] + core[:k]) for k in range(len(core))]
        best = min(rotations)
        return best + (best[0],)

    # Original cell 20, line 955.
    def canonicalize_cycle_undirected(cycle: Sequence[Any]) -> Tuple[Any, ...]:
        """
        对闭合 cycle 做旋转归一化 + 正反方向合并。
        """
        cycle = list(cycle)
        if len(cycle) < 2 or cycle[0] != cycle[-1]:
            raise ValueError(f"Not a closed cycle: {cycle}")

        core = cycle[:-1]
        fwd = canonicalize_cycle_directional(core + [core[0]])
        rev_core = list(reversed(core))
        rev = canonicalize_cycle_directional(rev_core + [rev_core[0]])
        return min(fwd, rev)

    # Original cell 20, line 970.
    def find_all_temporal_cycles_one_subject(
        hypnogram: Sequence[Any],
        compress_consecutive: bool = True,
        min_distinct_states: int = 3,
        max_distinct_states: Optional[int] = None,
        merge_reverse: bool = False,
        epoch_sec: float = 30.0,
    ) -> Dict[str, Any]:
        """
        在单个 subject 的 hypnogram 中找出所有真实出现过的时序循环。

        duration 的定义：
        - duration_epochs: 从 cycle 起点，到“再次回到起始状态”这一时刻的时长
          也就是不包含最后那个重复起始状态 run 的停留时间。
        - duration_including_terminal_epochs: 包含最后那个重复起始状态 run 的完整覆盖时长

        如果你要算一个 loop 完成一圈所需时间，通常建议用 duration_epochs。
        """
        if compress_consecutive:
            comp = compress_hypnogram_with_runs(hypnogram)
            seq = comp["seq"]
            run_starts = comp["run_starts"]
            run_ends = comp["run_ends"]
            run_lengths = comp["run_lengths"]
        else:
            seq = list(hypnogram)
            run_starts = list(range(len(seq)))
            run_ends = [i + 1 for i in range(len(seq))]
            run_lengths = [1] * len(seq)

        n = len(seq)

        occurrences = []
        family_counts = Counter()

        for i in range(n):
            start_state = seq[i]

            for j in range(i + 1, n):
                if seq[j] != start_state:
                    continue

                path = seq[i:j + 1]

                if not is_simple_cycle(path, min_distinct_states=min_distinct_states):
                    continue

                n_distinct = len(set(path[:-1]))
                if max_distinct_states is not None and n_distinct > max_distinct_states:
                    continue

                cycle_raw = tuple(path)
                if merge_reverse:
                    cycle_family = canonicalize_cycle_undirected(path)
                else:
                    cycle_family = canonicalize_cycle_directional(path)

                start_epoch = run_starts[i]
                return_epoch = run_starts[j]              # 回到起始状态的时刻
                terminal_end_epoch = run_ends[j]          # 最后一个重复起始状态 run 结束时刻

                duration_epochs = return_epoch - start_epoch
                duration_including_terminal_epochs = terminal_end_epoch - start_epoch

                occ = {
                    "start": i,   # 压缩后序列中的起点
                    "end": j,     # 压缩后序列中的终点

                    "cycle_raw": cycle_raw,
                    "cycle_family": cycle_family,
                    "n_distinct": n_distinct,
                    "length": len(path),  # 仍保留旧字段，表示压缩后节点数

                    # 原始序列中的位置
                    "start_epoch": start_epoch,
                    "return_epoch": return_epoch,
                    "terminal_end_epoch": terminal_end_epoch,
                    "terminal_run_epochs": run_lengths[j],

                    # 推荐使用：完成一圈回到起点的时间
                    "duration_epochs": duration_epochs,
                    "duration_sec": duration_epochs * epoch_sec,
                    "duration_min": duration_epochs * epoch_sec / 60.0,

                    # 可选：如果你想把最后那个重复起始状态的停留也算进去
                    "duration_including_terminal_epochs": duration_including_terminal_epochs,
                    "duration_including_terminal_sec": duration_including_terminal_epochs * epoch_sec,
                    "duration_including_terminal_min": duration_including_terminal_epochs * epoch_sec / 60.0,
                }

                occurrences.append(occ)
                family_counts[cycle_family] += 1

        return {
            "hypnogram_used": seq,
            "occurrences": occurrences,
            "family_counts": dict(family_counts),
        }

    # Original cell 20, line 1070.
    def find_all_temporal_cycles_all_subjects(
        data: Dict[Any, Sequence[Any]],
        compress_consecutive: bool = True,
        min_distinct_states: int = 3,
        max_distinct_states: Optional[int] = 5,
        merge_reverse: bool = False,
        epoch_sec: float = 30.0,
    ) -> Dict[Any, Dict[str, Any]]:
        """
        对 data 中所有 subject 找出真实出现过的时序循环。
        """
        out = {}
        for subj, hyp in data.items():
            out[subj] = find_all_temporal_cycles_one_subject(
                hypnogram=hyp,
                compress_consecutive=compress_consecutive,
                min_distinct_states=min_distinct_states,
                max_distinct_states=max_distinct_states,
                merge_reverse=merge_reverse,
                epoch_sec=epoch_sec,
            )
        return out

    # Original cell 20, line 1094.
    def cycles_to_dataframe(results_by_subject: Dict[Any, Dict[str, Any]]) -> pd.DataFrame:
        """
        把每个 subject 的 cycle family 的计数 + 平均时长整理成 DataFrame。
        """
        rows = []

        for subj, res in results_by_subject.items():
            occs = res["occurrences"]
            if len(occs) == 0:
                continue

            df_occ = pd.DataFrame(occs)
            g = (
                df_occ.groupby("cycle_family", dropna=False)
                .agg(
                    count=("cycle_family", "size"),
                    n_distinct=("n_distinct", "first"),
                    length=("length", "first"),
                    mean_duration_epochs=("duration_epochs", "mean"),
                    mean_duration_min=("duration_min", "mean"),
                    mean_duration_including_terminal_epochs=("duration_including_terminal_epochs", "mean"),
                    mean_duration_including_terminal_min=("duration_including_terminal_min", "mean"),
                )
                .reset_index()
            )

            g.insert(0, "subject", subj)
            rows.append(g)

        if len(rows) == 0:
            return pd.DataFrame()

        df = pd.concat(rows, axis=0, ignore_index=True)
        df = df.sort_values(["subject", "count"], ascending=[True, False]).reset_index(drop=True)
        return df

    # Original cell 20, line 1131.
    def occurrences_to_dataframe(results_by_subject: Dict[Any, Dict[str, Any]]) -> pd.DataFrame:
        """
        把每次 cycle occurrence 展开成 DataFrame。
        """
        rows = []
        for subj, res in results_by_subject.items():
            for occ in res["occurrences"]:
                rows.append({
                    "subject": subj,
                    "start": occ["start"],
                    "end": occ["end"],
                    "cycle_raw": occ["cycle_raw"],
                    "cycle_family": occ["cycle_family"],
                    "n_distinct": occ["n_distinct"],
                    "length": occ["length"],

                    "start_epoch": occ["start_epoch"],
                    "return_epoch": occ["return_epoch"],
                    "terminal_end_epoch": occ["terminal_end_epoch"],
                    "terminal_run_epochs": occ["terminal_run_epochs"],

                    "duration_epochs": occ["duration_epochs"],
                    "duration_sec": occ["duration_sec"],
                    "duration_min": occ["duration_min"],

                    "duration_including_terminal_epochs": occ["duration_including_terminal_epochs"],
                    "duration_including_terminal_sec": occ["duration_including_terminal_sec"],
                    "duration_including_terminal_min": occ["duration_including_terminal_min"],
                })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values(["subject", "start", "end"]).reset_index(drop=True)
        return df

    return {'find_all_temporal_cycles_all_subjects': find_all_temporal_cycles_all_subjects, 'occurrences_to_dataframe': occurrences_to_dataframe, 'cycles_to_dataframe': cycles_to_dataframe, 'count_total_loops_all_subjects': count_total_loops_all_subjects}

_temporal_helpers = _build_temporal_helpers()
find_all_temporal_cycles_all_subjects = _temporal_helpers['find_all_temporal_cycles_all_subjects']
occurrences_to_dataframe = _temporal_helpers['occurrences_to_dataframe']
cycles_to_dataframe = _temporal_helpers['cycles_to_dataframe']
count_total_loops_all_subjects = _temporal_helpers['count_total_loops_all_subjects']


# ======================================================================
# Definitions from cleaned notebook cell 22 (zero-based)
# ======================================================================

def _build_flow_helpers():
    # Original cell 127, line 5.
    import numpy as np

    # Original cell 127, line 408.
    def decompose_flux_lists(Pss, Tmat, mode='loop', zero_diag=False):
        """
        Pss: list of (5,)
        Tmat: list of (5,5), row->col

        mode:
            'loop'  -> F_sym = min(F, F.T), F_irr = F - F_sym
            'avg'   -> F_sym = (F + F.T)/2, F_asym = (F - F.T)/2
        """
        Flux_list = []
        Sym_list = []
        Other_list = []

        for p, T in zip(Pss, Tmat):
            p = np.asarray(p, dtype=float).reshape(-1)
            T = np.asarray(T, dtype=float)

            F = p[:, None] * T

            if mode == 'loop':
                Sym = np.minimum(F, F.T)
                Other = F - Sym
            elif mode == 'avg':
                Sym = 0.5 * (F + F.T)
                Other = 0.5 * (F - F.T)
            else:
                raise ValueError("mode must be 'loop' or 'avg'")

            if zero_diag:
                np.fill_diagonal(Sym, 0.0)
                np.fill_diagonal(Other, 0.0)

            Flux_list.append(F)
            Sym_list.append(Sym)
            Other_list.append(Other)

        return Flux_list, Sym_list, Other_list

    # Original cell 127, line 445.
    def is_nrem_rem_loop(loop, nrem_states={1, 2, 3}, rem_state=4):
        """
        判断一个 loop 是否属于 NREM-REM loop:
        只要同时包含 REM(4) 和任意 NREM(1/2/3) 即可
        """
        s = set(loop)
        return (rem_state in s) and (len(s & set(nrem_states)) > 0)

    # Original cell 127, line 454.
    def sum_nrem_rem_flux(AllCycle, AllStrength, nrem_states={1, 2, 3}, rem_state=4):
        """
        返回：
        1. per_subject_sum: 每个subject中 NREM-REM loop 的总flux之和
        2. per_subject_detail: 每个subject对应的匹配loop及其flux
        """
        per_subject_sum = {}
        per_subject_detail = {}

        subjects = sorted(AllCycle.keys())

        for subj in subjects:
            loops = AllCycle[subj]
            strengths = AllStrength[subj]

            if len(loops) != len(strengths):
                raise ValueError(f"Subject {subj}: len(AllCycle[{subj}]) != len(AllStrength[{subj}])")

            matched_loops = []
            matched_strengths = []

            for loop, flux in zip(loops, strengths):
                if is_nrem_rem_loop(loop, nrem_states=nrem_states, rem_state=rem_state):
                    matched_loops.append(loop)
                    matched_strengths.append(flux)

            per_subject_sum[subj] = float(np.sum(matched_strengths)) if matched_strengths else 0.0
            per_subject_detail[subj] = {
                "loops": matched_loops,
                "strengths": matched_strengths,
            }

        return per_subject_sum, per_subject_detail

    return {'decompose_flux_lists': decompose_flux_lists, 'sum_nrem_rem_flux': sum_nrem_rem_flux}

_flow_helpers = _build_flow_helpers()
decompose_flux_lists = _flow_helpers['decompose_flux_lists']
sum_nrem_rem_flux = _flow_helpers['sum_nrem_rem_flux']


# ======================================================================
# Definitions from cleaned notebook cell 27 (zero-based)
# ======================================================================

def _build_n2n3_helpers():
    # Original cell 138, line 1.
    import numpy as np

    # Original cell 138, line 3.
    def compute_N2_N3_transition_frequency(NocHypDict):
        """
        Count N2-N3 transitions in each hypnogram and divide by hypnogram length.

        State coding:
            N2 = 2
            N3 = 3

        Returns
        -------
        freq_array : np.ndarray
            Shape = [n_hypnograms]
            N2-N3 transition frequency per epoch.
        """

        freq_list = []

        for key in NocHypDict.keys():
            hyp = np.asarray(NocHypDict[key]).astype(int)

            # transitions between consecutive epochs
            pre = hyp[:-1]
            post = hyp[1:]

            # N2 -> N3 or N3 -> N2
            n_N2_N3_transitions = np.sum(
                ((pre == 2) & (post == 3)) |
                ((pre == 3) & (post == 2))
            )

            # divide by hypnogram length
            freq = n_N2_N3_transitions / len(hyp)

            freq_list.append(freq)

        return np.asarray(freq_list)

    return {'compute_N2_N3_transition_frequency': compute_N2_N3_transition_frequency}

_n2n3_helpers = _build_n2n3_helpers()
compute_N2_N3_transition_frequency = _n2n3_helpers['compute_N2_N3_transition_frequency']


# ======================================================================
# Definitions from cleaned notebook cell 30 (zero-based)
# ======================================================================

def _early_sleep_forces(matrices, record_ids, n_steps=20):
    flux_means, symmetric_means = [], []
    for record_id in record_ids:
        T = matrices[record_id]
        P = np.array([1, 0, 0, 0, 0])
        flux_values, symmetric_values = [], []
        for _ in range(n_steps):
            J = compute_J_matrix(T, P)
            _, sym, _ = decompose_flux_lists([P], [T], mode='loop', zero_diag=True)
            flux_values.append(J[0, 1:].sum())
            symmetric_values.append(sym[0][0, 1:].sum())
            P = P @ T
        flux_means.append(np.array(flux_values).mean())
        symmetric_means.append(np.array(symmetric_values).mean())
    return np.array(flux_means), np.array(symmetric_means)


# ======================================================================
# Definitions from cleaned notebook cell 32 (zero-based)
# ======================================================================

def _build_inside_outside_helpers():
    # Original cell 67, line 2.
    import pandas as pd

    # Original cell 74, line 1.
    import numpy as np

    # Original cell 52, line 6.
    STATE_NAME_MAP = {
        0: "Wake",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM",
    }

    # Original cell 52, line 19.
    def to_1d_hypnogram(x):
        h0 = np.asarray(x)
        h = np.squeeze(h0)

        if h.ndim == 1:
            return h.astype(int)

        if h.ndim == 2 and 1 in h.shape:
            return h.reshape(-1).astype(int)

        raise ValueError(
            f"Invalid hypnogram shape: original={h0.shape}, squeezed={h.shape}"
        )

    # Original cell 52, line 34.
    def iter_hypnograms(data):
        if isinstance(data, dict):
            items = sorted(data.items(), key=lambda kv: kv[0])
            return [(i, k, v) for i, (k, v) in enumerate(items)]

        arr = np.asarray(data)

        if arr.ndim == 1:
            return [(0, 0, data)]

        if arr.ndim == 2 and arr.dtype != object:
            return [(i, i, arr[i]) for i in range(arr.shape[0])]

        return [(i, i, hyp) for i, hyp in enumerate(data)]

    # Original cell 52, line 50.
    def find_runs(x):
        x = np.asarray(x)

        if len(x) == 0:
            return (
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )

        change = np.where(np.diff(x) != 0)[0] + 1
        starts = np.r_[0, change]
        ends = np.r_[change - 1, len(x) - 1]
        states = x[starts]

        return states.astype(int), starts.astype(int), ends.astype(int)

    # Original cell 52, line 68.
    def compress_consecutive(seq):
        seq = list(seq)

        if len(seq) == 0:
            return []

        out = [int(seq[0])]

        for x in seq[1:]:
            x = int(x)
            if x != out[-1]:
                out.append(x)

        return out

    # Original cell 52, line 84.
    def structure_to_code(seq):
        if seq is None:
            return ""

        if isinstance(seq, str):
            return (
                seq.replace(" ", "")
                   .replace(",", "")
                   .replace("-", "")
                   .replace("_", "")
                   .replace("→", "")
            )

        return "".join(str(int(x)) for x in seq)

    # Original cell 52, line 100.
    def structure_to_tuple(seq):
        if seq is None:
            return tuple()

        if isinstance(seq, str):
            s = structure_to_code(seq)
            return tuple(int(ch) for ch in s if ch.isdigit())

        return tuple(int(x) for x in seq)

    # Original cell 52, line 111.
    def structure_to_name(seq, state_name_map=None, arrow="→"):
        if state_name_map is None:
            state_name_map = STATE_NAME_MAP

        return arrow.join(state_name_map.get(int(x), str(x)) for x in seq)

    # Original cell 52, line 118.
    def close_loop(loop):
        loop = structure_to_tuple(loop)

        if len(loop) == 0:
            return tuple()

        if loop[0] != loop[-1]:
            loop = loop + (loop[0],)

        return loop

    # Original cell 52, line 130.
    def canonical_rotation_code(loop):
        """
        Direction-preserving, rotation-invariant closed-loop code.

        Example:
            1241, 2412, 4124 -> 1241
        """
        loop = close_loop(loop)

        if len(loop) <= 1:
            return structure_to_code(loop)

        core = loop[:-1]

        rotations = []
        for k in range(len(core)):
            r = core[k:] + core[:k]
            rotations.append(r + (r[0],))

        return min(structure_to_code(r) for r in rotations)

    # Original cell 52, line 152.
    def canonical_rotation_loop(loop):
        loop = close_loop(loop)

        if len(loop) <= 1:
            return loop

        core = loop[:-1]

        rotations = []
        for k in range(len(core)):
            r = core[k:] + core[:k]
            rotations.append(r + (r[0],))

        return sorted(rotations, key=structure_to_code)[0]

    # Original cell 52, line 168.
    def get_allcycle_for_hypnogram(AllCycle, hyp_index, subject):
        if isinstance(AllCycle, dict):
            if subject in AllCycle:
                return AllCycle[subject]
            if hyp_index in AllCycle:
                return AllCycle[hyp_index]
            return []

        return AllCycle[hyp_index]

    # Original cell 52, line 183.
    def extract_nremrem_intervals_all_hypnograms(
        data,
        epoch_sec=30,
        wake_state=0,
        rem_state=4,
        nrem_states=(1, 2, 3),
        min_nrem_min=0,
        min_rem_min=0,
        include_post_rem_state=True,
        include_wake_in_structure=False,
    ):
        """
        Extract raw NREM-REM cycle intervals.

        For inside/outside classification:
            if inside_uses_structure_end=True later,
            inside interval = cycle_start_epoch -> structure_end_epoch.

        This lets post-REM return state close loops such as N1→N2→REM→N1.
        """

        rows = []

        min_nrem_epochs = int(np.ceil(min_nrem_min * 60 / epoch_sec))
        min_rem_epochs = int(np.ceil(min_rem_min * 60 / epoch_sec))

        for hyp_index, subject, hyp in iter_hypnograms(data):
            h = to_1d_hypnogram(hyp)

            sleep_states = set(nrem_states) | {rem_state}
            sleep_onset_candidates = np.where(np.isin(h, list(sleep_states)))[0]

            if len(sleep_onset_candidates) == 0:
                continue

            sleep_onset = int(sleep_onset_candidates[0])

            run_states, run_starts, run_ends = find_runs(h)

            rem_runs = []
            for s, st, ed in zip(run_states, run_starts, run_ends):
                if int(s) == rem_state:
                    rem_runs.append((int(st), int(ed)))

            cycle_start = sleep_onset
            cycle_index = 0

            for rem_start, rem_end in rem_runs:
                if rem_end < cycle_start:
                    continue

                pre_rem_segment = h[cycle_start:rem_start]

                nrem_count = int(np.sum(np.isin(pre_rem_segment, nrem_states)))
                rem_count = int(rem_end - rem_start + 1)

                if nrem_count >= min_nrem_epochs and rem_count >= min_rem_epochs:
                    segment_start = int(cycle_start)
                    segment_end = int(rem_end)

                    if include_post_rem_state and rem_end + 1 < len(h):
                        structure_end = int(rem_end + 1)
                    else:
                        structure_end = int(rem_end)

                    segment = h[segment_start:structure_end + 1]

                    structure_with_wake = tuple(compress_consecutive(segment))

                    if include_wake_in_structure:
                        structure = structure_with_wake
                    else:
                        segment_no_wake = [
                            x for x in segment if int(x) != wake_state
                        ]
                        structure = tuple(compress_consecutive(segment_no_wake))

                    duration_epochs = int(segment_end - segment_start + 1)
                    duration_min = duration_epochs * epoch_sec / 60.0

                    rows.append({
                        "hyp_index": int(hyp_index),
                        "subject": subject,
                        "cycle_index": int(cycle_index),

                        "cycle_start_epoch": int(segment_start),
                        "cycle_end_epoch": int(segment_end),
                        "structure_end_epoch": int(structure_end),

                        "rem_start_epoch": int(rem_start),
                        "rem_end_epoch": int(rem_end),

                        "nrem_epochs_before_rem": int(nrem_count),
                        "rem_epochs": int(rem_count),

                        "duration_epochs": int(duration_epochs),
                        "duration_min": float(duration_min),

                        "structure": structure,
                        "structure_code": structure_to_code(structure),
                        "structure_name": structure_to_name(structure),

                        "structure_with_wake": structure_with_wake,
                        "structure_with_wake_code": structure_to_code(structure_with_wake),
                        "structure_with_wake_name": structure_to_name(structure_with_wake),
                    })

                    cycle_index += 1
                    cycle_start = rem_end + 1

        return pd.DataFrame(rows)

    # Original cell 52, line 296.
    def build_interval_lookup(
        df_nremrem_intervals,
        inside_uses_structure_end=True,
    ):
        """
        hyp_index -> list of (inside_start_epoch, inside_end_epoch, cycle_index)
        """

        lookup = {}

        if len(df_nremrem_intervals) == 0:
            return lookup

        for _, row in df_nremrem_intervals.iterrows():
            hyp_index = int(row["hyp_index"])
            start = int(row["cycle_start_epoch"])

            if inside_uses_structure_end:
                end = int(row["structure_end_epoch"])
            else:
                end = int(row["cycle_end_epoch"])

            cycle_index = int(row["cycle_index"])

            lookup.setdefault(hyp_index, []).append((start, end, cycle_index))

        for hyp_index in lookup:
            lookup[hyp_index] = sorted(lookup[hyp_index], key=lambda x: x[0])

        return lookup

    # Original cell 52, line 495.
    def unique_flux_loop_families_for_hypnogram(AllCycle, hyp_index, subject):
        loops = get_allcycle_for_hypnogram(
            AllCycle=AllCycle,
            hyp_index=hyp_index,
            subject=subject,
        )

        family = {}

        for loop in loops:
            loop_closed = close_loop(loop)

            if len(loop_closed) <= 1:
                continue

            code = canonical_rotation_code(loop_closed)

            if code not in family:
                loop_canonical = canonical_rotation_loop(loop_closed)
                family[code] = {
                    "loop_family_code": code,
                    "loop": loop_canonical,
                    "loop_code": structure_to_code(loop_canonical),
                    "loop_name": structure_to_name(loop_canonical),
                    "loop_n_edges": len(loop_canonical) - 1,
                }

        return list(family.values())

    # Original cell 57, line 9.
    def classify_interval_location(
        start_epoch,
        end_epoch,
        intervals,
    ):
        """
        Original inside rule.

        inside if:
            1. [start_epoch, end_epoch] is fully contained in one NREM-REM interval
            OR
            2. it overlaps multiple NREM-REM intervals

        otherwise outside.
        """

        if intervals is None or len(intervals) == 0:
            return {
                "location": "outside",
                "containing_nremrem_cycle_index": None,
                "interval_relation": "outside",
                "overlapped_nremrem_cycle_indices": tuple(),
            }

        fully_contained = []
        overlaps = []

        for st, ed, cycle_index in intervals:
            st = int(st)
            ed = int(ed)
            cycle_index = int(cycle_index)

            is_fully_contained = (
                start_epoch >= st
                and end_epoch <= ed
            )

            has_overlap = not (
                end_epoch < st
                or start_epoch > ed
            )

            if is_fully_contained:
                fully_contained.append((st, ed, cycle_index))

            if has_overlap:
                overlaps.append((st, ed, cycle_index))

        if len(fully_contained) > 0:
            return {
                "location": "inside",
                "containing_nremrem_cycle_index": int(fully_contained[0][2]),
                "interval_relation": "fully_contained_in_one_nremrem_interval",
                "overlapped_nremrem_cycle_indices": tuple(
                    int(x[2]) for x in overlaps
                ),
            }

        if len(overlaps) >= 2:
            return {
                "location": "inside",
                "containing_nremrem_cycle_index": tuple(
                    int(x[2]) for x in overlaps
                ),
                "interval_relation": "crosses_multiple_nremrem_intervals",
                "overlapped_nremrem_cycle_indices": tuple(
                    int(x[2]) for x in overlaps
                ),
            }

        return {
            "location": "outside",
            "containing_nremrem_cycle_index": None,
            "interval_relation": "outside",
            "overlapped_nremrem_cycle_indices": tuple(
                int(x[2]) for x in overlaps
            ),
        }

    # Original cell 57, line 89.
    def _force_inside_if_observed_loop_contains_rem(
        interval_info,
        observed_loop,
        rem_state=4,
        rem_observed_loop_counts_as_inside=True,
    ):
        """
        Only for complete AllCycle loop occurrences.

        If the actually observed hypnogram loop fragment contains REM,
        force this loop occurrence to inside.

        Reversible exchange does not use this override.
        """

        out = dict(interval_info)

        observed_loop = tuple(int(x) for x in observed_loop)
        observed_loop_contains_rem = int(rem_state) in set(observed_loop)

        out["observed_loop_contains_rem"] = bool(observed_loop_contains_rem)
        out["location_before_rem_override"] = interval_info["location"]
        out["interval_relation_before_rem_override"] = interval_info[
            "interval_relation"
        ]
        out["rem_inside_override_applied"] = False

        if not rem_observed_loop_counts_as_inside:
            return out

        if not observed_loop_contains_rem:
            return out

        if interval_info["location"] == "inside":
            return out

        overlapped = tuple(
            int(x) for x in interval_info.get(
                "overlapped_nremrem_cycle_indices",
                tuple(),
            )
        )

        if len(overlapped) == 1:
            containing_idx = int(overlapped[0])
            relation = (
                "observed_loop_contains_rem_treated_as_inside_"
                "partial_overlap_one_interval"
            )
        elif len(overlapped) >= 2:
            containing_idx = tuple(overlapped)
            relation = (
                "observed_loop_contains_rem_treated_as_inside_"
                "overlaps_multiple_intervals"
            )
        else:
            containing_idx = None
            relation = (
                "observed_loop_contains_rem_treated_as_inside_"
                "without_nremrem_interval_overlap"
            )

        out["location"] = "inside"
        out["containing_nremrem_cycle_index"] = containing_idx
        out["interval_relation"] = relation
        out["rem_inside_override_applied"] = True

        return out

    # Original cell 57, line 184.
    def scan_complete_flux_loop_occurrences_one_hypnogram(
        hypnogram,
        hyp_index,
        subject,
        AllCycle,
        intervals,
        epoch_sec=30,
        rem_state=4,
        rem_observed_loop_counts_as_inside=True,
    ):
        """
        Find complete temporal occurrences of AllCycle flux loops
        across the whole hypnogram.

        Covered edges:
            run-state transition edges from p to p + loop_n_edges - 1

        Location:
            1. original interval rule
            2. if actually observed loop fragment contains REM,
               force this complete loop occurrence to inside
        """

        h = to_1d_hypnogram(hypnogram)
        run_states, run_starts, run_ends = find_runs(h)

        rows = []

        loop_families = unique_flux_loop_families_for_hypnogram(
            AllCycle=AllCycle,
            hyp_index=hyp_index,
            subject=subject,
        )

        if len(loop_families) == 0:
            return rows

        for lf in loop_families:
            loop_family_code = lf["loop_family_code"]
            loop_n_edges = int(lf["loop_n_edges"])
            window_len = loop_n_edges + 1

            if len(run_states) < window_len:
                continue

            for p in range(0, len(run_states) - window_len + 1):
                candidate = tuple(
                    int(x) for x in run_states[p:p + window_len]
                )

                if len(candidate) <= 1:
                    continue

                if candidate[0] != candidate[-1]:
                    continue

                candidate_code = canonical_rotation_code(candidate)

                if candidate_code != loop_family_code:
                    continue

                start_epoch = int(run_starts[p])
                return_epoch = int(run_starts[p + loop_n_edges])
                terminal_end_epoch = int(run_ends[p + loop_n_edges])

                duration_epochs = int(return_epoch - start_epoch)
                duration_min = duration_epochs * epoch_sec / 60.0

                duration_including_terminal_epochs = int(
                    terminal_end_epoch - start_epoch + 1
                )
                duration_including_terminal_min = (
                    duration_including_terminal_epochs * epoch_sec / 60.0
                )

                original_interval_info = classify_interval_location(
                    start_epoch=start_epoch,
                    end_epoch=return_epoch,
                    intervals=intervals,
                )

                interval_info = _force_inside_if_observed_loop_contains_rem(
                    interval_info=original_interval_info,
                    observed_loop=candidate,
                    rem_state=rem_state,
                    rem_observed_loop_counts_as_inside=(
                        rem_observed_loop_counts_as_inside
                    ),
                )

                covered_edge_indices = tuple(
                    int(k) for k in range(p, p + loop_n_edges)
                )

                rows.append({
                    "hyp_index": int(hyp_index),
                    "subject": subject,

                    "location": interval_info["location"],
                    "containing_nremrem_cycle_index": interval_info[
                        "containing_nremrem_cycle_index"
                    ],
                    "interval_relation": interval_info["interval_relation"],
                    "overlapped_nremrem_cycle_indices": interval_info[
                        "overlapped_nremrem_cycle_indices"
                    ],

                    "observed_loop_contains_rem": interval_info[
                        "observed_loop_contains_rem"
                    ],
                    "rem_inside_override_applied": interval_info[
                        "rem_inside_override_applied"
                    ],
                    "location_before_rem_override": interval_info[
                        "location_before_rem_override"
                    ],
                    "interval_relation_before_rem_override": interval_info[
                        "interval_relation_before_rem_override"
                    ],

                    "loop_family_code": loop_family_code,
                    "loop": lf["loop"],
                    "loop_code": lf["loop_code"],
                    "loop_name": lf["loop_name"],
                    "loop_n_edges": loop_n_edges,

                    "observed_loop": candidate,
                    "observed_loop_code": structure_to_code(candidate),
                    "observed_loop_name": structure_to_name(candidate),

                    "start_epoch": start_epoch,
                    "return_epoch": return_epoch,
                    "terminal_end_epoch": terminal_end_epoch,

                    "start_run_index": int(p),
                    "return_run_index": int(p + loop_n_edges),
                    "covered_edge_indices": covered_edge_indices,

                    "duration_min": float(duration_min),
                    "duration_including_terminal_min": float(
                        duration_including_terminal_min
                    ),
                })

        return rows

    # Original cell 57, line 336.
    def scan_maximal_reversible_exchange_blocks_one_hypnogram(
        hypnogram,
        hyp_index,
        subject,
        intervals,
        epoch_sec=30,
        reversible_states=None,
        count_open_alternating_blocks=True,
        covered_edge_indices=None,
    ):
        """
        Reversible exchange = every transition edge in the whole hypnogram
        that is not covered by any complete AllCycle loop occurrence.

        No REM override is applied here.
        """

        h = to_1d_hypnogram(hypnogram)
        run_states, run_starts, run_ends = find_runs(h)

        rows = []

        if len(run_states) < 2:
            return rows

        if covered_edge_indices is None:
            covered_edge_indices = set()
        else:
            covered_edge_indices = set(int(x) for x in covered_edge_indices)

        if reversible_states is not None:
            reversible_states = set(map(int, reversible_states))

        n_edges_total = len(run_states) - 1

        for edge_index in range(n_edges_total):
            if edge_index in covered_edge_indices:
                continue

            a = int(run_states[edge_index])
            b = int(run_states[edge_index + 1])

            if a == b:
                continue

            if reversible_states is not None:
                if (a not in reversible_states) or (b not in reversible_states):
                    continue

            start_epoch = int(run_starts[edge_index])
            end_epoch = int(run_starts[edge_index + 1])

            edge_duration_min = (
                (end_epoch - start_epoch) * epoch_sec / 60.0
            )

            if edge_duration_min <= 0:
                continue

            pair = tuple(sorted((a, b)))
            pair_code = f"{pair[0]}{pair[1]}"
            pair_name = (
                f"{STATE_NAME_MAP.get(pair[0], str(pair[0]))}"
                f"-"
                f"{STATE_NAME_MAP.get(pair[1], str(pair[1]))}"
            )

            block_states = (a, b)

            interval_info = classify_interval_location(
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                intervals=intervals,
            )

            rows.append({
                "hyp_index": int(hyp_index),
                "subject": subject,

                "location": interval_info["location"],
                "containing_nremrem_cycle_index": interval_info[
                    "containing_nremrem_cycle_index"
                ],
                "interval_relation": interval_info["interval_relation"],
                "overlapped_nremrem_cycle_indices": interval_info[
                    "overlapped_nremrem_cycle_indices"
                ],

                "start_epoch": start_epoch,
                "end_epoch": end_epoch,

                "reversible_block": block_states,
                "reversible_block_code": structure_to_code(block_states),
                "reversible_block_name": structure_to_name(block_states),
                "reversible_block_n_edges": 1,
                "is_closed_block": False,

                "exchange_pair": pair,
                "exchange_pair_code": pair_code,
                "exchange_pair_name": pair_name,

                "edge_index": int(edge_index),
                "edge_from_state": a,
                "edge_to_state": b,
                "edge_from_name": STATE_NAME_MAP.get(a, str(a)),
                "edge_to_name": STATE_NAME_MAP.get(b, str(b)),
                "covered_by_loop": False,

                "full_block_duration_min": float(edge_duration_min),
                "exchange_unit_duration_min": float(edge_duration_min),
            })

        return rows

    # Original cell 57, line 455.
    def analyze_inside_outside_duration_significance_by_category(data, AllCycle, epoch_sec=30, wake_state=0, rem_state=4, nrem_states=(1, 2, 3), min_nrem_min=0, min_rem_min=0, include_post_rem_state=True, include_wake_in_structure=False, inside_uses_structure_end=True, rem_observed_loop_counts_as_inside=True, reversible_states=None, count_open_alternating_blocks=True, min_n_per_group=3, alpha=0.05):
        """
        Pooled occurrence-level category analysis.

        Irreversible component:
            complete AllCycle loop occurrences detected over the whole hypnogram.

        Reversible component:
            all hypnogram transition edges not covered by those loop occurrences.

        Loop location:
            original interval rule,
            plus observed-loop contains REM -> inside.

        Reversible location:
            original interval rule only.
        """
        df_nremrem_intervals = extract_nremrem_intervals_all_hypnograms(data=data, epoch_sec=epoch_sec, wake_state=wake_state, rem_state=rem_state, nrem_states=nrem_states, min_nrem_min=min_nrem_min, min_rem_min=min_rem_min, include_post_rem_state=include_post_rem_state, include_wake_in_structure=include_wake_in_structure)
        interval_lookup = build_interval_lookup(df_nremrem_intervals=df_nremrem_intervals, inside_uses_structure_end=inside_uses_structure_end)
        reversible_rows_all = []
        loop_rows_all = []
        for hyp_index, subject, hyp in iter_hypnograms(data):
            intervals = interval_lookup.get(int(hyp_index), [])
            loop_rows = scan_complete_flux_loop_occurrences_one_hypnogram(hypnogram=hyp, hyp_index=hyp_index, subject=subject, AllCycle=AllCycle, intervals=intervals, epoch_sec=epoch_sec, rem_state=rem_state, rem_observed_loop_counts_as_inside=rem_observed_loop_counts_as_inside)
            covered_edge_indices = set()
            for row in loop_rows:
                for edge_index in row.get('covered_edge_indices', tuple()):
                    covered_edge_indices.add(int(edge_index))
            reversible_rows = scan_maximal_reversible_exchange_blocks_one_hypnogram(hypnogram=hyp, hyp_index=hyp_index, subject=subject, intervals=intervals, epoch_sec=epoch_sec, reversible_states=reversible_states, count_open_alternating_blocks=count_open_alternating_blocks, covered_edge_indices=covered_edge_indices)
            loop_rows_all.extend(loop_rows)
            reversible_rows_all.extend(reversible_rows)
        df_reversible_occurrence = pd.DataFrame(reversible_rows_all)
        df_loop_occurrence = pd.DataFrame(loop_rows_all)
        return {'df_nremrem_intervals': df_nremrem_intervals, 'df_reversible_occurrence': df_reversible_occurrence, 'df_loop_occurrence': df_loop_occurrence}

    return {'analyze_inside_outside_duration_significance_by_category': analyze_inside_outside_duration_significance_by_category}

_inside_outside_helpers = _build_inside_outside_helpers()
analyze_inside_outside_duration_significance_by_category = _inside_outside_helpers['analyze_inside_outside_duration_significance_by_category']


# ======================================================================
# Definitions from cleaned notebook cell 34 (zero-based)
# ======================================================================

def _build_loop_matrix_helpers():
    # Original cell 147, line 194.
    import pandas as pd

    # Original cell 147, line 10.
    def canonical_loop(loop, merge_rotations=True, merge_reverse=False):
        """
        将 loop 规范化。

        例如:
            [0, 1, 2, 4, 0]

        merge_rotations=True:
            [0,1,2,4,0] 和 [1,2,4,0,1] 视为同一个 loop。

        merge_reverse=False:
            默认不合并反向 loop。
            因为非平衡 flux 的方向是有意义的。
        """
        loop = list(loop)

        if len(loop) < 2:
            raise ValueError(f"Invalid loop: {loop}")

        # 去掉最后重复的起点
        if loop[0] == loop[-1]:
            nodes = loop[:-1]
        else:
            nodes = loop[:]

        if len(nodes) == 0:
            raise ValueError(f"Invalid loop after removing repeated endpoint: {loop}")

        if not merge_rotations:
            return tuple(nodes + [nodes[0]])

        n = len(nodes)

        # 方向不变，只合并循环起点
        rotations = []
        for i in range(n):
            r = nodes[i:] + nodes[:i]
            rotations.append(tuple(r))

        candidates = rotations

        # 可选：是否把反向 loop 也合并
        if merge_reverse:
            rev = list(reversed(nodes))
            rev_rotations = []
            for i in range(n):
                r = rev[i:] + rev[:i]
                rev_rotations.append(tuple(r))
            candidates = rotations + rev_rotations

        best = min(candidates)
        return tuple(list(best) + [best[0]])

    # Original cell 147, line 64.
    def loop_to_str(loop_tuple):
        """
        将 loop tuple 转成字符串。
        例如:
            (0,1,2,4,0) -> "01240"
        """
        return "".join(map(str, loop_tuple))

    # Original cell 147, line 77.
    def build_loop_strength_matrix(
        per_subject_detail,
        merge_rotations=True,
        merge_reverse=False,
    ):
        """
        输入:
            per_subject_detail = {
                subject_id: {
                    "loops": [...],
                    "strengths": [...]
                }
            }

        输出:
            loop_matrix:
                index = subject
                columns = loop
                values = flux strength
        """

        subjects = sorted(per_subject_detail.keys())
        all_loops = set()

        # 先收集所有出现过的 loop
        for sid in subjects:
            loops = per_subject_detail[sid].get("loops", [])
            for loop in loops:
                canon = canonical_loop(
                    loop,
                    merge_rotations=merge_rotations,
                    merge_reverse=merge_reverse,
                )
                all_loops.add(canon)

        all_loops = sorted(all_loops)
        loop_cols = [loop_to_str(lp) for lp in all_loops]

        # 初始化 subject × loop 矩阵
        loop_matrix = pd.DataFrame(
            0.0,
            index=subjects,
            columns=loop_cols,
        )

        # 填入每个 subject 的 loop flux
        for sid in subjects:
            loops = per_subject_detail[sid].get("loops", [])
            strengths = per_subject_detail[sid].get("strengths", [])

            if len(loops) != len(strengths):
                raise ValueError(
                    f"Subject {sid}: loops and strengths have different lengths. "
                    f"len(loops)={len(loops)}, len(strengths)={len(strengths)}"
                )

            for loop, strength in zip(loops, strengths):
                canon = canonical_loop(
                    loop,
                    merge_rotations=merge_rotations,
                    merge_reverse=merge_reverse,
                )
                col = loop_to_str(canon)

                # 如果同一个 subject 里同一个规范化 loop 出现多次，累加
                loop_matrix.loc[sid, col] += float(strength)

        return loop_matrix

    return {'build_loop_strength_matrix': build_loop_strength_matrix}

_loop_matrix_helpers = _build_loop_matrix_helpers()
build_loop_strength_matrix = _loop_matrix_helpers['build_loop_strength_matrix']


# ======================================================================
# Definitions from cleaned notebook cell 37 (zero-based)
# ======================================================================

def _build_simulation_helpers():
    # Original cell 156, line 1.
    import numpy as np

    # Original cell 156, line 2.
    import pandas as pd

    # Original cell 155, line 519.
    def _stationary_distribution(T, tol=1e-12):
        """
        计算离散时间 Markov 矩阵 T 的稳态分布 pi，满足
            pi @ T = pi,   sum(pi)=1
        """
        T = np.asarray(T, dtype=float)
        n = T.shape[0]

        A = T.T - np.eye(n)
        A[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0

        pi = np.linalg.solve(A, b)
        pi = np.real_if_close(pi)

        # 数值清理
        pi[np.abs(pi) < tol] = 0.0

        if np.any(pi < -tol):
            raise ValueError("Stationary distribution has negative entries.")
        pi = np.clip(pi, 0.0, None)

        s = pi.sum()
        if s <= tol:
            raise ValueError("Failed to compute a valid stationary distribution.")
        pi /= s
        return pi

    # Original cell 155, line 549.
    def _loop_edges(loop):
        """
        loop = (i1, i2, ..., ik)
        返回 [(i1,i2), (i2,i3), ..., (ik,i1)]
        """
        loop = tuple(loop)
        if len(loop) < 2:
            raise ValueError("Loop length must be at least 2.")
        return [(loop[t], loop[(t + 1) % len(loop)]) for t in range(len(loop))]

    # Original cell 155, line 560.
    def _contains_loop_in_Jplus(Jp, loop, tol=1e-12):
        """
        若 loop 上每条边的 J+ 都 > tol，则认为 J+ 包含该 loop
        """
        for i, j in _loop_edges(loop):
            if Jp[i, j] <= tol:
                return False
        return True

    # Original cell 155, line 570.
    def _greedy_loop_flux(Jp, loop):
        """
        greedy loop flux = loop 上所有边流的最小值
        """
        return min(Jp[i, j] for i, j in _loop_edges(loop))

    # Original cell 155, line 577.
    def _remove_loop_from_Jplus(Jp, loop, target_ratio=1.0, tol=1e-12):
        """
        从 J+ 中删除指定 loop 的某个比例。

        参数
        ----
        Jp : array
            J+ 矩阵
        loop : tuple/list
            指定 loop
        target_ratio : float in [0, 1]
            删除比例：
              - 1.0 -> 完全删除该 loop 一次（删掉 greedy flux）
              - 0.5 -> 删除 greedy flux 的一半
              - 0.0 -> 不删除
        tol : float
            数值容差

        返回
        ----
        Jp_new : array or None
        full_flux : float or None
            该 loop 的 greedy flux
        removed_flux : float or None
            实际删掉的 flux = target_ratio * full_flux
        """
        Jp = np.asarray(Jp, dtype=float).copy()

        if not (0.0 <= target_ratio <= 1.0):
            raise ValueError("target_ratio must be in [0, 1].")

        if not _contains_loop_in_Jplus(Jp, loop, tol=tol):
            return None, None, None

        full_flux = _greedy_loop_flux(Jp, loop)
        removed_flux = target_ratio * full_flux

        if removed_flux <= tol:
            return Jp, full_flux, 0.0

        for i, j in _loop_edges(loop):
            Jp[i, j] -= removed_flux
            if abs(Jp[i, j]) <= tol:
                Jp[i, j] = 0.0
            if Jp[i, j] < -tol:
                raise ValueError("Loop subtraction produced negative J+ entry.")

        return Jp, full_flux, removed_flux

    # Original cell 155, line 627.
    def _reconstruct_T_from_Jplus_and_original_T(Jp_new, T_orig, tol=1e-12):
        """
        在“保持原始对称 off-diagonal edge flow 不变”的条件下，
        从新的 J+ 重构新的 T。

        新规则：
          - 若某个状态 i 的稳态概率 pi[i] <= tol，则该状态对应的 T[i, :]
            保持原样不变，不参与重构。
          - 只对 pi[i] > tol 的状态子集进行重构。

        具体做法：
          1) 原始稳态分布 pi 来自 T_orig
          2) 找到 active states: pi[i] > tol
          3) 仅在 active 子空间里：
                F = diag(pi_active) @ T_active
                D_ij = min(F_ij, F_ji), i != j
                F'_ij = D_ij + Jp_new_ij
                F'_ii = pi_i - sum_{j != i} F'_ij
                T'_active = diag(pi_active)^{-1} F'
          4) inactive states (pi=0) 的整行 T 保持不变
          5) active -> inactive 的概率置为 0
             （这与 pi[inactive]=0 的稳态条件一致）

        注意：
          这里保持不变的是 edge flow 的对称部分，不是 T 本身的对称部分。
        """
        T_orig = np.asarray(T_orig, dtype=float)
        Jp_new = np.asarray(Jp_new, dtype=float)

        if T_orig.ndim != 2 or T_orig.shape[0] != T_orig.shape[1]:
            raise ValueError("T_orig must be a square matrix.")
        if Jp_new.shape != T_orig.shape:
            raise ValueError("Shape mismatch between Jp_new and T_orig.")

        n = T_orig.shape[0]

        # J+ 理论上应无对角元
        Jp_new = Jp_new.copy()
        np.fill_diagonal(Jp_new, 0.0)

        pi = _stationary_distribution(T_orig, tol=tol)

        active = pi > tol
        inactive = ~active

        if not np.any(active):
            raise ValueError("No active states with positive stationary probability.")

        # 默认先保持原矩阵
        T_new = T_orig.copy()

        # 如果所有状态都 inactive（理论上不会）或只有 inactive，不处理
        active_idx = np.where(active)[0]
        m = len(active_idx)

        # 只在 active 子空间内重构
        T_sub = T_orig[np.ix_(active, active)]
        Jp_sub = Jp_new[np.ix_(active, active)]
        pi_sub = pi[active]

        # 原始 active 子空间中的 edge flow
        F_orig_sub = np.diag(pi_sub) @ T_sub

        # 对称 off-diagonal 部分 D
        D_sub = np.zeros_like(F_orig_sub)
        for a in range(m):
            for b in range(m):
                if a != b:
                    D_sub[a, b] = min(F_orig_sub[a, b], F_orig_sub[b, a])

        # 新的 active 子空间 off-diagonal flow
        F_new_sub = np.zeros_like(F_orig_sub)
        for a in range(m):
            for b in range(m):
                if a != b:
                    F_new_sub[a, b] = D_sub[a, b] + Jp_sub[a, b]

        # 用固定 pi_sub 补对角元
        for a in range(m):
            offdiag_sum = F_new_sub[a, :].sum()
            F_new_sub[a, a] = pi_sub[a] - offdiag_sum

            if F_new_sub[a, a] < -tol:
                raise ValueError(
                    f"Reconstructed diagonal flow became negative at active state "
                    f"{active_idx[a]}: {F_new_sub[a, a]}"
                )
            if abs(F_new_sub[a, a]) <= tol:
                F_new_sub[a, a] = 0.0

        # 转回 active 子空间中的 T
        T_sub_new = np.zeros_like(T_sub)
        for a in range(m):
            T_sub_new[a, :] = F_new_sub[a, :] / pi_sub[a]

        # 数值清理
        T_sub_new[np.abs(T_sub_new) < tol] = 0.0

        # 检查 active 子空间合法性
        row_sums_sub = T_sub_new.sum(axis=1)
        if not np.allclose(row_sums_sub, 1.0, atol=1e-8):
            raise ValueError("Reconstructed active submatrix does not have row sums equal to 1.")
        if np.any(T_sub_new < -1e-10):
            raise ValueError("Reconstructed active submatrix has negative entries.")

        T_sub_new = np.clip(T_sub_new, 0.0, None)
        T_sub_new = T_sub_new / T_sub_new.sum(axis=1, keepdims=True)

        # 写回：
        # 1) active -> active 用重构结果
        T_new[np.ix_(active, active)] = T_sub_new

        # 2) active -> inactive 置 0
        #    这样 active 行仍是完整概率分布，且与 pi[inactive]=0 一致
        if np.any(inactive):
            T_new[np.ix_(active, inactive)] = 0.0

        # 3) inactive 行完全保持原样，不重构
        #    T_new[inactive, :] 已经是 T_orig[inactive, :]

        # 最终检查
        T_new[np.abs(T_new) < tol] = 0.0
        if np.any(T_new < -1e-10):
            raise ValueError("Reconstructed T has negative entries.")

        T_new = np.clip(T_new, 0.0, None)

        row_sums = T_new.sum(axis=1)
        # if not np.allclose(row_sums, 1.0, atol=1e-8):
        #     print(T_new)
        #     raise ValueError("Final reconstructed T does not have row sums equal to 1.")

        return T_new, pi

    # Original cell 155, line 762.
    def modify_loops_and_reconstruct_Ts(
        Jss,
        Ts,
        loop=None,
        remove_all_loops=False,
        target_ratio=1.0,
        tol=1e-12,
        return_details=False,
    ):
        """
        通用接口：

        模式 A：删除指定 loop 的一定比例
            remove_all_loops=False, loop=(...), target_ratio in [0,1]

        模式 B：删除所有 loop，仅保留平衡成分
            remove_all_loops=True
            此时相当于对每个 J+ 直接设 Jp_new = 0

        参数
        ----
        Jss : list of (n,n) arrays
            每个元素是 J+ 矩阵
        Ts : list of (n,n) arrays
            对应的原始转移矩阵 T
        loop : tuple/list or None
            指定的有向 loop，例如 (0,1,2)
            当 remove_all_loops=True 时可忽略
        remove_all_loops : bool
            True -> 去掉所有非平衡 loop，仅保留平衡成分
        target_ratio : float in [0,1]
            删除指定 loop 的比例
        tol : float
            数值容差
        return_details : bool
            True 时返回附加信息

        返回
        ----
        T_list : list
            - 指定 loop 模式下：
                若第 i 个 J+ 包含该 loop，则返回重构后的 T_new；
                否则返回 None
            - 删除所有 loop 模式下：
                对每个样本都返回 equilibrium-only 的 T_new

        details : list of dict, optional
        """
        if len(Jss) != len(Ts):
            raise ValueError("Jss and Ts must have the same length.")

        if not remove_all_loops:
            if loop is None:
                raise ValueError("When remove_all_loops=False, you must provide loop.")
            loop = tuple(loop)

        if not (0.0 <= target_ratio <= 1.0):
            raise ValueError("target_ratio must be in [0,1].")

        T_list = []
        details = []

        for idx, (Jp, T_orig) in enumerate(zip(Jss, Ts)):
            Jp = np.asarray(Jp, dtype=float)
            T_orig = np.asarray(T_orig, dtype=float)

            if Jp.shape != T_orig.shape:
                raise ValueError(f"Shape mismatch at index {idx}: Jp {Jp.shape}, T {T_orig.shape}")

            if remove_all_loops:
                # 直接令所有非平衡部分为 0，仅保留平衡成分
                Jp_new = np.zeros_like(Jp)
                T_new, pi = _reconstruct_T_from_Jplus_and_original_T(Jp_new, T_orig, tol=tol)

                T_list.append(T_new)
                details.append({
                    "index": idx,
                    "mode": "remove_all_loops",
                    "contains_loop": None,
                    "full_loop_flux": None,
                    "removed_flux": None,
                    "target_ratio": None,
                    "Jp_new": Jp_new,
                    "T_new": T_new,
                    "pi": pi,
                })
                continue

            # 删除指定 loop 的一定比例
            contains = _contains_loop_in_Jplus(Jp, loop, tol=tol)

            if not contains:
                T_list.append(None)
                details.append({
                    "index": idx,
                    "mode": "remove_specified_loop",
                    "contains_loop": False,
                    "full_loop_flux": None,
                    "removed_flux": None,
                    "target_ratio": target_ratio,
                    "Jp_new": None,
                    "T_new": None,
                    "pi": None,
                })
                continue

            Jp_new, full_flux, removed_flux = _remove_loop_from_Jplus(
                Jp, loop, target_ratio=target_ratio, tol=tol
            )
            T_new, pi = _reconstruct_T_from_Jplus_and_original_T(Jp_new, T_orig, tol=tol)

            T_list.append(T_new)
            details.append({
                "index": idx,
                "mode": "remove_specified_loop",
                "contains_loop": True,
                "full_loop_flux": full_flux,
                "removed_flux": removed_flux,
                "target_ratio": target_ratio,
                "Jp_new": Jp_new,
                "T_new": T_new,
                "pi": pi,
            })

        if return_details:
            return T_list, details
        return T_list

    # Original cell 156, line 9.
    def normalize_transition_matrix(T, tol=1e-12):
        """
        保证每一行是合法概率分布。
        对于行和 > tol 的行，做归一化；
        对于全 0 行，保持全 0。
        """
        T = np.asarray(T, dtype=float).copy()

        if T.ndim != 2 or T.shape[0] != T.shape[1]:
            raise ValueError(f"T must be a square matrix, got shape {T.shape}")

        T[np.abs(T) < tol] = 0.0

        if np.any(T < -tol):
            raise ValueError("T contains negative entries beyond tolerance.")

        T = np.clip(T, 0.0, None)

        row_sums = T.sum(axis=1, keepdims=True)
        nonzero_rows = row_sums[:, 0] > tol

        if np.any(nonzero_rows):
            T[nonzero_rows] = T[nonzero_rows] / row_sums[nonzero_rows]

        if np.any(~nonzero_rows):
            T[~nonzero_rows] = 0.0

        return T

    # Original cell 156, line 39.
    def simulate_markov_chain(T, init_state=0, n_steps=200, rng=None, tol=1e-12):
        """
        用转移矩阵 T 模拟一条 Markov hypnogram。

        返回：
            states: 长度为 n_steps + 1
        """
        if rng is None:
            rng = np.random.default_rng()

        T = normalize_transition_matrix(T, tol=tol)
        n_states = T.shape[0]

        states = np.empty(n_steps + 1, dtype=int)
        states[0] = init_state

        for t in range(n_steps):
            s = states[t]
            p = T[s]

            if p.sum() <= tol:
                states[t + 1] = s
            else:
                states[t + 1] = rng.choice(n_states, p=p)

        return states

    # Original cell 156, line 67.
    def compress_consecutive_states(states):
        """
        压缩连续重复状态。
        例如：
            0,0,1,1,2,2,4 -> 0,1,2,4
        """
        states = np.asarray(states, dtype=int)

        if states.size == 0:
            return states.copy()

        keep = np.empty(states.size, dtype=bool)
        keep[0] = True
        keep[1:] = states[1:] != states[:-1]

        return states[keep]

    # Original cell 156, line 85.
    def _safe_nanmean(x):
        x = np.asarray(x, dtype=float)
        if np.all(np.isnan(x)):
            return np.nan
        return float(np.nanmean(x))

    # Original cell 156, line 96.
    def simulate_hypnograms_for_one_matrix(
        T,
        n_sims=100,
        n_steps=200,
        init_state=0,
        seed=None,
        tol=1e-12,
    ):
        """
        对单个 T 模拟 n_sims 条 hypnogram。

        返回：
            {
                "states": array, shape = (n_sims, n_steps + 1),
                "normalized_T": T_norm,
                "n_steps_used": n_steps,
                "n_sims": n_sims,
            }
        """
        rng = np.random.default_rng(seed)
        T_norm = normalize_transition_matrix(T, tol=tol)

        all_states = []

        for _ in range(n_sims):
            states = simulate_markov_chain(
                T=T_norm,
                init_state=init_state,
                n_steps=int(n_steps),
                rng=rng,
                tol=tol,
            )
            all_states.append(states)

        all_states = np.stack(all_states, axis=0)

        return {
            "states": all_states,
            "normalized_T": T_norm,
            "n_steps_used": int(n_steps),
            "n_sims": int(n_sims),
        }

    # Original cell 156, line 140.
    def generate_simulated_hypnogram_data(
        T_new_list,
        Tmat,
        AllSleepLength,
        n_sims=100,
        init_state=0,
        base_seed=1234,
        tol=1e-12,
        simulate_orig_when_new_none=False,
    ):
        """
        Step 1 主函数：只生成模拟数据，不做任何 cycle 统计。

        输入：
            T_new_list:
                去掉 loop 后的 T 列表。
                如果某个 subject 没有 T_new，可以是 None。

            Tmat:
                原始 T 列表。

            AllSleepLength:
                每个 subject 的模拟长度，单位为 epoch 数。

        输出：
            sim_data:
                {
                    "new": list,
                    "orig": list,
                    "meta": dict
                }

                sim_data["new"][i]["states"]  shape = (n_sims, n_steps_i + 1)
                sim_data["orig"][i]["states"] shape = (n_sims, n_steps_i + 1)
        """
        if len(T_new_list) != len(Tmat):
            raise ValueError("T_new_list and Tmat must have the same length.")

        AllSleepLength = np.asarray(AllSleepLength)

        if AllSleepLength.ndim != 1:
            raise ValueError("AllSleepLength must be 1D.")

        if len(AllSleepLength) != len(Tmat):
            raise ValueError("AllSleepLength must have same length as Tmat.")

        n_subjects = len(Tmat)

        sim_new = [None] * n_subjects
        sim_orig = [None] * n_subjects

        for i in range(n_subjects):
            n_steps_i = AllSleepLength[i]

            if np.isnan(n_steps_i):
                continue

            n_steps_i = int(round(n_steps_i))

            if n_steps_i <= 0:
                continue

            T_new = T_new_list[i]
            T_old = Tmat[i]

            seed_new = base_seed + 10000 * i + 1
            seed_old = base_seed + 10000 * i + 2

            # ------------------------------
            # new condition
            # ------------------------------
            if T_new is not None:
                sim_new[i] = simulate_hypnograms_for_one_matrix(
                    T=T_new,
                    n_sims=n_sims,
                    n_steps=n_steps_i,
                    init_state=init_state,
                    seed=seed_new,
                    tol=tol,
                )

            # ------------------------------
            # orig condition
            # ------------------------------
            if (T_new is not None) or simulate_orig_when_new_none:
                sim_orig[i] = simulate_hypnograms_for_one_matrix(
                    T=T_old,
                    n_sims=n_sims,
                    n_steps=n_steps_i,
                    init_state=init_state,
                    seed=seed_old,
                    tol=tol,
                )

        sim_data = {
            "new": sim_new,
            "orig": sim_orig,
            "meta": {
                "n_subjects": n_subjects,
                "n_sims": n_sims,
                "init_state": init_state,
                "base_seed": base_seed,
                "tol": tol,
            }
        }

        return sim_data

    # Original cell 156, line 253.
    def extract_metrics_from_traj(states, wake_state=0):
        """
        从单条 hypnogram 中提取：
        1. sleep latency
        2. wake count after sleep onset
        3. state switch count after sleep onset
        """
        states = np.asarray(states, dtype=int)

        sleep_onset_candidates = np.where(states != wake_state)[0]

        if len(sleep_onset_candidates) == 0:
            sleep_latency = np.nan
            wake_count_after_sleep = 0
            state_switch_count_after_sleep = 0
            return sleep_latency, wake_count_after_sleep, state_switch_count_after_sleep

        sleep_onset = sleep_onset_candidates[0]
        sleep_latency = sleep_onset

        after_states = states[sleep_onset:]

        state_switch_count_after_sleep = int(
            np.sum(after_states[1:] != after_states[:-1])
        )

        wake_count_after_sleep = int(
            np.sum(
                (after_states[:-1] != wake_state)
                & (after_states[1:] == wake_state)
            )
        )

        return sleep_latency, wake_count_after_sleep, state_switch_count_after_sleep

    # Original cell 156, line 289.
    def count_loops_in_hypnogram(states, min_unique_states=3):
        """
        统计一般 loop 数量。

        逻辑：
        1. 先压缩连续重复状态；
        2. 再贪心、非重叠地找首尾相同的 loop；
        3. 默认至少包含 3 个 unique states。
        """
        hyp = compress_consecutive_states(states)

        if hyp.size == 0:
            return {
                "compressed_hypnogram": hyp,
                "n_loops": 0,
                "loops": [],
            }

        path = [int(hyp[0])]
        loops = []

        for s in hyp[1:]:
            s = int(s)
            path.append(s)

            prev_idx = [j for j, x in enumerate(path[:-1]) if x == s]

            chosen_j = None

            for j in prev_idx:
                candidate = path[j:]
                n_unique = len(set(candidate[:-1]))

                if n_unique >= min_unique_states:
                    chosen_j = j
                    break

            if chosen_j is not None:
                loop = path[chosen_j:].copy()
                loops.append(loop)
                path = [path[-1]]

        return {
            "compressed_hypnogram": hyp,
            "n_loops": len(loops),
            "loops": loops,
        }

    # Original cell 156, line 338.
    def states_to_bouts(states):
        """
        将 hypnogram 转成 bout 序列。

        例如：
            states = [0,0,1,1,1,2,2,4,4]
        返回：
            [(0,2), (1,3), (2,2), (4,2)]

        第二个元素是该 bout 持续的 epoch 数。
        """
        states = np.asarray(states, dtype=int)

        if states.size == 0:
            return []

        bouts = []

        current_state = int(states[0])
        current_len = 1

        for s in states[1:]:
            s = int(s)

            if s == current_state:
                current_len += 1
            else:
                bouts.append((current_state, current_len))
                current_state = s
                current_len = 1

        bouts.append((current_state, current_len))

        return bouts

    # Original cell 156, line 374.
    def count_nrem_rem_cycles_with_min_duration(
        states,
        wake_state=0,
        rem_state=4,
        nrem_states=(1, 2, 3),
        epoch_sec=30,
        min_nrem_min=0.0,
        min_rem_min=0.0,
        count_after_sleep_onset=True,
        wake_breaks_cycle=True,
    ):
        """
        带 duration threshold 的 NREM-REM cycle 统计。

        定义：
            一个 cycle = 累积 NREM 时长 >= min_nrem_min，
                      随后进入一个 REM bout，
                      且该 REM bout 时长 >= min_rem_min。

        默认：
            wake_breaks_cycle=True
            即如果 NREM 和 REM 之间出现 Wake，则之前累积的 NREM 作废。

        参数：
            min_nrem_min:
                NREM 最小时长，单位分钟。

            min_rem_min:
                REM 最小时长，单位分钟。

        返回：
            n_cycles
        """
        states = np.asarray(states, dtype=int)

        if states.size == 0:
            return 0

        if count_after_sleep_onset:
            sleep_onset_candidates = np.where(states != wake_state)[0]

            if len(sleep_onset_candidates) == 0:
                return 0

            states = states[sleep_onset_candidates[0]:]

        nrem_states = set(nrem_states)

        min_nrem_epochs = int(np.ceil(min_nrem_min * 60.0 / epoch_sec))
        min_rem_epochs = int(np.ceil(min_rem_min * 60.0 / epoch_sec))

        bouts = states_to_bouts(states)

        n_cycles = 0
        nrem_accum_epochs = 0

        for state, length in bouts:
            state = int(state)
            length = int(length)

            if state in nrem_states:
                nrem_accum_epochs += length

            elif state == rem_state:
                if (
                    nrem_accum_epochs >= min_nrem_epochs
                    and length >= min_rem_epochs
                ):
                    n_cycles += 1

                # 进入 REM 后，无论这个 REM 是否满足阈值，
                # 前面这段 NREM accumulation 都结束。
                nrem_accum_epochs = 0

            else:
                # Wake 或其他非 NREM / 非 REM 状态
                if wake_breaks_cycle:
                    nrem_accum_epochs = 0

        return int(n_cycles)

    # Original cell 156, line 460.
    def compute_metrics_from_simulated_hypnograms(
        sim_item,
        wake_state=0,
        rem_state=4,
        nrem_states=(1, 2, 3),
        epoch_sec=30,
        min_nrem_min=0.0,
        min_rem_min=0.0,
        count_after_sleep_onset=True,
        wake_breaks_cycle=True,
        min_unique_states_for_loop=3,
    ):
        """
        对某一个 subject、某一个 condition 的模拟 hypnograms 统计指标。

        sim_item:
            generate_simulated_hypnogram_data 生成的单个元素。
            sim_item["states"] shape = (n_sims, n_steps + 1)
        """
        if sim_item is None:
            return None

        all_states = np.asarray(sim_item["states"], dtype=int)

        sleep_latencies = []
        wake_counts = []
        switch_counts = []
        loop_counts = []
        nrem_rem_cycle_counts = []

        for states in all_states:
            latency, wake_ct, switch_ct = extract_metrics_from_traj(
                states,
                wake_state=wake_state,
            )

            loop_info = count_loops_in_hypnogram(
                states,
                min_unique_states=min_unique_states_for_loop,
            )

            n_cycles = count_nrem_rem_cycles_with_min_duration(
                states,
                wake_state=wake_state,
                rem_state=rem_state,
                nrem_states=nrem_states,
                epoch_sec=epoch_sec,
                min_nrem_min=min_nrem_min,
                min_rem_min=min_rem_min,
                count_after_sleep_onset=count_after_sleep_onset,
                wake_breaks_cycle=wake_breaks_cycle,
            )

            sleep_latencies.append(latency)
            wake_counts.append(wake_ct)
            switch_counts.append(switch_ct)
            loop_counts.append(loop_info["n_loops"])
            nrem_rem_cycle_counts.append(n_cycles)

        sleep_latencies = np.asarray(sleep_latencies, dtype=float)
        wake_counts = np.asarray(wake_counts, dtype=int)
        switch_counts = np.asarray(switch_counts, dtype=int)
        loop_counts = np.asarray(loop_counts, dtype=int)
        nrem_rem_cycle_counts = np.asarray(nrem_rem_cycle_counts, dtype=int)

        min_nrem_epochs = int(np.ceil(min_nrem_min * 60.0 / epoch_sec))
        min_rem_epochs = int(np.ceil(min_rem_min * 60.0 / epoch_sec))

        return {
            "mean_sleep_latency": _safe_nanmean(sleep_latencies),
            "mean_wake_count_after_sleep": float(np.mean(wake_counts)),
            "mean_state_switch_count_after_sleep": float(np.mean(switch_counts)),
            "mean_loop_count": float(np.mean(loop_counts)),
            "mean_nrem_rem_cycle_count": float(np.mean(nrem_rem_cycle_counts)),

            "all_sleep_latencies": sleep_latencies,
            "all_wake_counts": wake_counts,
            "all_switch_counts": switch_counts,
            "all_loop_counts": loop_counts,
            "all_nrem_rem_cycle_counts": nrem_rem_cycle_counts,

            "normalized_T": sim_item.get("normalized_T", None),
            "n_steps_used": sim_item.get("n_steps_used", None),
            "n_sims": sim_item.get("n_sims", len(all_states)),

            "min_nrem_min": float(min_nrem_min),
            "min_rem_min": float(min_rem_min),
            "min_nrem_epochs": int(min_nrem_epochs),
            "min_rem_epochs": int(min_rem_epochs),
            "epoch_sec": int(epoch_sec),
            "wake_breaks_cycle": bool(wake_breaks_cycle),
        }

    # Original cell 156, line 554.
    def compare_results_with_nrem_rem(results_new, results_orig, as_list=True):
        """
        比较 new vs orig 的统计结果。
        每一行对应一个 subject。
        """
        scalar_keys = [
            "mean_sleep_latency",
            "mean_wake_count_after_sleep",
            "mean_state_switch_count_after_sleep",
            "mean_loop_count",
            "mean_nrem_rem_cycle_count",
        ]

        array_keys = [
            "all_sleep_latencies",
            "all_wake_counts",
            "all_switch_counts",
            "all_loop_counts",
            "all_nrem_rem_cycle_counts",
        ]

        matrix_keys = [
            "normalized_T",
        ]

        extra_keys = [
            "n_steps_used",
            "n_sims",
            "min_nrem_min",
            "min_rem_min",
            "min_nrem_epochs",
            "min_rem_epochs",
            "epoch_sec",
            "wake_breaks_cycle",
        ]

        rows = []

        for i, (rn, ro) in enumerate(zip(results_new, results_orig)):
            if rn is None or ro is None:
                continue

            row = {"index": i}

            for key in extra_keys:
                row[key] = rn.get(key, np.nan)

            for key in scalar_keys:
                new_val = rn.get(key, np.nan)
                orig_val = ro.get(key, np.nan)

                row[f"new_{key}"] = new_val
                row[f"orig_{key}"] = orig_val

                if pd.notna(new_val) and pd.notna(orig_val):
                    row[f"diff_{key}"] = new_val - orig_val
                    row[f"reduction_{key}"] = orig_val - new_val

                    if orig_val != 0:
                        row[f"pct_reduction_{key}"] = 100.0 * (orig_val - new_val) / orig_val
                    else:
                        row[f"pct_reduction_{key}"] = np.nan
                else:
                    row[f"diff_{key}"] = np.nan
                    row[f"reduction_{key}"] = np.nan
                    row[f"pct_reduction_{key}"] = np.nan

            for key in array_keys:
                new_val = rn.get(key, None)
                orig_val = ro.get(key, None)

                if new_val is not None:
                    new_val = np.asarray(new_val)
                if orig_val is not None:
                    orig_val = np.asarray(orig_val)

                row[f"new_{key}"] = new_val.tolist() if (as_list and new_val is not None) else new_val
                row[f"orig_{key}"] = orig_val.tolist() if (as_list and orig_val is not None) else orig_val

                if new_val is not None and orig_val is not None and new_val.shape == orig_val.shape:
                    diff_val = new_val - orig_val
                    row[f"diff_{key}"] = diff_val.tolist() if as_list else diff_val
                else:
                    row[f"diff_{key}"] = None

            for key in matrix_keys:
                new_val = rn.get(key, None)
                orig_val = ro.get(key, None)

                if new_val is not None:
                    new_val = np.asarray(new_val)
                if orig_val is not None:
                    orig_val = np.asarray(orig_val)

                row[f"new_{key}"] = new_val.tolist() if (as_list and new_val is not None) else new_val
                row[f"orig_{key}"] = orig_val.tolist() if (as_list and orig_val is not None) else orig_val

                if new_val is not None and orig_val is not None and new_val.shape == orig_val.shape:
                    diff_val = new_val - orig_val
                    row[f"diff_{key}"] = diff_val.tolist() if as_list else diff_val
                else:
                    row[f"diff_{key}"] = None

            rows.append(row)

        return pd.DataFrame(rows)

    # Original cell 156, line 662.
    def analyze_simulated_data_for_one_k(
        sim_data,
        k,
        epoch_sec=30,
        wake_state=0,
        rem_state=4,
        nrem_states=(1, 2, 3),
        count_after_sleep_onset=True,
        wake_breaks_cycle=True,
        min_unique_states_for_loop=3,
        as_list=True,
    ):
        """
        对固定 k 统计模拟结果。

        设置：
            min_rem_min = k
            min_nrem_min = 3 * k
        """
        min_rem_min = float(k)
        min_nrem_min = 3.0 * float(k)

        sim_new = sim_data["new"]
        sim_orig = sim_data["orig"]

        n_subjects = len(sim_new)

        results_new = [None] * n_subjects
        results_orig = [None] * n_subjects
        paired_results = []

        for i in range(n_subjects):
            item_new = sim_new[i]
            item_orig = sim_orig[i]

            if item_new is None or item_orig is None:
                continue

            res_new = compute_metrics_from_simulated_hypnograms(
                item_new,
                wake_state=wake_state,
                rem_state=rem_state,
                nrem_states=nrem_states,
                epoch_sec=epoch_sec,
                min_nrem_min=min_nrem_min,
                min_rem_min=min_rem_min,
                count_after_sleep_onset=count_after_sleep_onset,
                wake_breaks_cycle=wake_breaks_cycle,
                min_unique_states_for_loop=min_unique_states_for_loop,
            )

            res_orig = compute_metrics_from_simulated_hypnograms(
                item_orig,
                wake_state=wake_state,
                rem_state=rem_state,
                nrem_states=nrem_states,
                epoch_sec=epoch_sec,
                min_nrem_min=min_nrem_min,
                min_rem_min=min_rem_min,
                count_after_sleep_onset=count_after_sleep_onset,
                wake_breaks_cycle=wake_breaks_cycle,
                min_unique_states_for_loop=min_unique_states_for_loop,
            )

            results_new[i] = res_new
            results_orig[i] = res_orig

            paired_results.append({
                "index": i,
                "k": float(k),
                "min_nrem_min": min_nrem_min,
                "min_rem_min": min_rem_min,
                "n_steps_used": res_new["n_steps_used"],
                "T_new_metrics": {
                    "mean_sleep_latency": res_new["mean_sleep_latency"],
                    "mean_wake_count_after_sleep": res_new["mean_wake_count_after_sleep"],
                    "mean_state_switch_count_after_sleep": res_new["mean_state_switch_count_after_sleep"],
                    "mean_loop_count": res_new["mean_loop_count"],
                    "mean_nrem_rem_cycle_count": res_new["mean_nrem_rem_cycle_count"],
                },
                "Tmat_metrics": {
                    "mean_sleep_latency": res_orig["mean_sleep_latency"],
                    "mean_wake_count_after_sleep": res_orig["mean_wake_count_after_sleep"],
                    "mean_state_switch_count_after_sleep": res_orig["mean_state_switch_count_after_sleep"],
                    "mean_loop_count": res_orig["mean_loop_count"],
                    "mean_nrem_rem_cycle_count": res_orig["mean_nrem_rem_cycle_count"],
                },
            })

        df_compare = compare_results_with_nrem_rem(
            results_new,
            results_orig,
            as_list=as_list,
        )

        df_compare["k"] = float(k)
        df_compare["min_nrem_min"] = min_nrem_min
        df_compare["min_rem_min"] = min_rem_min

        return {
            "k": float(k),
            "min_nrem_min": min_nrem_min,
            "min_rem_min": min_rem_min,
            "results_new": results_new,
            "results_orig": results_orig,
            "paired_results": paired_results,
            "df_compare": df_compare,
        }

    # Original cell 156, line 772.
    def summarize_df_compare_across_subjects(df_compare):
        """
        对某个 k 的 df_compare 做 across-subject summary。
        """
        metrics = [
            "mean_sleep_latency",
            "mean_wake_count_after_sleep",
            "mean_state_switch_count_after_sleep",
            "mean_loop_count",
            "mean_nrem_rem_cycle_count",
        ]

        if df_compare.empty:
            return pd.DataFrame()

        row = {
            "k": df_compare["k"].iloc[0],
            "min_nrem_min": df_compare["min_nrem_min"].iloc[0],
            "min_rem_min": df_compare["min_rem_min"].iloc[0],
            "n_subjects_paired": len(df_compare),
        }

        for m in metrics:
            new_col = f"new_{m}"
            orig_col = f"orig_{m}"
            diff_col = f"diff_{m}"
            red_col = f"reduction_{m}"
            pct_col = f"pct_reduction_{m}"

            row[f"mean_new_{m}"] = df_compare[new_col].mean()
            row[f"mean_orig_{m}"] = df_compare[orig_col].mean()
            row[f"mean_diff_{m}"] = df_compare[diff_col].mean()
            row[f"mean_reduction_{m}"] = df_compare[red_col].mean()
            row[f"mean_pct_reduction_{m}"] = df_compare[pct_col].mean()

            row[f"median_new_{m}"] = df_compare[new_col].median()
            row[f"median_orig_{m}"] = df_compare[orig_col].median()
            row[f"median_diff_{m}"] = df_compare[diff_col].median()
            row[f"median_pct_reduction_{m}"] = df_compare[pct_col].median()

        return pd.DataFrame([row])

    # Original cell 156, line 815.
    def analyze_simulated_data_over_k(
        sim_data,
        k_values,
        epoch_sec=30,
        wake_state=0,
        rem_state=4,
        nrem_states=(1, 2, 3),
        count_after_sleep_onset=True,
        wake_breaks_cycle=True,
        min_unique_states_for_loop=3,
        as_list=True,
    ):
        """
        Step 2 主函数：对已经生成好的模拟 hypnogram 扫描 k。

        对每个 k：
            min_nrem_min = 3*k
            min_rem_min = k

        返回：
            results_by_k:
                dict, 每个 k 对应完整 results_new / results_orig / df_compare

            df_subject_k:
                所有 k、所有 subject 的比较结果拼接

            df_k_summary:
                每个 k 的 across-subject summary
        """
        k_values = np.asarray(k_values, dtype=float)

        results_by_k = {}
        df_subject_k_list = []
        df_k_summary_list = []

        for k in k_values:
            out_k = analyze_simulated_data_for_one_k(
                sim_data=sim_data,
                k=k,
                epoch_sec=epoch_sec,
                wake_state=wake_state,
                rem_state=rem_state,
                nrem_states=nrem_states,
                count_after_sleep_onset=count_after_sleep_onset,
                wake_breaks_cycle=wake_breaks_cycle,
                min_unique_states_for_loop=min_unique_states_for_loop,
                as_list=as_list,
            )

            results_by_k[float(k)] = out_k

            df_compare_k = out_k["df_compare"]
            df_subject_k_list.append(df_compare_k)

            df_summary_k = summarize_df_compare_across_subjects(df_compare_k)
            df_k_summary_list.append(df_summary_k)

        if len(df_subject_k_list) > 0:
            df_subject_k = pd.concat(df_subject_k_list, axis=0, ignore_index=True)
        else:
            df_subject_k = pd.DataFrame()

        if len(df_k_summary_list) > 0:
            df_k_summary = pd.concat(df_k_summary_list, axis=0, ignore_index=True)
        else:
            df_k_summary = pd.DataFrame()

        return results_by_k, df_subject_k, df_k_summary

    return {'modify_loops_and_reconstruct_Ts': modify_loops_and_reconstruct_Ts, 'generate_simulated_hypnogram_data': generate_simulated_hypnogram_data, 'analyze_simulated_data_over_k': analyze_simulated_data_over_k}

_simulation_helpers = _build_simulation_helpers()
modify_loops_and_reconstruct_Ts = _simulation_helpers['modify_loops_and_reconstruct_Ts']
generate_simulated_hypnogram_data = _simulation_helpers['generate_simulated_hypnogram_data']
analyze_simulated_data_over_k = _simulation_helpers['analyze_simulated_data_over_k']


# ======================================================================
# Definitions from cleaned notebook cell 43 (zero-based)
# ======================================================================

def _build_relaxation_helpers():
    # Original cell 161, line 1.
    import numpy as np

    # Original cell 161, line 2.
    import pandas as pd

    # Original cell 161, line 5.
    def normalize_T_5x5(T, tol=1e-12):
        """
        确保 T 是 5x5 转移矩阵。
        Tij 表示从状态 i 到状态 j 的概率。
        """
        T = np.asarray(T, dtype=float).copy()

        if T.shape != (5, 5):
            raise ValueError(f"T must be 5x5, got shape {T.shape}")

        T[np.abs(T) < tol] = 0.0

        if np.any(T < -tol):
            raise ValueError("T contains negative entries.")

        T = np.clip(T, 0.0, None)

        row_sums = T.sum(axis=1, keepdims=True)
        nonzero_rows = row_sums[:, 0] > tol

        T[nonzero_rows] = T[nonzero_rows] / row_sums[nonzero_rows]

        # 如果某一行全 0，表示该状态没有转移信息，则设为停留在原状态
        zero_rows = ~nonzero_rows
        for i in np.where(zero_rows)[0]:
            T[i, :] = 0.0
            T[i, i] = 1.0

        return T

    # Original cell 161, line 36.
    def normalize_prob_vector_5(P, tol=1e-12):
        """
        确保概率向量是长度为 5 的合法概率分布。
        """
        P = np.asarray(P, dtype=float).copy()

        if P.shape != (5,):
            raise ValueError(f"P must be length 5, got shape {P.shape}")

        P[np.abs(P) < tol] = 0.0

        if np.any(P < -tol):
            raise ValueError("P contains negative entries.")

        P = np.clip(P, 0.0, None)

        s = P.sum()
        if s <= tol:
            raise ValueError("P has zero total probability.")

        P = P / s
        return P

    # Original cell 161, line 60.
    def prob_distance(P, Pss, metric="max_abs"):
        """
        比较当前概率分布 P 和稳态概率分布 Pss 的距离。

        metric:
        - "max_abs": max_i |P_i - Pss_i|，要求每个状态都足够接近
        - "l1": sum_i |P_i - Pss_i|
        - "tv": 0.5 * sum_i |P_i - Pss_i|
        - "l2": sqrt(sum_i (P_i - Pss_i)^2)
        """
        P = np.asarray(P, dtype=float)
        Pss = np.asarray(Pss, dtype=float)

        if metric == "max_abs":
            return float(np.max(np.abs(P - Pss)))
        elif metric == "l1":
            return float(np.sum(np.abs(P - Pss)))
        elif metric == "tv":
            return float(0.5 * np.sum(np.abs(P - Pss)))
        elif metric == "l2":
            return float(np.linalg.norm(P - Pss))
        else:
            raise ValueError("metric must be one of: max_abs, l1, tv, l2")

    # Original cell 161, line 85.
    def evolve_probability_distribution(
        T,
        n_steps,
        init_state=0,
        tol=1e-12,
    ):
        """
        从 P0 = [1,0,0,0,0] 开始，模拟概率分布演化。

        返回：
        Ps.shape = (n_steps + 1, 5)
        Ps[t] 是第 t 个时刻的状态概率分布。
        """
        T = normalize_T_5x5(T, tol=tol)

        if not (0 <= init_state < 5):
            raise ValueError("init_state must be in 0,1,2,3,4")

        Ps = np.zeros((int(n_steps) + 1, 5), dtype=float)
        Ps[0, init_state] = 1.0

        for t in range(int(n_steps)):
            Ps[t + 1] = Ps[t] @ T

        return Ps

    # Original cell 161, line 112.
    def find_steady_time_from_probability_evolution(
        T,
        Pss,
        n_steps,
        init_state=0,
        threshold=0.01,
        consecutive=1,
        metric="max_abs",
        tol=1e-12,
    ):
        """
        判断概率分布演化什么时候到达稳态。

        判据：
        distance(P(t), Pss) <= threshold

        如果 consecutive > 1，则要求连续 consecutive 个时刻都满足，
        返回第一次开始满足的位置。
        """
        Pss = normalize_prob_vector_5(Pss, tol=tol)

        Ps = evolve_probability_distribution(
            T=T,
            n_steps=n_steps,
            init_state=init_state,
            tol=tol,
        )

        distances = np.array([
            prob_distance(Ps[t], Pss, metric=metric)
            for t in range(Ps.shape[0])
        ])

        reached = distances <= threshold

        steady_time = np.nan
        confirm_time = np.nan

        if consecutive <= 1:
            idx = np.where(reached)[0]
            if len(idx) > 0:
                steady_time = int(idx[0])
                confirm_time = int(idx[0])
        else:
            for t in range(len(reached) - consecutive + 1):
                if np.all(reached[t:t + consecutive]):
                    steady_time = int(t)
                    confirm_time = int(t + consecutive - 1)
                    break

        return {
            "reached_steady": not np.isnan(steady_time),
            "steady_time": steady_time,
            "confirm_time": confirm_time,
            "min_distance": float(np.min(distances)),
            "final_distance": float(distances[-1]),
            "distance_series": distances,
            "probability_series": Ps,
            "Pss": Pss,
        }

    # Original cell 161, line 174.
    def steady_time_for_T_list(
        T_list,
        Pss_list,
        n_steps_list,
        label="T",
        init_state=0,
        threshold=0.01,
        consecutive=1,
        metric="max_abs",
        tol=1e-12,
    ):
        """
        对一组 5x5 转移矩阵统计到达稳态的时间。

        T_list: list of 5x5 matrices
        Pss_list: 每个 subject 对应的稳态概率分布，形如 NocPss
        n_steps_list: 每个 subject 模拟多少步，例如 NocTST * 2
        """
        if len(T_list) != len(Pss_list):
            raise ValueError("T_list and Pss_list must have the same length.")

        if len(T_list) != len(n_steps_list):
            raise ValueError("T_list and n_steps_list must have the same length.")

        rows = []

        for i, T in enumerate(T_list):
            if T is None:
                continue

            n_steps_i = n_steps_list[i]

            if np.isnan(n_steps_i):
                continue

            n_steps_i = int(round(n_steps_i))

            if n_steps_i <= 0:
                continue

            info = find_steady_time_from_probability_evolution(
                T=T,
                Pss=Pss_list[i],
                n_steps=n_steps_i,
                init_state=init_state,
                threshold=threshold,
                consecutive=consecutive,
                metric=metric,
                tol=tol,
            )

            rows.append({
                "index": i,
                "matrix_type": label,
                "n_steps": n_steps_i,
                "reached_steady": info["reached_steady"],
                "steady_time": info["steady_time"],
                "confirm_time": info["confirm_time"],
                "min_distance": info["min_distance"],
                "final_distance": info["final_distance"],
                "Pss": info["Pss"],
                "distance_series": info["distance_series"],
                "probability_series": info["probability_series"],
            })

        return pd.DataFrame(rows)

    # Original cell 161, line 242.
    def compare_steady_time_for_Tnew_and_Tmat(
        T_new_list,
        Tmat,
        Pss_list,
        AllSleepLength,
        init_state=0,
        threshold=0.01,
        consecutive=1,
        metric="max_abs",
        tol=1e-12,
    ):
        """
        同时统计 T_new_list 和原始 Tmat 的概率分布到达稳态时间。
        """
        AllSleepLength = np.asarray(AllSleepLength)

        df_new = steady_time_for_T_list(
            T_list=T_new_list,
            Pss_list=Pss_list,
            n_steps_list=AllSleepLength,
            label="new",
            init_state=init_state,
            threshold=threshold,
            consecutive=consecutive,
            metric=metric,
            tol=tol,
        )

        df_orig = steady_time_for_T_list(
            T_list=Tmat,
            Pss_list=Pss_list,
            n_steps_list=AllSleepLength,
            label="orig",
            init_state=init_state,
            threshold=threshold,
            consecutive=consecutive,
            metric=metric,
            tol=tol,
        )

        return pd.concat([df_new, df_orig], ignore_index=True)

    return {'compare_steady_time_for_Tnew_and_Tmat': compare_steady_time_for_Tnew_and_Tmat}

_relaxation_helpers = _build_relaxation_helpers()
compare_steady_time_for_Tnew_and_Tmat = _relaxation_helpers['compare_steady_time_for_Tnew_and_Tmat']


# ======================================================================
# Definitions from cleaned notebook cell 49 (zero-based)
# ======================================================================

"""Cell 1: def. Requires only numpy and matplotlib."""
from pathlib import Path
from collections.abc import Mapping
import json
import csv
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, ConnectionPatch
from matplotlib.path import Path as MplPath
from matplotlib.colors import LinearSegmentedColormap

_STATES = ("WAKE", "N1", "N2", "N3", "REM")
_SHORT = ("W", "N1", "N2", "N3", "R")

def _stage_name(value):
    aliases = {"W":"WAKE", "WAKE":"WAKE", "N1":"N1", "N2":"N2",
               "N3":"N3", "R":"REM", "REM":"REM"}
    key = str(value).strip().upper()
    if key not in aliases:
        raise ValueError(f"Unknown stage label: {value!r}")
    return aliases[key]

def _read_hypnogram(data, stage_map):
    """No epochs are deleted or fabricated."""
    if isinstance(data, Mapping):
        keys = [k for k in ("hypnogram","hypno","stages","stage") if k in data]
        if len(keys) != 1:
            raise ValueError("Pass a 1D hypnogram, or a record with one hypno/stage field.")
        data = data[keys[0]]
    if hasattr(data, "columns"):
        keys = [k for k in ("hypnogram","hypno","stages","stage") if k in data.columns]
        if len(keys) == 1:
            data = data[keys[0]]
    a = np.asarray(data).squeeze()
    if a.ndim != 1 or len(a) < 8:
        raise ValueError(f"Expected at least 8 scored epochs in a 1D array, got {a.shape}.")
    lookup = {k:_stage_name(v) for k,v in stage_map.items()}
    result = []
    for i,v in enumerate(a):
        try:
            name = _stage_name(v) if isinstance(v,str) else lookup[v]
        except (KeyError,ValueError,TypeError):
            raise ValueError(f"Unknown/missing sleep stage at epoch {i}: {v!r}. "
                             "Check stage_map; invalid epochs are not dropped.") from None
        result.append(_STATES.index(name))
    return np.array(result, dtype=int)

def _matrix_stack(data, name, matrix_axes="auto"):
    """Collect every matrix in dict/list/ndarray inputs; preserve all record weights."""
    records = []
    def walk(item, label):
        if isinstance(item, Mapping):
            if not item: raise ValueError(f"{label} is empty.")
            for key,value in item.items(): walk(value, f"{label}[{key!r}]")
            return
        if isinstance(item,(list,tuple)):
            try: a = np.asarray(item,dtype=float)
            except (ValueError,TypeError): a = None
            if a is not None and a.shape == (5,5):
                records.append(a.copy()); return
            if not item: raise ValueError(f"{label} is empty.")
            for i,value in enumerate(item): walk(value,f"{label}[{i}]")
            return
        a = np.asarray(item,dtype=float)
        if a.ndim == 2:
            if a.shape != (5,5): raise ValueError(f"{label}: expected 5x5, got {a.shape}.")
            records.append(a.copy()); return
        if a.ndim < 3: raise ValueError(f"{label}: invalid matrix batch shape {a.shape}.")
        if isinstance(matrix_axes,str):
            if matrix_axes != "auto": raise ValueError("matrix_axes must be 'auto' or an axis pair.")
            candidates = [(k,k+1) for k in range(a.ndim-1) if a.shape[k:k+2] == (5,5)]
            if len(candidates) != 1:
                raise ValueError(f"{label}: ambiguous matrix axes in {a.shape}. "
                                 "Set *_matrix_axes=(-2,-1) for (n,5,5), or (0,1) for (5,5,n).")
            axes = candidates[0]
        else:
            if len(matrix_axes) != 2: raise ValueError("matrix_axes must contain two axes.")
            axes = tuple(int(k)%a.ndim for k in matrix_axes)
            if axes[0] == axes[1] or any(a.shape[k] != 5 for k in axes):
                raise ValueError(f"{label}: selected axes do not define a 5x5 matrix.")
        records.extend(np.moveaxis(a,axes,(-2,-1)).reshape(-1,5,5).copy())
    walk(data,name)
    if not records: raise ValueError(f"{name} has no matrices.")
    out = np.stack(records)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains NaN/Inf. No silent NaN averaging or record exclusion.")
    if np.any(out < 0) or np.any(out > 1+1e-6):
        raise ValueError(f"{name} must contain nonnegative probabilities/flows, not counts.")
    return out

def _runs(hyp):
    starts = np.r_[0,np.flatnonzero(hyp[1:] != hyp[:-1])+1]
    return starts, np.r_[starts[1:],len(hyp)], hyp[starts]

def _nrem_rem_intervals(hyp, epoch_seconds):
    """Canonical thresholds: accumulated >=15 min NREM and a >=5 min REM bout.

    A cycle begins at first sleep, then at the end of the previous accepted REM
    bout, matching the manuscript's accumulated-NREM definition. Short REM bouts
    and intervening Wake do not reset NREM accumulation. Stop indices are exclusive.
    """
    sleeps=np.flatnonzero(hyp!=0)
    if not len(sleeps):return []
    begin=int(sleeps[0]);intervals=[]
    starts,ends,stages=_runs(hyp)
    for start,stop,stage in zip(starts,ends,stages):
        if stage!=4 or (stop-start)*epoch_seconds<300:continue
        if np.count_nonzero(np.isin(hyp[begin:start],[1,2,3]))*epoch_seconds>=900:
            intervals.append((begin,int(stop)));begin=int(stop)
    return intervals


def _n2_n3_exchange_interval(hyp, cycle):
    """Choose the most-switching N2/N3 segment, shade from its first N3.

    The initial N2 dwell is excluded from the shade. The entry N2->N3 at the
    left boundary still belongs to this exchange episode's transition count.
    """
    a,b=cycle;starts,ends,stages=_runs(hyp[a:b]);candidates=[];i=0
    while i<len(stages):
        if stages[i] not in (2,3):i+=1;continue
        j=i+1
        while j<len(stages) and stages[j] in (2,3):j+=1
        changes=j-i-1
        if changes>=3:candidates.append((changes,int(starts[i])+a,int(ends[j-1])+a))
        i=j
    if not candidates:
        raise ValueError('The selected NREM-REM cycle has no consecutive N2/N3 segment '
                         'with at least 3 actual N2<->N3 switches. No interval is fabricated.')
    changes,start,stop=max(candidates,key=lambda z:(z[0],z[2]-z[1],-z[1]))
    first_n3=int(np.flatnonzero(hyp[start:stop]==3)[0])+start
    return first_n3,stop,changes


def _record_matrix(data,key,name,matrix_axes):
    """Dict: exact hypnogram key; sequence/array: zero-based record index."""
    if isinstance(data,Mapping):
        if key not in data:
            raise KeyError(f'{name} has no record key {key!r}; supply matrices keyed to the hypnogram IDs.')
        stack=_matrix_stack(data[key],f'{name}[{key!r}]',matrix_axes)
        if len(stack)!=1:raise ValueError(f'{name}[{key!r}] must represent exactly one record.')
        return stack[0]
    stack=_matrix_stack(data,name,matrix_axes)
    if not isinstance(key,(int,np.integer)) or not 0<=key<len(stack):
        raise IndexError(f'{name}: record index {key!r} is outside 0..{len(stack)-1}.')
    return stack[int(key)]


def _record_collection_index(data,key,name,matrix_axes):
    if not isinstance(data,Mapping):return int(key)
    offset=0
    for record_key,value in data.items():
        if record_key==key:return offset
        offset+=len(_matrix_stack(value,f'{name}[{record_key!r}]',matrix_axes))
    raise KeyError(f'{name} has no record key {key!r}.')

def _timing_fig1(hyp, epoch_seconds, lights_off_epoch, sleep_onset_epoch,
                 sleep_offset_epoch, transition_epoch, meta,
                 awakening_epoch=None, falling_asleep_epoch=None):
    n=len(hyp)
    def index(value,name,allow_end=False):
        if value is None:return None
        if not isinstance(value,(int,np.integer)) or not 0<=value<=(n if allow_end else n-1):
            raise ValueError(f"{name} must be a valid integer epoch index.")
        return int(value)
    off=index(lights_off_epoch,"lights_off_epoch")
    onset=index(sleep_onset_epoch,"sleep_onset_epoch")
    if onset is None:
        after=0 if off is None else off
        candidates=np.flatnonzero(hyp[after:]!=0)
        onset=int(after+candidates[0]) if len(candidates) else None
    if onset is not None and (hyp[onset]==0 or (off is not None and onset<off)):
        raise ValueError("Sleep onset must be a scored sleep epoch at/after lights off.")
    offset=index(sleep_offset_epoch,"sleep_offset_epoch",True)
    offset_source="provided" if offset is not None else "not observed"
    if offset is None:
        sleeps=np.flatnonzero(hyp!=0)
        if len(sleeps) and sleeps[-1]+1<n:
            offset=int(sleeps[-1]+1);offset_source="terminal Wake after last scored sleep"
    if offset is not None and onset is not None and offset<=onset:
        raise ValueError("Sleep offset must occur after sleep onset.")
    trans=index(transition_epoch,"transition_epoch")
    changes=np.flatnonzero(hyp[1:]!=hyp[:-1])+1
    if trans is not None:
        if trans==0 or hyp[trans]==hyp[trans-1]:
            raise ValueError("transition_epoch must point to an actual stage change.")
    else:
        choices=[int(k) for k in changes if hyp[k]!=0 and hyp[k-1]!=0 and k!=onset]
        if not choices:choices=[int(k) for k in changes if k!=onset]
        if choices:
            a,b=meta["hypnocycle"]["start_epoch"],meta["hypnocycle"]["stop_epoch_exclusive"]
            trans=min(choices,key=lambda k:((a<=k<b),abs(k-.55*n)))
    # Awakenings are intermediate sleep->Wake entries, preferentially clustered
    # in the latter half of sleep. Terminal Wake belongs only to sleep offset.
    end=n if offset is None else offset
    intermediate=[int(k) for k in changes if hyp[k]==0 and hyp[k-1]!=0
                  and (onset is None or k>onset) and k<end]
    late_start=int((0 if onset is None else onset)+.5*(end-(0 if onset is None else onset)))
    late=[k for k in intermediate if k>=late_start]
    awaken=index(awakening_epoch,"awakening_epoch")
    if awaken is not None and awaken not in intermediate:
        raise ValueError("awakening_epoch must be an intermediate sleep-to-Wake entry before sleep offset.")
    pool=late if late else intermediate
    window=max(1,int(.15*(end-(0 if onset is None else onset))))
    if awaken is None and pool:
        groups=[[k for k in pool if first<=k<=first+window] for first in pool]
        selected=max(groups,key=lambda group:(len(group),group[-1]))
        awaken=selected[len(selected)//2]
    else:selected=[k for k in pool if awaken is not None and abs(k-awaken)<=window/2]
    if awaken is not None and awaken not in selected:selected.append(awaken)
    selected=sorted(selected)
    after=0 if off is None else off
    first_sleep=np.flatnonzero(hyp[after:]!=0)
    initial=int(after+first_sleep[0]) if len(first_sleep) else None
    falling=index(falling_asleep_epoch,"falling_asleep_epoch")
    if falling is not None:
        if falling==0 or hyp[falling-1]!=0 or hyp[falling]==0 or falling!=initial:
            raise ValueError("falling_asleep_epoch must mark the first actual Wake-to-sleep transition after lights off.")
    else:
        falling=(initial if initial is not None and initial>0 and hyp[initial-1]==0 else None)
    meta["timing"]={"lights_off_epoch":off,"sleep_onset_epoch":onset,
      "sleep_offset_epoch":offset,"sleep_offset_source":offset_source,
      "SOL_seconds":None if off is None or onset is None else float((onset-off)*epoch_seconds),
      "onset_definition":"first scored non-Wake epoch at/after lights off unless explicitly supplied",
      "lights_off_source":"not supplied" if off is None else "explicitly supplied",
      "transition_epoch":trans,"awakening_epoch":awaken,"falling_asleep_epoch":falling,
      "falling_asleep_scope":"first observed Wake-to-sleep transition at/after lights off",
      "awakening_scope":"actual intermediate sleep-to-Wake entries, preferentially a late-sleep cluster; terminal Wake excluded",
      "awakening_epochs":selected,"awakening_is_late_cluster":bool(selected and all(k>=late_start for k in selected))}
    return onset

def _prepare_fig1(hypnogram, NapTmat, NocTmat, NapFS, NocFS, epoch_seconds,
                  stage_map, matrix_order, t_matrix_axes, flow_matrix_axes,
                  sequence_start, sequence_length, hypnocycle_index, nrem_rem_window,
                  nrem_rem_index, example_record_key):
    if not np.isfinite(epoch_seconds) or epoch_seconds <= 0:
        raise ValueError("epoch_seconds must be positive.")
    hyp = _read_hypnogram(hypnogram,stage_map)
    n = len(hyp)
    order = tuple(_stage_name(s) for s in matrix_order)
    if len(order) != 5 or set(order) != set(_STATES):
        raise ValueError("matrix_order must list WAKE,N1,N2,N3,REM exactly once.")
    perm = [order.index(s) for s in _STATES]
    stacks = {}
    for name,data,axes in [("NapTmat",NapTmat,t_matrix_axes),("NocTmat",NocTmat,t_matrix_axes),
                           ("NapFS",NapFS,flow_matrix_axes),("NocFS",NocFS,flow_matrix_axes)]:
        a = _matrix_stack(data,name,axes)
        stacks[name] = a[:,perm,:][:,:,perm]
    # Collect all records for the background stacks; front maps are record-specific.
    all_T = np.concatenate([stacks["NapTmat"],stacks["NocTmat"]],axis=0)
    all_Q = np.concatenate([stacks["NapFS"],stacks["NocFS"]],axis=0)
    T=_record_matrix(NocTmat,example_record_key,'NocTmat',t_matrix_axes)[perm,:][:,perm]
    Q=_record_matrix(NocFS,example_record_key,'NocFS',flow_matrix_axes)[perm,:][:,perm]
    row_sums = all_T.sum(axis=2)
    bad_rows = int(np.count_nonzero((row_sums>1e-12)&(np.abs(row_sums-1)>1e-3)))
    zero_rows = int(np.count_nonzero(row_sums<=1e-12))
    if bad_rows or zero_rows:
        warnings.warn(f"T has {bad_rows} nonstochastic positive rows and {zero_rows} zero rows. "
                      "All supplied values are retained without renormalization.",stacklevel=3)
    if len(all_T) != len(all_Q):
        warnings.warn("T and Q record counts differ; background stacks are independent collections.",stacklevel=3)
    starts,ends,stages = _runs(hyp)
    matches = [r for r in range(len(stages)-3) if np.array_equal(stages[r:r+4],[2,0,1,2])]
    if not matches:
        raise ValueError("No exact N2-WAKE-N1-N2 hypnocycle exists in this hypnogram after "
                         "compressing consecutive repeats. Check stage_map or choose another real recording.")
    if hypnocycle_index is None:
        selected = min(range(len(matches)),
                       key=lambda k:abs((starts[matches[k]]+starts[matches[k]+3])/2-n/2))
    else:
        if not isinstance(hypnocycle_index,(int,np.integer)) or not 0 <= hypnocycle_index < len(matches):
            raise ValueError(f"hypnocycle_index must be in 0..{len(matches)-1}.")
        selected = int(hypnocycle_index)
    r = matches[selected]
    hs,hr,he = int(starts[r]),int(starts[r+3]),int(starts[r+3]+1)
    assert np.array_equal(_runs(hyp[hs:he])[2],[2,0,1,2])
    rng=np.random.default_rng(2026)
    illustrative_sequences=[[_SHORT[int(k)] for k in rng.integers(0,5,5)] for _ in range(3)]
    cycles=_nrem_rem_intervals(hyp,epoch_seconds)
    if not isinstance(nrem_rem_index,(int,np.integer)) or not 0<=nrem_rem_index<len(cycles):
        raise ValueError(f'Requested NREM-REM cycle #{nrem_rem_index+1}, but only {len(cycles)} qualifying cycles exist.')
    cycle=cycles[int(nrem_rem_index)]
    if nrem_rem_window is not None and tuple(nrem_rem_window)!=cycle:
        raise ValueError('nrem_rem_window must match the selected chronological NREM-REM cycle. '
                         'Use nrem_rem_index=1 for the second cycle.')
    n23_start,n23_stop,n23_changes=_n2_n3_exchange_interval(hyp,cycle)
    hours = np.arange(n+1)*epoch_seconds/3600
    sleeps = np.flatnonzero(hyp != 0)
    sol = int(sleeps[0]) if len(sleeps) else None
    wakes = np.flatnonzero((hyp[1:]==0)&(hyp[:-1]!=0))+1
    wake = int(min(wakes,key=lambda k:abs(k-.65*n))) if len(wakes) else None
    counts = {name:len(a) for name,a in stacks.items()}
    counts.update(T_total=len(all_T),Q_total=len(all_Q))
    meta = {"epoch_seconds":float(epoch_seconds),"n_epochs":n,"duration_h":float(hours[-1]),
      "matrix_order":list(_STATES),"input_matrix_order":list(order),"matrix_counts":counts,
      "matrix_display":"Single-record foreground maps; no mean matrices computed or displayed",
      "foreground_pooled_indices":{"T":len(stacks["NapTmat"])+_record_collection_index(NocTmat,example_record_key,"NocTmat",t_matrix_axes),
                                   "Q":len(stacks["NapFS"])+_record_collection_index(NocFS,example_record_key,"NocFS",flow_matrix_axes)},
      "example_record_key":int(example_record_key) if isinstance(example_record_key,(int,np.integer)) else str(example_record_key),
      "example_matrix_sources":{"T":f"NocTmat[{example_record_key!r}]","Q":f"NocFS[{example_record_key!r}]"},
      "nrem_rem_cycles_epochs":[list(c) for c in cycles],"nrem_rem_index_0based":int(nrem_rem_index),
      "nrem_rem_definition":"Accumulated >=15 min NREM and continuous >=5 min REM; manuscript definition",
      "n2_n3_interval_epochs":[n23_start,n23_stop],"n2_n3_transition_count":n23_changes,
      "n2_n3_shade_start":"first entry into N3 within the selected exchange episode",
      "n2_n3_transition_count_definition":"Includes N2->N3 at the shade left boundary, if present",
      "zero_T_rows_retained":zero_rows,"nonstochastic_T_rows_retained":bad_rows,
      "sequence":{"type":"illustrative random examples, not excerpts from panel a", "seed":2026,
                  "examples":illustrative_sequences,"displayed_count":3,"collection_continues":True},
      "hypnocycle":{"match_count":len(matches),"selected_match":selected,"start_epoch":hs,
                   "return_epoch":hr,"stop_epoch_exclusive":he,"time_h":[float(hours[hs]),float(hours[he])],
                   "compressed_labels":["N2","WAKE","N1","N2"]},
      "nrem_rem_interval_epochs":list(cycle) if cycle is not None else None,
      "network_type":"fixed conceptual schematics, independent of the supplied matrices",
      "flow_decomposition_computed":False}
    return hyp,hours,T,Q,meta,cycle,sol,wake,all_T,all_Q

def _draw_fig1(hyp, boundaries, T, Q, meta, cycle, sol, wake, all_T, all_Q, font_size, fig_width_mm, fig_height_mm):
    """Observed examples and pooled matrices, followed by fixed schematics."""
    BLUE,ORANGE,INK = '#0072B2','#D55E00','#263844'
    GRAY,LIGHT,PURPLE,TEAL = '#71808A','#D5DDE2','#7B4190','#006F8C'
    SOL_COLOR='#E69F00'
    W,H=180.,200.  # Logical layout coordinates; output size is set independently.
    fs=lambda value:value*font_size/7.
    fig=plt.figure(figsize=(fig_width_mm/25.4,H/25.4),facecolor='white',dpi=150)
    canvas=fig.add_axes([0,0,1,1],label='layout')
    canvas.set(xlim=(0,W),ylim=(0,H));canvas.axis('off')
    def text(x,y,s,size=7,weight='normal',ha='left',color=INK):
        return canvas.text(x,y,s,fontsize=fs(size),fontweight=weight,ha=ha,va='center',color=color)
    def axes(x,y,w,h,label):
        return fig.add_axes([x/W,y/H,w/W,h/H],label=label)
    def arrow(ax,a,b,color=INK,style='-|>',lw=.9,scale=7,shrink=0,rad=0):
        patch=FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=scale,color=color,
            linewidth=lw,shrinkA=shrink,shrinkB=shrink,connectionstyle=f'arc3,rad={rad}',zorder=3)
        ax.add_patch(patch);return patch
    def flow_path(vertices,color=INK,lw=1.1,scale=10):
        path=MplPath(vertices,[MplPath.MOVETO]+[MplPath.LINETO]*(len(vertices)-1))
        patch=FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=scale,
            linewidth=lw,color=color,capstyle='round',joinstyle='round',zorder=4)
        canvas.add_patch(patch);return patch
    def title(letter,x,y,label):
        return text(x,y,letter,8,'bold'),text(x+7,y,label,7.2,'bold')
    def bracket(ax,xa,xb,y,color,lw=.7):
        ax.plot([xa,xa,xb,xb],[y-.8,y,y,y-.8],color=color,lw=lw,solid_capstyle='butt')
    total_h=boundaries[-1]
    to_x=lambda h:15+155*h/total_h
    seq,hc=meta['sequence'],meta['hypnocycle']
    hs,he=hc['start_epoch'],hc['stop_epoch_exclusive']
    sy=np.array([4,2,1,0,3])
    timing=meta['timing'];off=timing['lights_off_epoch'];onset=timing['sleep_onset_epoch']
    offset=timing['sleep_offset_epoch'];falling=timing['falling_asleep_epoch']
    a_title=(text(3,H-5,'a',8,'bold'),text(92.5,H-5,'Example hypnogram',7.2,'bold',ha='center'))
    ah=axes(15,128,155,30,'a')
    n23a,n23b=meta['n2_n3_interval_epochs']
    CYCLE_COLOR='#4C956C'
    for first,last,color,alpha,name in (
        (cycle[0],cycle[1],CYCLE_COLOR,.20,'NREM_REM_cycle'),
        (n23a,n23b,CYCLE_COLOR,.44,'N2_N3_transitions'),
        (off,onset,SOL_COLOR,.36,'SOL'),
        (hs,he,PURPLE,.26,'hypnocycle')):
        if first is not None and last is not None and last>first:
            shade=ah.axvspan(boundaries[first],boundaries[last],color=color,alpha=alpha,lw=0,zorder=0)
            shade.set_gid(name)
    ah.step(boundaries,np.r_[sy[hyp],sy[hyp[-1]]],where='post',color=INK,lw=.85,zorder=3)
    # Match each highlighted trace to its interval shade; smaller excerpts are
    # drawn last so their colors remain visible if intervals overlap.
    for first,last,color in ((off,onset,SOL_COLOR),(cycle[0],cycle[1],CYCLE_COLOR),
                             (hs,he,PURPLE)):
        if first is None or last is None or last<=first:continue
        ah.step(boundaries[first:last+1],np.r_[sy[hyp[first:last]],sy[hyp[last-1]]],
                where='post',color=color,lw=1.4,zorder=4)
    step=1 if total_h<=12 else 2 if total_h<=24 else max(1,np.ceil(total_h/10))
    ah.set(xlim=(0,total_h),ylim=(-.25,4.35),xticks=np.arange(0,total_h+1e-9,step),
           yticks=[4,3,2,1,0],yticklabels=['Wake','REM','N1','N2','N3'])
    ah.tick_params(length=2,width=.5,pad=2,labelsize=fs(5.8))
    ah.set_xlabel('Time from recording start (h)',labelpad=3,loc='right')
    ah.set_ylabel('Sleep stage',labelpad=4)
    for event in (off,onset,offset):
        if event is not None:ah.axvline(boundaries[event],color=GRAY,lw=.45,ls=(0,(2,2)),zorder=2)
    # Every annotation remains above the full trace. There are no zoom axes.
    events=[('Hypnocycle: N2 → Wake → N1 → N2',hs,he,PURPLE,'interval'),
            (f"NREM–REM cycle {meta['nrem_rem_index_0based']+1}",cycle[0],cycle[1],CYCLE_COLOR,'interval'),
            ('N2 ↔ N3 transitions',n23a,n23b,CYCLE_COLOR,'interval')]
    if off is not None:events.append(('Lights off',off,off,INK,'reference'))
    else:events.append(('Lights off not supplied',0,0,GRAY,'status'))
    if onset is not None:
        label='Sleep onset'
        events.append((label,onset,onset,INK,'stagechange'))
    if off is not None and onset is not None:
        events.append((f"SOL ({timing['SOL_seconds']/60:g} min)",off,onset,SOL_COLOR,'interval'))
    if offset is not None:
        label='Sleep offset'
        events.append((label,offset,offset,INK,'stagechange' if wake==offset else 'reference'))
    elif wake is None:events.append(('Sleep offset not observed',len(hyp),len(hyp),GRAY,'status'))
    else:events.append(('Sleep offset not observed',len(hyp),len(hyp),GRAY,'status'))
    awakening_points=timing['awakening_epochs']
    if awakening_points:
        events.append(('Awakenings' if len(awakening_points)>1 else 'Awakening',
                       awakening_points[0],awakening_points[-1],INK,'awakening_cluster'))
    trans=timing['transition_epoch']
    if trans is not None:events.append(('Transition',trans,trans,INK,'stagechange'))
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    occupied=[[] for _ in events];placed=[]
    fixed_centers={};edge=15.
    for event_name in ('Lights off','Sleep onset'):
        if any(e[0]==event_name for e in events):
            probe=text(0,166,event_name,6.1)
            width=probe.get_window_extent(renderer).width/fig.dpi*25.4*W/fig_width_mm;probe.remove()
            fixed_centers[event_name]=edge+width/2;edge+=width+5
    order=sorted(events,key=lambda e:(0 if e[0] in fixed_centers else 1,(e[1]+e[2])/2))
    for label,first,stop,color,kind in order:
        xa,xb=to_x(boundaries[first]),to_x(boundaries[stop]);cx=(xa+xb)/2
        ha='right' if label.startswith('Sleep offset') else 'center'
        if label in fixed_centers:cx=fixed_centers[label]
        if kind=='awakening_cluster':cx-=8.0
        probe=text(cx,166,label,6.1,ha=ha)
        tw=probe.get_window_extent(renderer).width/fig.dpi*25.4*W/fig_width_mm;probe.remove()
        x0=cx-tw if ha=='right' else cx if ha=='left' else cx-tw/2;x1=x0+tw
        if x0<4:cx+=4-x0;x1+=4-x0;x0=4
        if x1>W-3:cx-=x1-(W-3);x0-=x1-(W-3);x1=W-3
        span=(x0-1,x1+1)
        for lane in range(len(occupied)):
            if all(span[1]<old[0] or old[1]<span[0] for old in occupied[lane]):
                occupied[lane].append(span);break
        placed.append((label,first,color,kind,xa,xb,cx,ha,lane))
    H=float(166+6.5*max(e[-1] for e in placed)+13)
    output_height=H*fig_width_mm/W if fig_height_mm is None else fig_height_mm
    fig.set_size_inches(fig_width_mm/25.4,output_height/25.4,forward=True);canvas.set_ylim(0,H)
    ah.set_position([15/W,128/H,155/W,30/H])
    for artist in a_title:artist.set_y(H-5)
    for label,first,color,kind,xa,xb,cx,ha,lane in placed:
        y=166+6.5*lane
        lab=text(cx,y,label,6.1,ha=ha,color=color);lab.set_gid('a-label-'+label)
        if kind=='interval':bracket(canvas,xa,xb,y-3.2,color);continue
        if kind=='status':continue
        if label in fixed_centers:
            canvas.plot([cx,cx,xa],[y-2.1,162.4,160.9],color=color,lw=.55)
        if kind=='awakening_cluster':
            if len(awakening_points)>1:
                bracket(canvas,xa,xb,y-3.2,color)
                canvas.plot([cx,(xa+xb)/2],[y-2.1,y-3.2],color=color,lw=.55)
            for k in awakening_points:
                fig.add_artist(ConnectionPatch(xyA=(to_x(boundaries[k]),161.2),coordsA=canvas.transData,
                    xyB=(boundaries[k],4.04),coordsB=ah.transData,arrowstyle='-|>',
                    mutation_scale=5.5,linewidth=.6,color=color,clip_on=False,zorder=20))
            continue
        if label=='Sleep offset' and 0<first<len(hyp):
            # Connect the text itself to the actual terminal sleep->Wake transition.
            fig.canvas.draw()
            box=lab.get_window_extent(fig.canvas.get_renderer())
            anchor=canvas.transData.inverted().transform([box.x1,box.y0])
            value=(sy[hyp[first-1]]+sy[hyp[first]])/2
            patch=ConnectionPatch(xyA=tuple(anchor),coordsA=canvas.transData,
                xyB=(boundaries[first],value),coordsB=ah.transData,arrowstyle='-|>',
                mutation_scale=5.5,linewidth=.6,color=color,clip_on=False,zorder=20)
            patch.set_gid('a-sleep-offset-connected-arrow');fig.add_artist(patch)
            meta['sleep_offset_arrow_label_anchor']=tuple(float(v) for v in anchor)
            continue
        if kind=='stagechange' and 0<first<len(hyp):
            value=(sy[hyp[first-1]]+sy[hyp[first]])/2
            fig.add_artist(ConnectionPatch(xyA=(xa,161.5),coordsA=canvas.transData,
                xyB=(boundaries[first],value),coordsB=ah.transData,arrowstyle='-|>',
                mutation_scale=5.5,linewidth=.6,color=color,clip_on=False,zorder=20))
        else:arrow(canvas,(xa,161.5),(xa,158.5),color,lw=.6,scale=5.5)
    meta['annotation_style']='above-trace labels; Lights off and Sleep onset share the lowest annotation row; actual late intermediate awakenings marked separately from terminal sleep offset'
    meta['cycle_shading']={'color':CYCLE_COLOR,'cycle_alpha':.20,'N2_N3_alpha':.44}

    text(3,116,'b',8,'bold')
    text(28,103,'Stage sequences',6.3,'bold',ha='center')
    for row,tokens in enumerate(seq['examples']):
        yy=92-8*row
        for i,label in enumerate(tokens):
            xx=12+i*6.3
            canvas.add_patch(Rectangle((xx,yy-2.6),6.3,5.2,facecolor='#E7F3F6',edgecolor='#91B9C4',lw=.45))
            text(xx+3.15,yy,label,5.8,ha='center',color=TEAL)
        text(8.5,yy,'…',7,ha='center',color=TEAL)
        text(47,yy,'…',7,ha='center',color=TEAL)
    text(28,68,r'$\vdots$',9,ha='center',color=TEAL).set_gid('sequence-collection-ellipsis')
    # Start at the actual rendered centre of the Sleep stage y-axis label.
    # Then route left, down, right into the middle sequence row.
    ah.yaxis.label.set_bbox(dict(facecolor='white',edgecolor='none',pad=.3))
    meta['a_to_b_flow_source']='centre of rendered Sleep stage ylabel; left, down, right to sequence collection'
    arrow(canvas,(49.2,84),(55.8,84),lw=1.05,scale=9)
    cmap=LinearSegmentedColormap.from_list('fig1_blue',['#F7FAFC','#C9E0EE','#72ADD0','#1D5F8A'])
    def matrix_stack(x,data,stack,label,heading):
        indices=np.unique(np.linspace(0,len(stack)-1,min(3,len(stack)),dtype=int)).tolist()
        # The exact selected-record maps are foreground; backgrounds are other
        # individual records sampled by the previous first/middle/last rule.
        indices=[i for i in indices if i!=meta['foreground_pooled_indices'][label.upper()]]
        sample_indices=[indices[0],indices[-1]] if len(indices)>1 else indices
        vmax=max(float(stack.max()),float(data.max())) if np.any(stack>0) or np.any(data>0) else 1.
        for layer,index in reversed(list(enumerate(sample_indices,1))):
            back=axes(x+2.2*layer,64+2.2*layer,32,32,f'{label}_background_{index}')
            back.pcolormesh(np.arange(6)-.5,np.arange(6)-.5,stack[index],cmap=cmap,vmin=0,vmax=vmax,rasterized=False)
            back.set(xlim=(-.5,4.5),ylim=(4.5,-.5));back.set_aspect('equal');back.set_xticks([]);back.set_yticks([])
            for spine in back.spines.values():
                spine.set_visible(True);spine.set_linewidth(.7);spine.set_edgecolor('#4E626F');spine.set_zorder(10)
        ax=axes(x,64,32,32,label)
        ax.pcolormesh(np.arange(6)-.5,np.arange(6)-.5,data,cmap=cmap,vmin=0,vmax=vmax,rasterized=False)
        ax.set(xlim=(-.5,4.5),ylim=(4.5,-.5),xticks=range(5),yticks=range(5),xticklabels=_SHORT,yticklabels=_SHORT)
        ax.set_aspect('equal');ax.tick_params(length=0,pad=2,labelsize=fs(5.8))
        for spine in ax.spines.values():
            spine.set_visible(True);spine.set_linewidth(.85);spine.set_edgecolor('#4E626F');spine.set_zorder(10)
        text(x+16+(5. if label=='t' else 2.),105,heading,6.4,'bold',ha='center').set_gid(label+'_matrix_heading')
        # A depth-axis arrow follows the actual foreground-to-background offset.
        # Its label names the third dimension: different hypnograms.
        if sample_indices:
            depth_arrow=arrow(canvas,(x-6,92),(x+2,100),GRAY,lw=.7,scale=6)
            depth_arrow.set_gid(label+'_hypnogram_dimension_arrow')
            p0,p1=canvas.transData.transform([(x-6,92),(x+2,100)])
            angle=float(np.degrees(np.arctan2(*(p1-p0)[::-1])))
            depth_label=text(x-9,104,'Different\nhypnograms ($r$)',5.2,ha='center',color=GRAY)
            depth_label.set_rotation(angle)
            depth_label.set_rotation_mode('anchor')
            depth_label.set_gid(label+'_hypnogram_dimension_label')
            dots=canvas.scatter([x+2.8,x+3.5,x+4.2],[100.8,101.5,102.2],
                                s=1.1*(font_size/7.)**2,color=GRAY,zorder=5)
            dots.set_gid(label+'_hypnogram_dimension_ellipsis')
        cax=axes(x+36.9,64,1.4,32,label+'_colorbar')
        cb=fig.colorbar(ax.collections[0],cax=cax,ticks=[0,vmax]);cb.outline.set_visible(False)
        cb.solids.set_rasterized(False);cb.ax.tick_params(length=1.6,width=.45,pad=1,labelsize=fs(5.5))
        cb.set_ticklabels(['0',f'{vmax:.2g}'])
        return ax,sample_indices
    at,ti=matrix_stack(62,T,all_T,'t',r'Transition probability ($T$)')
    aq,qi=matrix_stack(134.5,Q,all_Q,'q',r'Probability flow ($Q$)')
    arrow(canvas,(104,85.2),(127.5,85.2),lw=.95,scale=8)
    text(115.75,80.5,r'$Q^{(r)}=\mathrm{diag}(P^{(r)})T^{(r)}$',5.7,ha='center',color=INK).set_gid('record-flow-formula')
    meta['displayed_record_matrix_indices']={'T_background':ti,'Q_background':qi}
    meta['foreground_record_key']=meta['example_record_key']
    meta['stack_dimension']='Different hypnograms, indexed by r, from the foreground toward the upper-right background'
    meta['hypnogram_index_symbol']='r'
    # Both networks share the same geometry: the two directed triangles sit on
    # opposite sides of their shared N1--N2 edge, without an X-shaped crossing.
    def network(x,letter,kind):
        ax=axes(x,16,79,26,letter);ax.set(xlim=(0,79),ylim=(0,26));ax.axis('off')
        text(x,45,letter,8,'bold')
        pos={0:(9,13),1:(37,23),2:(37,3),3:(70,3),4:(65,13)}
        if kind=='eq':
            edge=arrow(ax,pos[0],pos[1],BLUE,'<->',.95,6.5,3.8,rad=0)
            edge.set_gid('c-wake-n1-straight-reversible')
            for i,j in [(1,2),(2,4),(2,3),(1,4),(0,2)]:
                arrow(ax,pos[i],pos[j],BLUE,'<->',.95,6.5,3.8)
            # Wake--REM connects directly, as requested.
            arrow(ax,pos[0],pos[4],BLUE,'<->',.9,6.5,3.8,rad=0)
            ax.text(16,25.4,'Awakening (REM → Wake)',fontsize=fs(5.5),ha='center',va='center',color=BLUE)
        else:
            ax.text(19,25.4,'Falling asleep',fontsize=fs(5.5),ha='center',va='center',color=PURPLE)
            arrow(ax,(36.1,23),(36.1,3),PURPLE,'-|>',1.05,7,3.8)
            arrow(ax,pos[0],pos[1],PURPLE,'-|>',1.05,7,3.8)
            arrow(ax,pos[2],pos[0],PURPLE,'-|>',1.05,7,3.8)
            arrow(ax,(37.9,23),(37.9,3),ORANGE,'-|>',1.05,7,3.8)
            arrow(ax,pos[2],pos[4],ORANGE,'-|>',1.05,7,3.8)
            arrow(ax,pos[4],pos[1],ORANGE,'-|>',1.05,7,3.8)
        for k,(px,py) in pos.items():
            inactive=kind!='eq' and k==3
            ax.add_patch(Circle((px,py),1.25,facecolor='white',edgecolor=LIGHT if inactive else INK,lw=.7,zorder=5))
            dx,dy={0:(-2,0),1:(3.5,1.3),2:(0,-3.3),3:(0,-3.3),4:(3,0)}[k]
            ha='right' if k==0 else 'left' if k in (1,4) else 'center'
            ax.text(px+dx,py+dy,'Wake' if k==0 else _STATES[k],fontsize=fs(6.4),
                    ha=ha,va='center',color=GRAY if inactive else INK,zorder=6,clip_on=False)
        return ax
    ac=network(5,'c','eq')
    ad=network(96,'d','flux')
    text(44.5,10.8,'Equilibrium',6.4,ha='center',color=BLUE)
    text(135.5,10.8,'Non-equilibrium',6.4,ha='center',color=ORANGE)
    arrow(canvas,(98,5.8),(103,5.8),ORANGE,lw=1,scale=6)
    text(105,5.8,'REM-containing flux loop: N1 → N2 → REM → N1',5.5,color=ORANGE)
    arrow(canvas,(98,2.1),(103,2.1),PURPLE,lw=1,scale=6)
    text(105,2.1,'Hypnocycle: Wake → N1 → N2 → Wake',5.5,color=PURPLE)
    meta['network_event_labels']={'c_awakening':['REM','WAKE'],'d_falling_asleep':['WAKE','N1']}
    meta['schematic_loops']={'REM_containing_flux_loop':['N1','N2','REM','N1'],
                             'hypnocycle':['WAKE','N1','N2','WAKE']}
    meta['equilibrium_schematic_pairs']=[['WAKE','N1'],['N1','N2'],['N2','REM'],
        ['REM','WAKE'],['N2','N3'],['N1','REM'],['WAKE','N2']]
    # Decomposition starts at the selected-record realized flow, and visibly branches.
    flow_path([(150.5,56.8),(150.5,54.5),(90,54.5),(90,51.8)],lw=1.1,scale=10)
    canvas.plot([44.5,90,135.5],[51.8,51.8,51.8],color=INK,lw=1.1)
    arrow(canvas,(44.5,51.8),(44.5,45.5),BLUE,lw=1.15,scale=10)
    arrow(canvas,(135.5,51.8),(135.5,45.5),ORANGE,lw=1.15,scale=10)
    text(43,49.5,r'Symmetric part: $Q^{\mathrm{sym}}$',6.,ha='right',color=BLUE).set_gid('symmetric-flow-branch-label')
    text(134,49.5,r'Asymmetric part: $J=Q-Q^{\mathrm{sym}}$',6.,ha='right',color=ORANGE).set_gid('asymmetric-flow-branch-label')
    # Keep all page content inside the canvas; reserve bottom space for the footer.
    footer=[a for a in canvas.texts if a.get_position()[1]<0]
    for artist in footer:artist.set_y(1.5)
    fig.canvas.draw()
    # Derive the route from the active renderer on every PNG/PDF/SVG draw.
    # Font metrics differ slightly between backends; a frozen pixel coordinate
    # would drift away from the ylabel centre after export.
    class _YLabelFlow(mpl.artist.Artist):
        def draw(self,renderer):
            box=ah.yaxis.label.get_window_extent(renderer)
            centre=canvas.transData.inverted().transform([(box.x0+box.x1)/2,(box.y0+box.y1)/2])
            # Attach at the left side of the label's mid-height, keeping ink clear.
            start=canvas.transData.inverted().transform([box.x0-1.5,(box.y0+box.y1)/2])
            meta['a_to_b_label_centre']=tuple(centre)
            vertices=[tuple(start),(1.,float(centre[1])),(1.,84.),(6.1,84.)]
            meta['a_to_b_flow_vertices']=vertices
            path=MplPath(vertices,[MplPath.MOVETO,MplPath.LINETO,MplPath.LINETO,MplPath.LINETO])
            patch=FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=10,
                linewidth=1.1,color=INK,capstyle='round',joinstyle='round',transform=canvas.transData)
            patch.set_figure(fig);patch.set_gid('a-to-b-flow');patch.draw(renderer)
            self.stale=False
    connector=_YLabelFlow();connector.set_zorder(30);fig.add_artist(connector)
    fig.canvas.draw()
    physical_width,physical_height=fig.get_size_inches()*25.4
    for a,b in ((at,aq),(ac,ad)):
        ra=np.array(a.get_position().bounds)*[physical_width,physical_height,physical_width,physical_height]
        rb=np.array(b.get_position().bounds)*[physical_width,physical_height,physical_width,physical_height]
        if not np.allclose(ra[[1,2,3]],rb[[1,2,3]],atol=1.5*25.4/72):
            raise RuntimeError('Comparable panels are not aligned.')
    meta['figure_size_mm']=[float(physical_width),float(physical_height)]
    meta['layout_reference_size_mm']=[W,H]
    meta['font_size_pt']=float(font_size)
    meta['font_size_policy']='All text scales from font_size; original size hierarchy retained'
    return {'figure':fig,'axes':{'hypnogram':ah,
            'T':at,'Q':aq,'equilibrium':ac,'non_equilibrium':ad}}


def plot_sleep_fig1(example_hypnogram, NapTmat, NocTmat, NapFS, NocFS, *,
    font_size=7., fig_width_mm=180., fig_height_mm=None,
    epoch_seconds=30, stage_map=None, matrix_order=("WAKE","N1","N2","N3","REM"),
    t_matrix_axes="auto", flow_matrix_axes="auto", sequence_start=None,
    sequence_length=6, hypnocycle_index=None, nrem_rem_window=None,
    nrem_rem_index=1, example_record_key=30,
    lights_off_epoch=None, sleep_onset_epoch=None, sleep_offset_epoch=None,
    transition_epoch=None, awakening_epoch=None, falling_asleep_epoch=None,
    output_dir="Fig1_output", basename="Fig1", save_formats=("pdf","svg","png"),
    dpi=600, export_source_data=True, _qa_callback=None):
    """Draw the agreed Figure 1 using actual inputs, with c/d as fixed schematics.

    font_size: one base size in pt; all text scales together, preserving hierarchy.
    At font_size=7, the original design has 5.5-8 pt text. This is not a minimum.
    fig_width_mm/fig_height_mm: exact canvas size in mm; height=None preserves
    the automatically computed layout aspect ratio at the requested width.
    Dimensions and text size are independent; a smaller canvas does not shrink fonts.
    Very large fonts on a small canvas may require more space; recheck the export.
    example_hypnogram: pass NocHypDict[30]; one label per epoch.
    stage_map: numeric hypnogram code -> stage name; default 0..4 = WAKE,N1,N2,N3,REM.
    matrix_order: row/column order of ALL supplied matrices.
    Matrix collections may be dicts, lists or ndarray batches. For (n,5,5) use
    *_matrix_axes=(-2,-1); for (5,5,n) use (0,1). Auto rejects ambiguous (5,5,5).
    example_record_key=30 selects the front NocTmat/NocFS matrices corresponding to
    NocHypDict[30]. Dict inputs require the exact same key; arrays/lists use the
    zero-based record index after resolving matrix axes. No mean is computed.
    NaN/Inf and invalid probabilities raise errors; no silent exclusions or normalization.
    sequence_start and sequence_length are legacy keywords, unused in this version.
    b displays three seeded random schematic stage sequences with collection ellipses.
    lights_off_epoch: explicitly supplied 0-based epoch; None does not guess lights off.
    sleep_onset_epoch: None selects first scored sleep at/after lights off (or record start).
    sleep_offset_epoch: None infers the terminal Wake boundary after the final sleep epoch.
    If recording ends asleep, offset is unobserved and is not equated with record end.
    transition_epoch: None selects a real stage change; an explicit value is validated.
    SOL is measured from the supplied lights_off_epoch to sleep onset.
    falling_asleep_epoch marks the FIRST observed Wake-to-sleep transition at/after
    lights off, never a later return to sleep. Awakening marks actual intermediate entries into Wake,
    preferentially clustered in late sleep; terminal Wake is Sleep offset. None selects these true boundaries automatically. An ending
    that is still scored sleep has no inferred sleep offset. Manual awakening values must refer to intermediate Wake entries.
    All panel-a event annotations and the extracted-cycle detail are above the trace.
    c includes straight WAKE<->N1 and WAKE<->REM; Awakening refers to REM->Wake.
    d labels Falling asleep on Wake->N1.
    d illustrates the REM-containing flux loop N1->N2->REM->N1 and WAKE->N1->N2->WAKE.
    In b, the foreground maps are the selected nocturnal record, with larger
    per-record background maps retaining the first/middle/last sampling rule.
    All timing labels and interval brackets are above the main trace; no insets.
    The a-panel example remains the actual extracted N2->WAKE->N1->N2 interval;
    the d-panel hypnocycle is the same directed topology with a different start node.
    hypnocycle_index: chronological index among exact compressed N2-WAKE-N1-N2 matches;
    None chooses the match nearest the recording midpoint. No exact match -> ValueError.
    Hypnocycle ends after the first epoch of the returned N2; its dwell times are omitted
    only in the conceptual network, never in the displayed hypnogram.
    nrem_rem_index=1 means the second chronological NREM-REM cycle. Detection
    uses accumulated >=15 min NREM followed by a continuous >=5 min REM bout.
    nrem_rem_window, if supplied, must match that detected selected interval.
    A darker shade marks its consecutive N2/N3 segment with the most actual
    N2<->N3 switches (at least 3), starting the shade at its first N3 entry.
    The initial N2 dwell is not shaded; the entry transition itself is counted.
    Missing cycles or switches raise clear errors.
    output_dir=None disables saving; otherwise saves editable SVG/PDF and PNG.
    Returns figure, axes, example_T, example_Q, metadata, paths.
    No flow decomposition is calculated: networks c/d are fixed conceptual diagrams.
    """
    for key,value in (("font_size",font_size),("fig_width_mm",fig_width_mm)):
        if not np.isscalar(value) or not np.isfinite(value) or value<=0:
            raise ValueError(f"{key} must be a positive finite number.")
    if fig_height_mm is not None and (not np.isscalar(fig_height_mm) or not np.isfinite(fig_height_mm) or fig_height_mm<=0):
        raise ValueError("fig_height_mm must be None or a positive finite number.")
    if stage_map is None:stage_map=dict(enumerate(_STATES))
    hyp,hours,T,Q,meta,cycle,sol,wake,all_T,all_Q = _prepare_fig1(
      example_hypnogram,NapTmat,NocTmat,NapFS,NocFS,epoch_seconds,stage_map,matrix_order,
      t_matrix_axes,flow_matrix_axes,sequence_start,sequence_length,hypnocycle_index,nrem_rem_window,
      nrem_rem_index,example_record_key)
    sol=_timing_fig1(hyp,epoch_seconds,lights_off_epoch,sleep_onset_epoch,
                     sleep_offset_epoch,transition_epoch,meta,
                     awakening_epoch=awakening_epoch,falling_asleep_epoch=falling_asleep_epoch)
    wake=meta["timing"]["awakening_epoch"]
    style={"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
      "font.size":font_size,"axes.labelsize":6.5*font_size/7,
      "xtick.labelsize":5.8*font_size/7,"ytick.labelsize":6*font_size/7,
      "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42,
      "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.55,
      "text.color":"#263844","axes.labelcolor":"#263844","xtick.color":"#4E5961",
      "ytick.color":"#4E5961","savefig.facecolor":"white","legend.frameon":False,
      "mathtext.fontset":"dejavusans","mathtext.default":"regular"}
    with mpl.rc_context(style):
        result=_draw_fig1(hyp,hours,T,Q,meta,cycle,sol,wake,all_T,all_Q,font_size,fig_width_mm,fig_height_mm)
        result.update(example_T=T,example_Q=Q,metadata=meta,paths={})
        if _qa_callback is not None:_qa_callback(result)
        if output_dir is not None:
            folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
            if not basename or Path(basename).name != basename:
                raise ValueError("basename must be a filename stem without directories.")
            for fmt in save_formats:
                fmt=str(fmt).lower().lstrip(".")
                if fmt not in ("pdf","svg","png"):raise ValueError("Use pdf, svg and/or png.")
                path=folder/f"{basename}.{fmt}"
                result["figure"].savefig(path,dpi=dpi,facecolor="white")
                result["paths"][fmt]=str(path.resolve())
            if export_source_data:
                for name,data in (("example_T",T),("example_Q",Q)):
                    with (folder/f"{basename}_{name}.csv").open("w",newline="",encoding="utf-8") as f:
                        writer=csv.writer(f);writer.writerow(["From / To"]+list(_STATES))
                        writer.writerows([label]+row.tolist() for label,row in zip(_STATES,data))
                with (folder/f"{basename}_selected_epochs.csv").open("w",newline="",encoding="utf-8") as f:
                    writer=csv.writer(f);writer.writerow(["selection","epoch_0based","start_h","stop_h","stage"])
                    for name,part in (("hypnocycle",meta["hypnocycle"]),):
                        writer.writerows([name,k,hours[k],hours[k+1],_STATES[hyp[k]]]
                          for k in range(part["start_epoch"],part["stop_epoch_exclusive"]))
                (folder/f"{basename}_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


# ======================================================================
# Definitions from cleaned notebook cell 52 (zero-based)
# ======================================================================

# Cell 1: def. Dependencies: numpy, pandas, scipy, matplotlib.
from pathlib import Path
from collections import Counter
from collections.abc import Mapping
import ast
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, norm, t as student_t


def _fig2_force_engine():
    """Only the supplied force analyzer and its actual dependency closure.

    Local scope avoids overwriting the user's other analysis functions.
    The supplied force-selection and weighting rules are retained.
    """
    STATE_NAME_MAP = {
        0: "Wake",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM",
    }

    def to_1d_hypnogram_force(hypnogram):
        h0 = np.asarray(hypnogram)
        h = np.squeeze(h0)
        if h.ndim == 1:
            return h.astype(int)
        if h.ndim == 2 and 1 in h.shape:
            return h.reshape(-1).astype(int)
        raise ValueError(
            f"Invalid hypnogram shape: original={h0.shape}, squeezed={h.shape}"
        )

    def iter_hypnograms_force(data):
        if isinstance(data, dict):
            items = sorted(data.items(), key=lambda kv: kv[0])
            return [(i, k, v) for i, (k, v) in enumerate(items)]
        arr = np.asarray(data)
        if arr.ndim == 1:
            return [(0, 0, data)]
        if arr.ndim == 2 and arr.dtype != object:
            return [(i, i, arr[i]) for i in range(arr.shape[0])]
        return [(i, i, hyp) for i, hyp in enumerate(data)]

    def find_runs_force(x):
        x = np.asarray(x)
        if len(x) == 0:
            return (
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )
        change = np.where(np.diff(x) != 0)[0] + 1
        starts = np.r_[0, change]
        ends = np.r_[change - 1, len(x) - 1]
        states = x[starts]
        return states.astype(int), starts.astype(int), ends.astype(int)

    def compress_consecutive_force(seq):
        seq = list(seq)
        if len(seq) == 0:
            return []
        out = [int(seq[0])]
        for x in seq[1:]:
            x = int(x)
            if x != out[-1]:
                out.append(x)
        return out

    def structure_to_code_force(seq):
        if seq is None:
            return ""
        if isinstance(seq, str):
            return (
                seq.replace(" ", "")
                   .replace(",", "")
                   .replace("-", "")
                   .replace("_", "")
                   .replace("→", "")
            )
        return "".join(str(int(x)) for x in seq)

    def structure_to_tuple_force(seq):
        if seq is None:
            return tuple()
        if isinstance(seq, str):
            s = structure_to_code_force(seq)
            return tuple(int(ch) for ch in s if ch.isdigit())
        return tuple(int(x) for x in seq)

    def structure_to_name_force(seq, state_name_map=None, arrow="→"):
        if state_name_map is None:
            state_name_map = STATE_NAME_MAP
        return arrow.join(state_name_map.get(int(x), str(x)) for x in seq)

    def close_loop_force(loop):
        loop = structure_to_tuple_force(loop)
        if len(loop) == 0:
            return tuple()
        if loop[0] != loop[-1]:
            loop = loop + (loop[0],)
        return loop

    def canonical_rotation_loop_force(loop):
        loop = close_loop_force(loop)
        if len(loop) <= 1:
            return loop
        core = loop[:-1]
        rotations = []
        for k in range(len(core)):
            r = core[k:] + core[:k]
            rotations.append(r + (r[0],))
        return sorted(rotations, key=structure_to_code_force)[0]

    def get_key_for_dict_force(hyp_index, subject, d):
        if subject in d:
            return subject
        if hyp_index in d:
            return hyp_index
        return None

    def get_flux_key_from_row_force(row, AllCycle):
        hyp_index = int(row["hyp_index"])
        subject = row.get("subject", hyp_index)
        if subject in AllCycle:
            return subject
        if hyp_index in AllCycle:
            return hyp_index
        return None

    def get_strength_list_for_hypnogram(hyp_index, subject, NocAllStrength):
        key = get_key_for_dict_force(hyp_index, subject, NocAllStrength)
        if key is None:
            return []
        return list(NocAllStrength.get(key, []))

    def get_sym_matrix_for_hypnogram(hyp_index, subject, NocFlux_sym_list):
        if isinstance(NocFlux_sym_list, dict):
            if subject in NocFlux_sym_list:
                return np.asarray(NocFlux_sym_list[subject], dtype=float)
            if hyp_index in NocFlux_sym_list:
                return np.asarray(NocFlux_sym_list[hyp_index], dtype=float)
            return np.zeros((5, 5), dtype=float)
        return np.asarray(NocFlux_sym_list[hyp_index], dtype=float)

    def extract_raw_nremrem_cycles_for_force(
        data,
        epoch_sec=30,
        wake_state=0,
        rem_state=4,
        nrem_states=(1, 2, 3),
        min_nrem_min=15,
        min_rem_min=5,
        include_post_rem_state=True,
        include_wake_in_structure=False,
    ):
        """
        Extract raw NREM-REM cycles. No reversible transitions are removed.
        """
        all_rows = []
        min_nrem_epochs = int(np.ceil(min_nrem_min * 60 / epoch_sec))
        min_rem_epochs = int(np.ceil(min_rem_min * 60 / epoch_sec))
        for hyp_index, subject, hyp in iter_hypnograms_force(data):
            h = to_1d_hypnogram_force(hyp)
            sleep_states = set(nrem_states) | {rem_state}
            sleep_onset_candidates = np.where(np.isin(h, list(sleep_states)))[0]
            if len(sleep_onset_candidates) == 0:
                continue
            sleep_onset = int(sleep_onset_candidates[0])
            run_states, run_starts, run_ends = find_runs_force(h)
            rem_runs = []
            for s, st, ed in zip(run_states, run_starts, run_ends):
                if int(s) == rem_state:
                    rem_runs.append((int(st), int(ed)))
            cycle_start = sleep_onset
            cycle_index = 0
            for rem_start, rem_end in rem_runs:
                if rem_end < cycle_start:
                    continue
                pre_rem_segment = h[cycle_start:rem_start]
                nrem_count = int(np.sum(np.isin(pre_rem_segment, nrem_states)))
                rem_count = int(rem_end - rem_start + 1)
                if nrem_count >= min_nrem_epochs and rem_count >= min_rem_epochs:
                    segment_start = int(cycle_start)
                    segment_end = int(rem_end)
                    if include_post_rem_state and rem_end + 1 < len(h):
                        structure_end = int(rem_end + 1)
                    else:
                        structure_end = int(rem_end)
                    segment = h[segment_start:structure_end + 1]
                    if include_wake_in_structure:
                        structure = compress_consecutive_force(segment)
                    else:
                        segment_no_wake = [
                            x for x in segment if int(x) != wake_state
                        ]
                        structure = compress_consecutive_force(segment_no_wake)
                    structure = tuple(structure)
                    duration_epochs = int(segment_end - segment_start + 1)
                    duration_min = duration_epochs * epoch_sec / 60.0
                    all_rows.append({
                        "hyp_index": int(hyp_index),
                        "subject": subject,
                        "cycle_index": int(cycle_index),
                        "cycle_start_epoch": int(segment_start),
                        "cycle_end_epoch": int(segment_end),
                        "structure_end_epoch": int(structure_end),
                        "rem_start_epoch": int(rem_start),
                        "rem_end_epoch": int(rem_end),
                        "nrem_epochs_before_rem": int(nrem_count),
                        "rem_epochs": int(rem_count),
                        "duration_epochs": int(duration_epochs),
                        "duration_min": float(duration_min),
                        "structure": structure,
                        "structure_code": structure_to_code_force(structure),
                        "structure_name": structure_to_name_force(structure),
                    })
                    cycle_index += 1
                    cycle_start = rem_end + 1
        return pd.DataFrame(all_rows)

    def all_directed_subpaths_from_closed_loop_force(loop):
        loop = close_loop_force(loop)
        if len(loop) <= 1:
            return []
        core = loop[:-1]
        m = len(core)
        subpaths = []
        for edge_len in range(1, m + 1):
            for start in range(m):
                path = []
                for k in range(edge_len + 1):
                    path.append(core[(start + k) % m])
                subpaths.append(tuple(path))
        return subpaths

    def build_flux_loop_segment_catalog_with_strength(
        flux_loops,
        flux_strengths,
    ):
        """
        Build directed segment catalog from one hypnogram's flux loops.
        Each support has:
            loop_index
            loop_strength
            segment
        """
        catalog = {}
        flux_strengths = list(flux_strengths)
        for loop_index, loop in enumerate(flux_loops):
            loop_closed = close_loop_force(loop)
            if len(loop_closed) <= 1:
                continue
            loop_canonical = canonical_rotation_loop_force(loop_closed)
            loop_code = structure_to_code_force(loop_canonical)
            loop_name = structure_to_name_force(loop_canonical)
            loop_n_edges = len(loop_canonical) - 1
            if loop_index < len(flux_strengths):
                loop_strength = float(flux_strengths[loop_index])
            else:
                loop_strength = 0.0
            subpaths = all_directed_subpaths_from_closed_loop_force(loop_canonical)
            for segment in subpaths:
                used_edges = len(segment) - 1
                if used_edges <= 0:
                    continue
                info = {
                    "loop_index": int(loop_index),
                    "loop": loop_canonical,
                    "loop_code": loop_code,
                    "loop_name": loop_name,
                    "loop_n_edges": int(loop_n_edges),
                    "loop_strength": float(loop_strength),
                    "segment": segment,
                    "segment_code": structure_to_code_force(segment),
                    "segment_name": structure_to_name_force(segment),
                    "used_edges": int(used_edges),
                }
                catalog.setdefault(segment, []).append(info)
        for segment in catalog:
            catalog[segment] = sorted(
                catalog[segment],
                key=lambda x: (
                    x["loop_strength"],
                    x["used_edges"],
                    -x["loop_n_edges"],
                    x["loop_code"],
                ),
                reverse=True,
            )
        return catalog

    def decompose_cycle_by_flux_loop_segments_force(
        target_structure,
        catalog,
        min_segment_edges=1,
        allow_loop_reuse=True,
    ):
        """
        Partial decomposition is allowed.
        It maximizes:
            1. number of covered edges
            2. fewer segments
            3. larger total selected loop strength
        """
        target = tuple(int(x) for x in target_structure)
        if len(target) <= 1:
            return [], tuple(), 0.0
        target_edges = [(target[i], target[i + 1]) for i in range(len(target) - 1)]
        n_target_edges = len(target_edges)
        n_states = len(target)
        memo = {}
        def better(a, b):
            if a is None:
                return b
            if b is None:
                return a
            key_a = (a[0], a[1], a[2])
            key_b = (b[0], b[1], b[2])
            if key_b > key_a:
                return b
            return a
        def solve(pos, used_loop_indices):
            if pos >= n_states - 1:
                return (0, 0, 0.0, [])
            if allow_loop_reuse:
                key = (pos, ())
            else:
                key = (pos, tuple(sorted(used_loop_indices)))
            if key in memo:
                return memo[key]
            best = None
            skipped = solve(pos + 1, used_loop_indices)
            best = better(best, skipped)
            for next_pos in range(pos + 1, n_states):
                segment = tuple(target[pos:next_pos + 1])
                segment_edges = len(segment) - 1
                if segment_edges < min_segment_edges:
                    continue
                if segment not in catalog:
                    continue
                for support in catalog[segment]:
                    loop_index = int(support["loop_index"])
                    if (not allow_loop_reuse) and (loop_index in used_loop_indices):
                        continue
                    if allow_loop_reuse:
                        new_used = used_loop_indices
                    else:
                        new_used = set(used_loop_indices)
                        new_used.add(loop_index)
                        new_used = frozenset(new_used)
                    rest = solve(next_pos, new_used)
                    if rest is None:
                        continue
                    piece = dict(support)
                    piece["target_start_state_index"] = pos
                    piece["target_end_state_index"] = next_pos
                    piece["target_start_edge_index"] = pos
                    piece["target_end_edge_index"] = next_pos - 1
                    candidate_pieces = [piece] + rest[3]
                    candidate = (
                        segment_edges + rest[0],
                        -len(candidate_pieces),
                        support["loop_strength"] + rest[2],
                        candidate_pieces,
                    )
                    best = better(best, candidate)
            memo[key] = best
            return best
        solution = solve(0, frozenset())
        if solution is None:
            pieces = []
        else:
            pieces = solution[3]
        covered_edge_indices = set()
        for p in pieces:
            for eidx in range(
                int(p["target_start_edge_index"]),
                int(p["target_end_edge_index"]) + 1,
            ):
                covered_edge_indices.add(eidx)
        missing_edges = tuple(
            edge for idx, edge in enumerate(target_edges)
            if idx not in covered_edge_indices
        )
        coverage_fraction = (
            len(covered_edge_indices) / n_target_edges
            if n_target_edges > 0 else 0.0
        )
        return pieces, missing_edges, coverage_fraction

    def analyze_nremrem_loop_flux_plus_sym_force_by_hypnogram(
        data,
        AllCycle,
        AllStrength,
        Flux_sym_list,
        temporal_cycle_occurrence_df=None,
        epoch_sec=30,
        wake_state=0,
        rem_state=4,
        nrem_states=(1, 2, 3),
        min_nrem_min=15,
        min_rem_min=5,
        include_post_rem_state=True,
        include_wake_in_structure=False,
        min_segment_edges=1,
        allow_loop_reuse=True,
        count_each_loop_once_per_cycle=True,
        sum_unique_missing_edges=False,
        multiply_loop_flux_by_n_edges=False,
    ):
        """
        For each raw NREM-REM cycle:
            force_estimate =
                sum of selected loop flux strengths
                +
                sum of symmetric flux values for uncovered directed edges.
        If multiply_loop_flux_by_n_edges=True:
            each loop contribution becomes:
                loop_strength * number_of_edges_in_loop
        Important:
            loop flux is not weighted by used-edge fraction.
            If count_each_loop_once_per_cycle=True, each loop contributes once per NREM-REM cycle,
            even if used multiple times.
        """
        df_raw_occ = extract_raw_nremrem_cycles_for_force(
            data=data,
            epoch_sec=epoch_sec,
            wake_state=wake_state,
            rem_state=rem_state,
            nrem_states=nrem_states,
            min_nrem_min=min_nrem_min,
            min_rem_min=min_rem_min,
            include_post_rem_state=include_post_rem_state,
            include_wake_in_structure=include_wake_in_structure,
        )
        if len(df_raw_occ) == 0:
            raise ValueError("No NREM-REM cycles were found.")
        catalog_cache = {}
        occurrence_rows = []
        loop_piece_rows = []
        sym_edge_rows = []
        for _, row in df_raw_occ.iterrows():
            row_dict = row.to_dict()
            hyp_index = int(row["hyp_index"])
            subject = row.get("subject", hyp_index)
            cycle_index = int(row.get("cycle_index", -1))
            flux_key = get_flux_key_from_row_force(row, AllCycle)
            target_structure = tuple(int(x) for x in row["structure"])
            target_code = structure_to_code_force(target_structure)
            target_name = structure_to_name_force(target_structure)
            if flux_key is None:
                pieces = []
                missing_edges = tuple(
                    (target_structure[i], target_structure[i + 1])
                    for i in range(len(target_structure) - 1)
                )
                coverage_fraction = 0.0
            else:
                if flux_key not in catalog_cache:
                    flux_loops = AllCycle.get(flux_key, [])
                    flux_strengths = get_strength_list_for_hypnogram(
                        hyp_index=hyp_index,
                        subject=subject,
                        NocAllStrength=AllStrength,
                    )
                    catalog_cache[flux_key] = build_flux_loop_segment_catalog_with_strength(
                        flux_loops=flux_loops,
                        flux_strengths=flux_strengths,
                    )
                catalog = catalog_cache[flux_key]
                pieces, missing_edges, coverage_fraction = (
                    decompose_cycle_by_flux_loop_segments_force(
                        target_structure=target_structure,
                        catalog=catalog,
                        min_segment_edges=min_segment_edges,
                        allow_loop_reuse=allow_loop_reuse,
                    )
                )
            if count_each_loop_once_per_cycle:
                used_loop_indices = sorted(set(int(p["loop_index"]) for p in pieces))
                loop_strength_sum = 0.0
                loop_strength_edge_weighted_sum = 0.0
                loop_codes = []
                loop_names = []
                loop_strengths_used = []
                loop_n_edges_used = []
                loop_contributions_used = []
                for loop_idx in used_loop_indices:
                    selected = [p for p in pieces if int(p["loop_index"]) == loop_idx]
                    if len(selected) == 0:
                        continue
                    p0 = selected[0]
                    strength = float(p0["loop_strength"])
                    loop_n_edges = int(p0["loop_n_edges"])
                    if multiply_loop_flux_by_n_edges:
                        contribution = strength * loop_n_edges
                    else:
                        contribution = strength
                    loop_strength_sum += contribution
                    loop_strength_edge_weighted_sum += strength * loop_n_edges
                    loop_codes.append(p0["loop_code"])
                    loop_names.append(p0["loop_name"])
                    loop_strengths_used.append(strength)
                    loop_n_edges_used.append(loop_n_edges)
                    loop_contributions_used.append(contribution)
            else:
                loop_strength_sum = 0.0
                loop_strength_edge_weighted_sum = 0.0
                loop_codes = []
                loop_names = []
                loop_strengths_used = []
                loop_n_edges_used = []
                loop_contributions_used = []
                for p in pieces:
                    strength = float(p["loop_strength"])
                    loop_n_edges = int(p["loop_n_edges"])
                    if multiply_loop_flux_by_n_edges:
                        contribution = strength * loop_n_edges
                    else:
                        contribution = strength
                    loop_strength_sum += contribution
                    loop_strength_edge_weighted_sum += strength * loop_n_edges
                    loop_codes.append(p["loop_code"])
                    loop_names.append(p["loop_name"])
                    loop_strengths_used.append(strength)
                    loop_n_edges_used.append(loop_n_edges)
                    loop_contributions_used.append(contribution)
            sym_matrix = get_sym_matrix_for_hypnogram(
                hyp_index=hyp_index,
                subject=subject,
                NocFlux_sym_list=Flux_sym_list,
            )
            if sum_unique_missing_edges:
                missing_edges_for_sum = tuple(sorted(set(missing_edges)))
            else:
                missing_edges_for_sum = tuple(missing_edges)
            sym_uncovered_edge_sum = 0.0
            sym_edge_codes = []
            sym_edge_names = []
            sym_edge_values = []
            for edge in missing_edges_for_sum:
                a, b = int(edge[0]), int(edge[1])
                if (
                    sym_matrix.ndim == 2
                    and a < sym_matrix.shape[0]
                    and b < sym_matrix.shape[1]
                ):
                    val = float(sym_matrix[a, b])
                else:
                    val = 0.0
                if np.isnan(val):
                    val = 0.0
                sym_uncovered_edge_sum += val
                edge_code = f"{a}{b}"
                edge_name = (
                    f"{STATE_NAME_MAP.get(a, str(a))}"
                    f"→{STATE_NAME_MAP.get(b, str(b))}"
                )
                sym_edge_codes.append(edge_code)
                sym_edge_names.append(edge_name)
                sym_edge_values.append(val)
                sym_edge_rows.append({
                    "hyp_index": hyp_index,
                    "subject": subject,
                    "flux_key": flux_key,
                    "cycle_index": cycle_index,
                    "structure_code": target_code,
                    "structure_name": target_name,
                    "uncovered_edge": edge,
                    "uncovered_edge_code": edge_code,
                    "uncovered_edge_name": edge_name,
                    "sym_flux_value": val,
                })
            total_force = loop_strength_sum + sym_uncovered_edge_sum
            for p in pieces:
                strength = float(p["loop_strength"])
                loop_n_edges = int(p["loop_n_edges"])
                if multiply_loop_flux_by_n_edges:
                    contribution = strength * loop_n_edges
                else:
                    contribution = strength
                loop_piece_rows.append({
                    "hyp_index": hyp_index,
                    "subject": subject,
                    "flux_key": flux_key,
                    "cycle_index": cycle_index,
                    "structure_code": target_code,
                    "structure_name": target_name,
                    "loop_index": int(p["loop_index"]),
                    "loop_code": p["loop_code"],
                    "loop_name": p["loop_name"],
                    "loop_strength": strength,
                    "loop_n_edges": loop_n_edges,
                    "loop_strength_times_n_edges": strength * loop_n_edges,
                    "loop_contribution_used": contribution,
                    "segment_code": p["segment_code"],
                    "segment_name": p["segment_name"],
                    "used_edges": int(p["used_edges"]),
                })
            occurrence_rows.append({
                **row_dict,
                "flux_key": flux_key,
                "target_structure_code": target_code,
                "target_structure_name": target_name,
                "loop_segment_coverage_fraction": float(coverage_fraction),
                "missing_edges_code": ";".join(f"{a}{b}" for a, b in missing_edges),
                "n_loop_pieces_used": int(len(pieces)),
                "n_unique_loops_used": int(len(set(int(p["loop_index"]) for p in pieces))),
                "loop_strength_sum": float(loop_strength_sum),
                "loop_strength_times_n_edges_sum": float(loop_strength_edge_weighted_sum),
                "sym_uncovered_edge_sum": float(sym_uncovered_edge_sum),
                "loop_plus_sym_force": float(total_force),
                "used_loop_codes": " + ".join(loop_codes),
                "used_loop_names": " + ".join(loop_names),
                "used_loop_strengths": " + ".join(f"{x:.8g}" for x in loop_strengths_used),
                "used_loop_n_edges": " + ".join(str(x) for x in loop_n_edges_used),
                "used_loop_contributions": " + ".join(f"{x:.8g}" for x in loop_contributions_used),
                "sym_uncovered_edge_codes": ";".join(sym_edge_codes),
                "sym_uncovered_edge_names": ";".join(sym_edge_names),
                "sym_uncovered_edge_values": ";".join(f"{x:.8g}" for x in sym_edge_values),
                "decomposition_code": " + ".join(p["segment_code"] for p in pieces),
            })
        df_occ_force = pd.DataFrame(occurrence_rows)
        df_loop_piece_force = pd.DataFrame(loop_piece_rows)
        df_sym_edge_force = pd.DataFrame(sym_edge_rows)
        numeric_cols = [
            "loop_segment_coverage_fraction",
            "n_loop_pieces_used",
            "n_unique_loops_used",
            "loop_strength_sum",
            "loop_strength_times_n_edges_sum",
            "sym_uncovered_edge_sum",
            "loop_plus_sym_force",
        ]
        for col in numeric_cols:
            if col in df_occ_force.columns:
                df_occ_force[col] = df_occ_force[col].fillna(0.0)
        hyp_rows = []
        for hyp_index, subject, _ in iter_hypnograms_force(data):
            df_i = df_occ_force[
                df_occ_force["hyp_index"].astype(int) == int(hyp_index)
            ].copy()
            if len(df_i) == 0:
                hyp_rows.append({
                    "hyp_index": int(hyp_index),
                    "subject": subject,
                    "n_nremrem_cycles": 0,
                    "mean_loop_strength_sum": 0.0,
                    "mean_loop_strength_times_n_edges_sum": 0.0,
                    "mean_sym_uncovered_edge_sum": 0.0,
                    "mean_loop_plus_sym_force": 0.0,
                    "sum_loop_strength_sum": 0.0,
                    "sum_loop_strength_times_n_edges_sum": 0.0,
                    "sum_sym_uncovered_edge_sum": 0.0,
                    "sum_loop_plus_sym_force": 0.0,
                    "mean_loop_segment_coverage_fraction": 0.0,
                    "fully_covered_cycle_fraction": 0.0,
                })
                continue
            hyp_rows.append({
                "hyp_index": int(hyp_index),
                "subject": subject,
                "n_nremrem_cycles": int(len(df_i)),
                "mean_loop_strength_sum": float(df_i["loop_strength_sum"].mean()),
                "mean_loop_strength_times_n_edges_sum": float(
                    df_i["loop_strength_times_n_edges_sum"].mean()
                ),
                "mean_sym_uncovered_edge_sum": float(df_i["sym_uncovered_edge_sum"].mean()),
                "mean_loop_plus_sym_force": float(df_i["loop_plus_sym_force"].mean()),
                "sum_loop_strength_sum": float(df_i["loop_strength_sum"].sum()),
                "sum_loop_strength_times_n_edges_sum": float(
                    df_i["loop_strength_times_n_edges_sum"].sum()
                ),
                "sum_sym_uncovered_edge_sum": float(df_i["sym_uncovered_edge_sum"].sum()),
                "sum_loop_plus_sym_force": float(df_i["loop_plus_sym_force"].sum()),
                "mean_loop_segment_coverage_fraction": float(
                    df_i["loop_segment_coverage_fraction"].mean()
                ),
                "fully_covered_cycle_fraction": float(
                    (df_i["loop_segment_coverage_fraction"] == 1.0).mean()
                ),
            })
        df_hyp_force = pd.DataFrame(hyp_rows)
        for col in df_hyp_force.columns:
            if pd.api.types.is_numeric_dtype(df_hyp_force[col]):
                df_hyp_force[col] = df_hyp_force[col].fillna(0.0)
        force_mean_by_hypnogram = df_hyp_force["mean_loop_plus_sym_force"].to_numpy()
        force_sum_by_hypnogram = df_hyp_force["sum_loop_plus_sym_force"].to_numpy()
        loop_flux_mean_by_hypnogram = df_hyp_force["mean_loop_strength_sum"].to_numpy()
        loop_flux_sum_by_hypnogram = df_hyp_force["sum_loop_strength_sum"].to_numpy()
        loop_flux_times_n_edges_mean_by_hypnogram = (
            df_hyp_force["mean_loop_strength_times_n_edges_sum"].to_numpy()
        )
        loop_flux_times_n_edges_sum_by_hypnogram = (
            df_hyp_force["sum_loop_strength_times_n_edges_sum"].to_numpy()
        )
        sym_flux_mean_by_hypnogram = df_hyp_force["mean_sym_uncovered_edge_sum"].to_numpy()
        sym_flux_sum_by_hypnogram = df_hyp_force["sum_sym_uncovered_edge_sum"].to_numpy()
        assert np.isnan(force_mean_by_hypnogram).sum() == 0
        assert np.isnan(force_sum_by_hypnogram).sum() == 0
        return {
            "df_raw_nremrem_occurrence": df_raw_occ,
            "df_occurrence_force": df_occ_force,
            "df_loop_piece_force": df_loop_piece_force,
            "df_sym_edge_force": df_sym_edge_force,
            "df_hypnogram_force": df_hyp_force,
            "force_mean_by_hypnogram": force_mean_by_hypnogram,
            "force_sum_by_hypnogram": force_sum_by_hypnogram,
            "loop_flux_mean_by_hypnogram": loop_flux_mean_by_hypnogram,
            "loop_flux_sum_by_hypnogram": loop_flux_sum_by_hypnogram,
            "loop_flux_times_n_edges_mean_by_hypnogram": (
                loop_flux_times_n_edges_mean_by_hypnogram
            ),
            "loop_flux_times_n_edges_sum_by_hypnogram": (
                loop_flux_times_n_edges_sum_by_hypnogram
            ),
            "sym_flux_mean_by_hypnogram": sym_flux_mean_by_hypnogram,
            "sym_flux_sum_by_hypnogram": sym_flux_sum_by_hypnogram,
        }
    return {"analyze":analyze_nremrem_loop_flux_plus_sym_force_by_hypnogram,
            "extract":extract_raw_nremrem_cycles_for_force,"iter":iter_hypnograms_force}

_FIG2_ENGINE=_fig2_force_engine()


def _fig2_vector(value,name):
    a=np.asarray(value,dtype=float).squeeze()
    if a.ndim!=1:raise ValueError(f'{name} must be 1D, got {a.shape}.')
    return a.copy()


def _fig2_pearson(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);n=len(x)
    if n<3 or np.ptp(x)==0 or np.ptp(y)==0:return np.nan,np.nan,np.nan,np.nan,n
    r,p=map(float,pearsonr(x,y))
    if n<=3:lo=hi=np.nan
    elif abs(r)>=1-1e-14:lo=hi=r
    else:
        z=np.arctanh(r);d=norm.ppf(.975)/np.sqrt(n-3);lo,hi=np.tanh([z-d,z+d])
    return r,p,float(lo),float(hi),n


def _fig2_bootstrap_difference(eq,lf,y,groups,n_boot,seed):
    """Paired recording-level bootstrap, stratified by nap/nocturnal cohort."""
    if n_boot<=0:return np.nan,np.nan,0
    rng=np.random.default_rng(seed);samples=[]
    cohorts=[np.flatnonzero(groups==g) for g in np.unique(groups)]
    for _ in range(int(n_boot)):
        ind=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in cohorts])
        a,b,z=eq[ind],lf[ind],y[ind]
        if min(np.ptp(a),np.ptp(b),np.ptp(z))<=0:continue
        samples.append(abs(np.corrcoef(b,z)[0,1])-abs(np.corrcoef(a,z)[0,1]))
    if len(samples)<max(20,.5*n_boot):return np.nan,np.nan,len(samples)
    lo,hi=np.quantile(samples,[.025,.975]);return float(lo),float(hi),len(samples)


def prepare_fig2_associations(cohorts,*,nap_sol_epoch_seconds=30):
    """Validate and align each panel independently, never across all panels.

    a/b/d require their own outcome, TST, EQ and LF arrays to match.
    c requires only SOL, InitialSymFlux and InitialFlux to match. It may use
    a different record subset. Within a panel, array positions must refer to
    the same recordings. Position IDs are panel-local, not cross-panel IDs.
    TST and nocturnal SOL are minutes; nap SOL uses the supplied epoch unit.
    """
    rows=[];exclusions=[]
    specifications=[
        ('a','TransNum','AllSymFlux','TotalFlux',True),
        ('b','ToWakeNum','AllSymFluxToWake','WakeCycleStrength',True),
        ('c','SOL','InitialSymFlux','InitialFlux',False),
        ('d','loop_counts','AllSymFlux','TotalFlux',True),
    ]
    for cohort in ('Nocturnal','Nap'):
        c=cohorts[cohort]
        for panel,target_key,eq_key,lf_key,is_rate in specifications:
            keys=[target_key,eq_key,lf_key]+(['TST'] if is_rate else [])
            d={k:_fig2_vector(c[k],f'{cohort}.{k}') for k in keys}
            sizes={k:len(v) for k,v in d.items()}
            if len(set(sizes.values()))!=1:
                raise ValueError(f'{cohort} panel {panel} array lengths differ: {sizes}. '
                                 'Only arrays within this panel must match; other panels may have different sample sizes.')
            if is_rate:
                if np.any(np.isfinite(d['TST'])&(d['TST']<=0)):
                    raise ValueError(f'{cohort} panel {panel}: TST must be positive minutes.')
                with np.errstate(divide='ignore',invalid='ignore'):
                    y=d[target_key]/d['TST']
            else:
                y=d[target_key]*(nap_sol_epoch_seconds/60 if cohort=='Nap' else 1)
            eq=d[eq_key];lf=d[lf_key]
            keep=np.isfinite(y)&np.isfinite(eq)&np.isfinite(lf)
            excluded=int(np.count_nonzero(~keep))
            exclusions.append({'panel':panel,'cohort':cohort,'n_input':len(y),'n_excluded':excluded})
            if excluded:
                warnings.warn(f'{panel}/{cohort}: {excluded} records excluded by this panel\'s shared finite-value mask.',stacklevel=2)
            for i in range(len(y)):
                rows.append({'panel':panel,'cohort':cohort,'record_position_0based':i,
                             'record_position_scope':'within cohort and panel',
                             'outcome':y[i],'equilibrium':eq[i],'non_equilibrium':lf[i],'included':bool(keep[i])})
    return pd.DataFrame(rows),pd.DataFrame(exclusions)


def _fig2_lookup(collection,subject,index,name):
    if isinstance(collection,Mapping):
        if subject in collection:return collection[subject]
        if index in collection:return collection[index]
        raise KeyError(f'{name}: no key for record {subject!r} (ordinal {index}).')
    return collection[index]


def scan_fig2_nremrem(cohorts,k_values,*,epoch_seconds=30,duration_without_cycle='zero',
                     progress=True,**analysis_options):
    """Pool individual-record targets and forces BEFORE computing correlations.

    Force analysis is the supplied algorithm. Options in the run cell preserve
    the supplied settings. Frequency uses cycles / TST minutes. A no-cycle
    duration is 0 by default, preserving the original scan; 'nan' is optional.
    """
    if duration_without_cycle not in ('zero','nan'):raise ValueError("Use duration_without_cycle='zero' or 'nan'.")
    ks=np.asarray(k_values,float)
    if ks.ndim!=1 or not len(ks) or np.any(~np.isfinite(ks)) or np.any(ks<0) or np.any(np.diff(ks)<=0):
        raise ValueError('k_values must be finite, nonnegative, unique and increasing.')
    options=dict(include_post_rem_state=True,include_wake_in_structure=True,min_segment_edges=1,
                 allow_loop_reuse=True,count_each_loop_once_per_cycle=True,
                 sum_unique_missing_edges=True,multiply_loop_flux_by_n_edges=True)
    options.update(analysis_options)
    extract_options={k:options[k] for k in ('include_post_rem_state','include_wake_in_structure')}
    data_rows=[];cycle_rows=[];outputs={}
    for cohort,c in cohorts.items():
        records=_FIG2_ENGINE['iter'](c['data'])
        if not records:raise ValueError(f'{cohort}: no hypnograms.')
        if not isinstance(c['AllCycle'],Mapping) or not isinstance(c['AllStrength'],Mapping):
            raise TypeError(f'{cohort}: AllCycle and AllStrength must be dicts, as in the supplied analyzer.')
        for idx,subject,hyp in records:
            h=_fig2_vector(hyp,f'{cohort}.{subject}')
            if not np.all(np.isfinite(h)) or not np.all(np.isin(h,[0,1,2,3,4])):raise ValueError(f'{cohort}/{subject}: invalid stages.')
            loops=_fig2_lookup(c['AllCycle'],subject,idx,'AllCycle')
            strengths=np.asarray(_fig2_lookup(c['AllStrength'],subject,idx,'AllStrength'),float)
            if len(loops)!=len(strengths) or np.any(~np.isfinite(strengths)) or np.any(strengths<0):
                raise ValueError(f'{cohort}/{subject}: loop and strength counts or values are invalid.')
            sym=np.asarray(_fig2_lookup(c['Flux_sym_list'],subject,idx,'Flux_sym_list'),float)
            if sym.shape!=(5,5) or np.any(~np.isfinite(sym)) or np.any(sym<0):raise ValueError(f'{cohort}/{subject}: invalid symmetric matrix.')
        for ik,k in enumerate(ks):
            if progress and (ik%10==0 or ik==len(ks)-1):print(f'{cohort}: k={k:.1f} ({ik+1}/{len(ks)})')
            raw=_FIG2_ENGINE['extract'](c['data'],epoch_sec=epoch_seconds,min_nrem_min=3*k,min_rem_min=k,**extract_options)
            if len(raw):
                out=_FIG2_ENGINE['analyze'](data=c['data'],AllCycle=c['AllCycle'],AllStrength=c['AllStrength'],
                    Flux_sym_list=c['Flux_sym_list'],epoch_sec=epoch_seconds,min_nrem_min=3*k,min_rem_min=k,**options)
                force=out['df_hypnogram_force'].set_index('hyp_index');outputs[(cohort,float(k))]=out
            else:force=pd.DataFrame();outputs[(cohort,float(k))]=None
            for idx,subject,hyp in records:
                tst=np.count_nonzero(np.isin(np.asarray(hyp).squeeze(),[1,2,3,4]))*epoch_seconds/60
                if tst<=0:raise ValueError(f'{cohort}/{subject}: no scored sleep, cannot normalize frequency.')
                part=raw[raw['hyp_index']==idx] if len(raw) else pd.DataFrame()
                n=len(part);duration=float(part['duration_min'].mean()) if n else (0. if duration_without_cycle=='zero' else np.nan)
                eq=float(force.loc[idx,'sum_sym_uncovered_edge_sum']) if len(force) else 0.
                lf=float(force.loc[idx,'sum_loop_strength_sum']) if len(force) else 0.
                total=float(force.loc[idx,'sum_loop_plus_sym_force']) if len(force) else 0.
                if not np.isclose(total,eq+lf):raise ValueError('Total force does not equal the selected loop plus uncovered symmetric force.')
                data_rows.append({'cohort':cohort,'hyp_index':idx,'subject':subject,'k':float(k),'TST_min':tst,
                    'n_nremrem_cycles':n,'nremrem_cycle_frequency':n/tst,'mean_nremrem_cycle_duration_min':duration,
                    'duration_defined':bool(n),'equilibrium':eq,'non_equilibrium':lf,'total':total})
                if n:
                    for _,rr in part.iterrows():cycle_rows.append({'cohort':cohort,'subject':subject,'k':float(k),
                        'cycle_duration_min':float(rr['duration_min'])})
    targets=pd.DataFrame(data_rows);cycles=pd.DataFrame(cycle_rows,columns=['cohort','subject','k','cycle_duration_min'])
    corr=[]
    for k,g in targets.groupby('k',sort=True):
        for component in ('equilibrium','non_equilibrium','total'):
            for target,col in (('frequency','nremrem_cycle_frequency'),('duration','mean_nremrem_cycle_duration_min')):
                keep=np.isfinite(g[component])&np.isfinite(g[col]);r,p,lo,hi,n=_fig2_pearson(g.loc[keep,component],g.loc[keep,col])
                corr.append({'k':k,'component':component,'target':target,'pearson_r':r,'pearson_p':p,
                             'r_ci_low':lo,'r_ci_high':hi,'n_valid':n})
    counts=targets.groupby('k',as_index=False).agg(mean_n_nremrem_cycles=('n_nremrem_cycles','mean'),
        sem_n_nremrem_cycles=('n_nremrem_cycles','sem'),n_hypnograms=('hyp_index','size'))
    durations=cycles.groupby('k',as_index=False).agg(mean_cycle_duration_min=('cycle_duration_min','mean'),
        sem_cycle_duration_min=('cycle_duration_min','sem'),n_cycles=('cycle_duration_min','size'))
    return {'df_corr':pd.DataFrame(corr),'df_target_by_k':targets,'df_cycle_count_by_k':targets.copy(),
            'df_cycle_duration_by_k':cycles,'df_cycle_summary_by_k':counts.merge(durations,on='k',how='left'),
            'out_force_by_k':outputs,'k_values':ks,'analysis_options':options,
            'duration_without_cycle':duration_without_cycle,'epoch_seconds':epoch_seconds}

def _fig2_rc(font_size):
    q=float(font_size)/7
    return {'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':7*q,'axes.labelsize':6*q,'xtick.labelsize':5.8*q,'ytick.labelsize':5.8*q,
        'axes.linewidth':.55,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':'#263844','text.color':'#263844','xtick.color':'#4E5961','ytick.color':'#4E5961',
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'savefig.facecolor':'white'}


def _fig2_size(font_size,width,height):
    for name,v in [('font_size',font_size),('fig_width_mm',width),('fig_height_mm',height)]:
        if not np.isscalar(v) or not np.isfinite(v) or v<=0:raise ValueError(f'{name} must be positive and finite.')


def _fig2_check_alignment(fig,groups):
    """Self-contained physical plot-area gate; the QA hook can run a fuller audit."""
    fig.canvas.draw();w,h=fig.get_size_inches()*25.4
    bounds=[]
    for group in groups:
        rects=np.array([a.get_position().bounds for a in group])*[w,h,w,h]
        if np.any(np.ptp(rects[:,[1,2,3]],axis=0)>1.5*25.4/72):
            raise RuntimeError('Comparable plot areas are not aligned within 1.5 pt.')
        bounds.extend([{'axes':a.get_label(),'rect_mm':r.tolist()} for a,r in zip(group,rects)])
    return bounds


def _fig2_save(fig,output_dir,basename,dpi=600):
    if output_dir is None:return {}
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    paths={}
    for fmt in ('pdf','svg','png'):
        path=out/f'{basename}.{fmt}'
        fig.savefig(path,dpi=dpi,facecolor='white')
        paths[fmt]=str(path.resolve())
    return paths


def _fig2_grouped_correlations(association_df,scan):
    """Separate cohort estimates, exported alongside the pooled main results."""
    ad=[];e=[]
    valid=association_df[association_df.included]
    for (panel,cohort),g in valid.groupby(['panel','cohort'],sort=True):
        for component in ('equilibrium','non_equilibrium'):
            r,p,lo,hi,n=_fig2_pearson(g[component],g.outcome)
            ad.append({'panel':panel,'cohort':cohort,'component':component,
                       'pearson_r':r,'pearson_p':p,'r_ci_low':lo,'r_ci_high':hi,'n':n})
    for (k,cohort),g in scan['df_target_by_k'].groupby(['k','cohort'],sort=True):
        for component in ('equilibrium','non_equilibrium','total'):
            for target,column in [('frequency','nremrem_cycle_frequency'),('duration','mean_nremrem_cycle_duration_min')]:
                keep=np.isfinite(g[component])&np.isfinite(g[column])
                r,p,lo,hi,n=_fig2_pearson(g.loc[keep,component],g.loc[keep,column])
                e.append({'k':k,'cohort':cohort,'component':component,'target':target,
                          'pearson_r':r,'pearson_p':p,'r_ci_low':lo,'r_ci_high':hi,'n':n})
    return pd.DataFrame(ad),pd.DataFrame(e)



def _fig2_text_geometry(fig):
    """Check complete rendered text artists, treating math scripts as one label."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer();items=[];seen=set()
    def keep(t):
        if id(t) in seen or not t.get_visible() or not t.get_text():return
        seen.add(id(t));items.append((t,t.get_window_extent(renderer)))
    for a in fig.axes:
        for t in a.texts:keep(t)
        if not a.axison:continue
        if a.xaxis.get_visible():
            keep(a.xaxis.label)
            lo,hi=sorted(a.get_xlim())
            for t in a.get_xticklabels():
                if lo-1e-10<=t.get_position()[0]<=hi+1e-10:keep(t)
        if a.yaxis.get_visible():
            keep(a.yaxis.label)
            lo,hi=sorted(a.get_ylim())
            for t in a.get_yticklabels():
                if lo-1e-10<=t.get_position()[1]<=hi+1e-10:keep(t)
    for legend in fig.legends:
        for t in legend.get_texts():keep(t)
    errors=[]
    for i,(t,box) in enumerate(items):
        if box.x0<0 or box.y0<0 or box.x1>fig.bbox.x1 or box.y1>fig.bbox.y1:
            errors.append('clipped: '+t.get_text())
        for other,otherbox in items[i+1:]:
            if box.overlaps(otherbox):errors.append(t.get_text()+' / '+other.get_text())
    if errors:raise RuntimeError('Rendered Fig2 text overlaps; enlarge physical canvas or adjust layout: '+'; '.join(errors[:8]))
    return {'result':'PASS','complete_text_artists':len(items),'boxes_pixels':[
        {'text':t.get_text(),'bounds':list(b.bounds)} for t,b in items]}

def _fig2_ptext(p):
    """Pearson p in scientific notation with a real superscript exponent."""
    if not np.isfinite(p):return r'$p=\mathrm{NA}$'
    if p==0:return r'$p\approx 0$'
    coefficient,exponent=f'{p:.2e}'.split('e')
    return rf'$p={coefficient}\times 10^{{{int(exponent)}}}$'


def plot_fig2_main(association_df,scan,*,family_counts,font_size=10,fig_width_mm=180,fig_height_mm=None,
                   bootstrap_n=2000,random_seed=2026,show_trend_ci=True,
                   output_dir='Fig2_output',basename='Fig2_combined',dpi=600,_qa_callback=None):
    """One a-f figure: 2x2 dual-y a-d, two dual-y e plots, full-height f.

    a-d x: sleep metric; blue left y: equilibrium; orange right y: flux.
    Both components use the same complete recording subset within each panel.
    Pearson r, scientific-notation two-sided p and n remain on-figure.
    Correlation CIs, paired Delta |r| CI and separate cohort estimates are exported. Dual-axis slopes are not compared.
    Force analysis, data units and panel-specific sample sizes are unchanged.
    """
    if fig_height_mm is None:fig_height_mm=max(165.,40.+2.9*len(family_counts))*max(1.,font_size/7)
    _fig2_size(font_size,fig_width_mm,fig_height_mm)
    blue,orange,gray,ink='#0072B2','#D55E00','#71808A','#263844'
    colors={'equilibrium':blue,'non_equilibrium':orange,'total':gray}
    fs=lambda n:n*font_size/7
    W,H=280.,165.
    left_scale=.92
    stats=[];differences=[];axes={}
    cohort_stats,e_cohort_stats=_fig2_grouped_correlations(association_df,scan)
    with mpl.rc_context(_fig2_rc(font_size)):
        fig=plt.figure(figsize=(fig_width_mm/25.4,fig_height_mm/25.4),dpi=150,facecolor='white')
        canvas=fig.add_axes([0,0,1,1],label='layout');canvas.set(xlim=(0,W),ylim=(0,H));canvas.axis('off')
        def label(x,y,s,size=7,weight='normal',ha='left',color=ink):
            return canvas.text(x,y,s,fontsize=fs(size),fontweight=weight,ha=ha,va='center',color=color)
        def add(x,y,w,h,name):return fig.add_axes([x/W,y/H,w/W,h/H],label=name)
        handles=[Line2D([],[],color=blue,lw=1.1,label='Equilibrium'),
                 Line2D([],[],color=orange,lw=1.1,label='Non-equilibrium'),
                 Line2D([],[],marker='o',color=ink,lw=0,markersize=3,label='Nocturnal'),
                 Line2D([],[],marker='^',color=ink,lw=0,markersize=3,label='Nap')]
        fig.legend(handles=handles,loc='center',bbox_to_anchor=(93*left_scale/W,158/H),ncol=2,
                   fontsize=fs(5.8),handlelength=1.8,handletextpad=.4,columnspacing=1.5)
        specs=[('a','Overall transitions','Transition frequency (/min)',r'EQ force ($\mathrm{DF}_{\mathrm{EQ}}$)',r'Loop flux ($\mathrm{DF}_{\mathrm{LF}}$)'),
               ('b','Awakenings','Awakening frequency (/min)',r'Wake EQ ($\mathrm{DF}_{\mathrm{EQ}}^{\mathrm{Wake}}$)',r'Wake LF ($\mathrm{DF}_{\mathrm{LF}}^{\mathrm{Wake}}$)'),
               ('c','Sleep onset','Sleep onset latency (min)',r'Early EQ ($\mathrm{DF}_{\mathrm{EQ}}^{\mathrm{Early}}$)',r'Early LF ($\mathrm{DF}_{\mathrm{LF}}^{\mathrm{Early}}$)'),
               ('d','Hypnocycles','Hypnocycle frequency (/min)',r'EQ force ($\mathrm{DF}_{\mathrm{EQ}}$)',r'Loop flux ($\mathrm{DF}_{\mathrm{LF}}$)')]
        rows=[[],[]]
        for i,(panel,heading,xlabel,yeq,ylf) in enumerate(specs):
            row,col=divmod(i,2);left=(28+98*col)*left_scale;top=146-47*row;bottom=109-47*row
            g=association_df[(association_df.panel==panel)&association_df.included].copy()
            if len(g)<3:raise ValueError(f'Panel {panel}: fewer than 3 complete records.')
            x=g.outcome.to_numpy();eq=g.equilibrium.to_numpy();lf=g.non_equilibrium.to_numpy()
            rc0=_fig2_pearson(eq,x);rc1=_fig2_pearson(lf,x)
            lo,hi,nb=_fig2_bootstrap_difference(eq,lf,x,g.cohort.to_numpy(),bootstrap_n,random_seed+i)
            differences.append({'panel':panel,'delta_abs_r_LF_minus_EQ':abs(rc1[0])-abs(rc0[0]),
                'ci_low':lo,'ci_high':hi,'bootstrap_valid':nb,'bootstrap_unit':'recording; paired, stratified by cohort'})
            label(left-13*left_scale,top,panel,8,'bold')
            label(left+44*left_scale,top-3,f'n = {len(g)}',5.8,ha='right',color=gray)
            ax=add(left,bottom,44*left_scale,24,f'{panel}_equilibrium');right=ax.twinx();right.set_label(f'{panel}_non_equilibrium')
            right.patch.set_visible(False)
            axes[f'{panel}_equilibrium']=ax;axes[f'{panel}_non_equilibrium']=right;rows[row].append(ax)
            xr=float(np.ptp(x));xp=.06*xr if xr else max(abs(float(x[0]))*.1,1.)
            xlim=(max(0.,float(x.min())-xp),float(x.max())+xp)
            for component,y,a,ylabel,rc in [('equilibrium',eq,ax,yeq,rc0),('non_equilibrium',lf,right,ylf,rc1)]:
                color=colors[component]
                yr=float(np.ptp(y));yp=.1*yr if yr else max(abs(float(y[0]))*.15,.001)
                ylim=(max(0.,float(y.min())-yp),float(y.max())+yp)
                if ylim[1]<=ylim[0]:ylim=(0.,.001)
                a.set_ylim(ylim)
                for cohort,marker in [('Nocturnal','o'),('Nap','^')]:
                    keep=g.cohort.to_numpy()==cohort
                    a.scatter(x[keep],y[keep],s=7.5,marker=marker,color=color,alpha=.43,linewidths=0,rasterized=False,zorder=3)
                if np.ptp(x)>0:
                    grid=np.linspace(x.min(),x.max(),160);slope,intercept=np.polyfit(x,y,1);pred=intercept+slope*grid
                    visible=(pred>=ylim[0])&(pred<=ylim[1]);a.plot(grid[visible],pred[visible],color=color,lw=1.05,zorder=4)
                    if len(x)>2:
                        var=np.sum((y-(intercept+slope*x))**2)/(len(x)-2)
                        se=np.sqrt(var*(1/len(x)+(grid-x.mean())**2/np.sum((x-x.mean())**2)))
                        half=student_t.ppf(.975,len(x)-2)*se
                        a.fill_between(grid,np.clip(pred-half,*ylim),np.clip(pred+half,*ylim),color=color,alpha=.055,lw=0,zorder=1)
                a.set_ylabel(ylabel,color=color,labelpad=1,fontsize=fs(7.2))
                a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
                a.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value,pos:f'{value:.2g}'))
                a.tick_params(axis='y',colors=color,length=2,width=.5,pad=2,labelsize=fs(5.8))
                if component=='equilibrium':a.spines['left'].set_edgecolor(color)
                else:
                    a.spines['left'].set_visible(False);a.spines['bottom'].set_visible(False)
                    a.spines['top'].set_visible(False);a.spines['right'].set_visible(True);a.spines['right'].set_edgecolor(color)
                r,p,clo,chi,n=rc
                caption=(f'r = {r:.2f}; ' if np.isfinite(r) else 'r = NA; ')+_fig2_ptext(p)
                label(left,top-(7 if component=='equilibrium' else 11),caption,5.8,color=color)
                stats.append({'panel':panel,'component':component,'pearson_r':r,'pearson_p':p,'r_ci_low':clo,'r_ci_high':chi,'n':n})
            ax.set_xlim(xlim);ax.set_xlabel(xlabel,labelpad=2)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value,pos:f'{value:.2g}'))
            ax.tick_params(axis='x',length=2,width=.5,pad=2,labelsize=fs(5.8))
            if not np.allclose(ax.get_position().bounds,right.get_position().bounds):raise RuntimeError('Dual axes are not aligned.')
        # e: two dual-axis plots on the same row.
        label(5*left_scale,50,'e',8,'bold')
        # Hue identifies the target/axis; shade and weight identify the force component.
        target_colors={
            'frequency':{'total':'#D4B6A0','equilibrium':'#D88D58','non_equilibrium':orange},
            'duration':{'total':'#B8CEDC','equilibrium':'#71A5C4','non_equilibrium':blue},
        }
        component_styles={'total':(.65,.85),'equilibrium':(.8,.9),'non_equilibrium':(1.65,1.)}
        # Column-major legend: orange frequency row, blue duration row.
        handles=[Line2D([],[],color=target_colors[target][component],ls='-',
                       lw=component_styles[component][0],alpha=component_styles[component][1],label=name)
                 for component,name in [('equilibrium','EQ'),('non_equilibrium','Loop flux'),('total','Total')]
                 for target in ('frequency','duration')]
        fig.legend(handles=handles,loc='center',bbox_to_anchor=(67*left_scale/W,45/H),ncol=3,
                   fontsize=fs(5.5),handlelength=1.5,columnspacing=.8,labelspacing=.35)
        corr=scan['df_corr'];e_corr=add(18*left_scale,14,57*left_scale,23,'e_frequency')
        e_corr_duration=e_corr.twinx();e_corr_duration.set_label('e_duration');e_corr_duration.patch.set_visible(False)
        axes['e_frequency']=e_corr;axes['e_duration']=e_corr_duration
        # Draw weak-background components first, the highlighted flux last.
        for component in ('total','equilibrium','non_equilibrium'):
            for target,a in [('frequency',e_corr),('duration',e_corr_duration)]:
                part=corr[(corr.target==target)&(corr.component==component)].sort_values('k')
                a.plot(part.k,part.pearson_r,color=target_colors[target][component],ls='-',
                       lw=component_styles[component][0],
                       alpha=component_styles[component][1],
                       zorder=8 if component=='non_equilibrium' else 2)
                if show_trend_ci:
                    a.fill_between(part.k,part.r_ci_low,part.r_ci_high,color=target_colors[target][component],
                                   alpha=.045 if component=='non_equilibrium' else .009,lw=0,zorder=0)
        # Both r axes use exactly the same scale.
        for a,ylabel,color in [(e_corr,'Frequency r',orange),(e_corr_duration,'Duration r',blue)]:
            a.set(xlim=(float(corr.k.min()),max(5.15,float(corr.k.max()))),ylim=(-1,1),yticks=[-1,0,1],xticks=np.arange(6))
            a.set_ylabel(ylabel,color=color,labelpad=1,fontsize=fs(6))
            a.tick_params(axis='y',colors=color,length=2,width=.5,pad=2,labelsize=fs(5.8))
            a.tick_params(axis='x',length=2,width=.5,pad=2,labelsize=fs(5.8))
        e_corr.spines['left'].set_edgecolor(orange)
        e_corr.axhline(0,color='#D4DCE1',lw=.5,zorder=0)
        e_corr.axvline(5,color=gray,lw=.6,ls=(0,(3,2)),zorder=0)
        e_corr.set_xlabel('Minimum REM duration, k (min)',labelpad=2)
        e_corr_duration.spines['left'].set_visible(False);e_corr_duration.spines['bottom'].set_visible(False)
        e_corr_duration.spines['top'].set_visible(False);e_corr_duration.spines['right'].set_visible(True);e_corr_duration.spines['right'].set_edgecolor(blue)
        # e descriptive panel: number of cycles versus mean duration.
        e_count=add(110*left_scale,14,57*left_scale,23,'e_count')
        e_duration=e_count.twinx();e_duration.set_label('e_mean_duration');e_duration.patch.set_visible(False)
        axes['e_count']=e_count;axes['e_mean_duration']=e_duration
        summary=scan['df_cycle_summary_by_k']
        for a,mean,sem,color,ls,ylabel in [
            (e_count,'mean_n_nremrem_cycles','sem_n_nremrem_cycles',orange,'-','Cycles / hypnogram'),
            (e_duration,'mean_cycle_duration_min','sem_cycle_duration_min',blue,'-','Duration (min)')]:
            valid=np.isfinite(summary[mean]);part=summary[valid];error=part[sem].fillna(0)
            a.plot(part.k,part[mean],color=color,ls=ls,lw=1)
            a.fill_between(part.k,part[mean]-error,part[mean]+error,color=color,alpha=.07,lw=0)
            a.set_xlim(float(corr.k.min()),max(5.15,float(corr.k.max())))
            a.set_ylabel(ylabel,color=color,labelpad=1,fontsize=fs(6))
            a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            a.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,p:f'{v:.3g}'))
            a.tick_params(axis='y',colors=color,length=2,width=.5,pad=2,labelsize=fs(5.8))
        e_count.set_xticks(np.arange(6));e_count.tick_params(axis='x',length=2,width=.5,pad=2,labelsize=fs(5.8))
        e_count.set_xlabel('Minimum REM duration, k (min)',labelpad=2)
        e_count.spines['left'].set_edgecolor(orange)
        e_duration.spines['left'].set_visible(False);e_duration.spines['bottom'].set_visible(False)
        e_duration.spines['top'].set_visible(False);e_duration.spines['right'].set_visible(True);e_duration.spines['right'].set_edgecolor(blue)
        e_count.axvline(5,color=gray,lw=.6,ls=(0,(3,2)),zorder=0)
        label(138.5*left_scale,45,'Mean ± SEM',5.5,ha='center',color=gray)
        # f occupies the full right-hand column. Retain every supplied family.
        label(177,155,'f',8,'bold')
        family_handles=[Line2D([],[],marker='o',color='#52606A',lw=0,markersize=3,label='Observed hypnocycles'),
                        Line2D([],[],marker='^',color=orange,lw=0,markersize=3,label='Inferred flux loops')]
        fig.legend(handles=family_handles,loc='center',bbox_to_anchor=(235/W,147/H),ncol=1,
                   fontsize=fs(5.7),handletextpad=.4,labelspacing=.35)
        af=add(229,19,43,119,'f');axes['f']=af
        for i,r in family_counts.iterrows():
            af.plot([r.cycle_count,r.loop_count],[i,i],color='#D7DFE4',lw=.65,zorder=1)
        af.scatter(family_counts.cycle_count,np.arange(len(family_counts)),s=10,color='#52606A',marker='o',zorder=3)
        af.scatter(family_counts.loop_count,np.arange(len(family_counts)),s=12,color=orange,marker='^',zorder=3)
        af.set_yticks(np.arange(len(family_counts)),family_counts.label.str.replace(' → ','→',regex=False),fontsize=fs(5.5));af.invert_yaxis()
        af.set_ylim(len(family_counts)-.5,-.5)
        af.set_xscale('symlog',linthresh=1)
        maximum=max(family_counts.cycle_count.max(),family_counts.loop_count.max(),1)
        af.set_xlim(-.15,maximum*1.35)
        ticks=[0,1]+[10**i for i in range(1,int(np.floor(np.log10(maximum)))+1)]
        # Sparse labels for large real-data counts; values and symlog scale stay intact.
        if len(ticks)>4:
            ticks=[ticks[i] for i in np.unique(np.linspace(0,len(ticks)-1,4).astype(int))]
        def count_tick(v):
            if v>=1_000_000:return f'{v/1_000_000:g}M'
            if v>=1000:return f'{v/1000:g}k'
            return str(int(v))
        af.set_xticks(ticks,[count_tick(v) for v in ticks]);af.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        af.tick_params(length=2,width=.5,pad=2,labelsize=fs(5.5))
        af.set_xlabel('Occurrence count\n'+('(symlog; k = 1000)' if maximum>=1000 else '(symlog)'),labelpad=2,loc='right')
        for i,r in family_counts.iterrows():
            if r.status:af.text(1.02,i,r.status,transform=af.get_yaxis_transform(),fontsize=fs(5.5),ha='left',va='center',color=gray)
        label(235,7.5,'W: Wake; R: REM',5.5,ha='center',color=gray)
        label(235,4.5,'LO: loop only; CO: hypnocycle only',5.5,ha='center',color=gray)
        label(88*left_scale,2.5,'k = 5: ≥15 min NREM and ≥5 min REM.',5.5,ha='center',color=gray)
        alignment=_fig2_check_alignment(fig,rows+[[e_corr,e_count]])
        for first,second in [(e_corr,e_corr_duration),(e_count,e_duration)]:
            if not np.allclose(first.get_position().bounds,second.get_position().bounds):raise RuntimeError('e dual axes are not aligned.')
        result={'figure':fig,'axes':axes,'stats':pd.DataFrame(stats),'correlation_differences':pd.DataFrame(differences),
                'cohort_stats':cohort_stats,'e_cohort_stats':e_cohort_stats,'family_counts':family_counts,
                'alignment':alignment,'paths':{},'layout_exemptions':{'f':'Full-height categorical panel, intentionally distinct from the quantitative grid'}}
        if _qa_callback is not None:_qa_callback(result)
        result['text_geometry']=_fig2_text_geometry(fig)
        result['paths']=_fig2_save(fig,output_dir,basename,dpi)
        if output_dir is not None:
            out=Path(output_dir)
            for key,suffix in [('stats','correlations'),('correlation_differences','correlation_differences'),
                               ('cohort_stats','cohort_correlations'),('e_cohort_stats','E_cohort_correlations')]:
                result[key].to_csv(out/f'{basename}_{suffix}.csv',index=False)
            (out/f'{basename}_text_geometry.json').write_text(json.dumps(result['text_geometry'],indent=2),encoding='utf-8')
            family_counts.to_csv(out/f'{basename}_family_counts.csv',index=False)
            scan['df_cycle_summary_by_k'].to_csv(out/f'{basename}_cycle_summary.csv',index=False)
            association_df.to_csv(out/f'{basename}_scatter_source.csv',index=False)
            (out/f'{basename}_alignment.json').write_text(json.dumps(alignment,indent=2),encoding='utf-8')
    return result



def _fig2_family_tuple(value):
    if isinstance(value,str):value=ast.literal_eval(value)
    a=np.asarray(value)
    if a.ndim!=1 or not len(a) or np.any(~np.isfinite(a.astype(float))) or not np.all(np.isin(a,[0,1,2,3,4])):
        raise ValueError(f'Invalid loop/cycle family: {value!r}')
    return tuple(int(x) for x in a)

def prepare_fig2_family_counts(Noc_df_occ,Nap_df_occ,NocAllCycle,NapAllCycle):
    """Retain the exact tuple-family definition: no new rotations or intersections."""
    totals={}
    for cohort,occ,loops in [('Nocturnal',Noc_df_occ,NocAllCycle),('Nap',Nap_df_occ,NapAllCycle)]:
        if 'cycle_family' not in occ:raise KeyError(f'{cohort}: cycle_family column is required.')
        totals[(cohort,'cycle')]=Counter(_fig2_family_tuple(x) for x in occ['cycle_family'])
        totals[(cohort,'loop')]=Counter(_fig2_family_tuple(x) for records in loops.values() for x in records)
    families=set().union(*(set(v) for v in totals.values()))
    rows=[]
    for family in families:
        r={'cycle_family':family,'label':' → '.join({0:'W',1:'N1',2:'N2',3:'N3',4:'R'}[s] for s in family)}
        for cohort in ('Nocturnal','Nap'):
            for kind in ('cycle','loop'):r[cohort+'_'+kind]=totals[(cohort,kind)].get(family,0)
        r['cycle_count']=r['Nocturnal_cycle']+r['Nap_cycle'];r['loop_count']=r['Nocturnal_loop']+r['Nap_loop']
        r['status']='LO' if r['cycle_count']==0 else 'CO' if r['loop_count']==0 else ''
        rows.append(r)
    if not rows:raise ValueError('No cycle/loop families to display.')
    return pd.DataFrame(rows).sort_values(['cycle_count','loop_count','label'],ascending=[False,False,True]).reset_index(drop=True)


# ======================================================================
# Definitions from cleaned notebook cell 55 (zero-based)
# ======================================================================

# Cell 1: def. Dependencies: numpy, pandas, scipy, matplotlib.
from pathlib import Path
from collections.abc import Mapping
import ast
import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

_FIG3_STAGE = {0:'W',1:'N1',2:'N2',3:'N3',4:'R'}


def _fig3_family(value):
    """Read existing states without reordering or rotation canonicalization."""
    if isinstance(value,str):
        text=value.strip()
        try:
            parsed=ast.literal_eval(text)
        except (ValueError,SyntaxError):
            parsed=None
        if isinstance(parsed,(tuple,list)):
            value=parsed
        elif re.fullmatch(r'[0-4]+',text):
            value=[int(c) for c in text]
        else:
            tokens=re.split(r'\s*(?:→|↔|->|<->|,|\s+|-)\s*',text.strip('()[] '))
            names={'W':0,'WAKE':0,'N1':1,'N2':2,'N3':3,'R':4,'REM':4}
            value=[names[t.upper()] if t.upper() in names else int(t) for t in tokens if t]
    a=np.asarray(value)
    if a.ndim!=1 or not len(a):raise ValueError(f'Invalid state family: {value!r}')
    a=a.astype(float)
    if not np.all(np.isfinite(a)) or not np.all(np.isin(a,list(_FIG3_STAGE))):
        raise ValueError(f'Invalid sleep stages: {value!r}')
    return tuple(a.astype(int))


def _fig3_label(value,pair=False):
    try:
        family=_fig3_family(value)
    except (ValueError,TypeError,KeyError):
        return str(value)  # Preserve an existing descriptive label in full.
    return (' ↔ ' if pair else ' → ').join(_FIG3_STAGE[s] for s in family)


def _fig3_vector(value,name):
    a=np.asarray(value,dtype=float)
    if a.ndim==2 and 1 in a.shape:a=a.reshape(-1)
    if a.ndim!=1:raise ValueError(f'{name} must be one-dimensional; got {a.shape}.')
    return a.copy()


def _fig3_occurrences(frame,kind,duration_factor):
    code,name,duration=(('loop_family_code','loop_name','duration_min') if kind=='loop' else
                        ('exchange_pair_code','exchange_pair_name','exchange_unit_duration_min'))
    needed={'location',code,name,duration}
    if not isinstance(frame,pd.DataFrame):raise TypeError(f'{kind} occurrences must be a DataFrame.')
    if needed-set(frame):raise KeyError(f'{kind} occurrence columns missing: {sorted(needed-set(frame))}')
    src=frame.copy(deep=True).reset_index(drop=True)
    src.insert(0,'fig3_source_position_0based',np.arange(len(src)))
    src['location']=src.location.astype('string').str.strip().str.lower()
    if src.location.isna().any() or not src.location.isin(['inside','outside']).all():
        raise ValueError(f'{kind}: location must be inside or outside; no unknown rows are silently dropped.')
    if src[[code,name]].isna().any().any():raise ValueError(f'{kind}: category code/name contains missing values.')
    src['fig3_category_code']=src[code].map(str)
    src['fig3_category_name']=src[name].map(str)
    src['fig3_display_label']=src[name].map(lambda x:_fig3_label(x,pair=kind=='exchange'))
    scale=1. if kind=='loop' else duration_factor
    src['fig3_duration_min']=pd.to_numeric(src[duration],errors='coerce')*scale
    src['fig3_duration_included']=np.isfinite(src.fig3_duration_min)&(src.fig3_duration_min>0)
    src['fig3_duration_exclusion']=np.where(src.fig3_duration_included,'','non-finite or non-positive duration')
    group=['fig3_category_code','fig3_category_name','fig3_display_label']
    if len(src):
        table=src.groupby(group+['location'],sort=False).size().unstack('location',fill_value=0)
        table=table.reindex(columns=['inside','outside'],fill_value=0).reset_index()
    else:
        table=pd.DataFrame(columns=group+['inside','outside'])
    table=table.rename(columns={'inside':'inside_count','outside':'outside_count',
        'fig3_category_code':'category_code','fig3_category_name':'category_name','fig3_display_label':'label'})
    for location in ('inside','outside'):
        table[location+'_count']=table[location+'_count'].astype(int)
        n=int((src.location==location).sum())
        table[location+'_percent']=100*table[location+'_count']/n if n else np.nan
    table['total_count']=table.inside_count+table.outside_count
    for location in ('inside','outside'):
        table[location+'_within_category_percent']=100*table[location+'_count']/table.total_count.where(table.total_count>0)
    table=table.sort_values(['total_count','inside_count','outside_count','label'],ascending=[False,False,False,True]).reset_index(drop=True)
    return src,table


def _fig3_duration_stats(source,kind,min_n):
    arrays={loc:source.loc[(source.location==loc)&source.fig3_duration_included,'fig3_duration_min'].to_numpy(float)
            for loc in ('inside','outside')}
    row={'kind':kind,'unit':'minutes','test_unit':'pooled occurrence; exploratory, no subject clustering',
         'min_n_per_group':int(min_n)}
    for loc,a in arrays.items():
        row.update({loc+'_n':len(a),loc+'_excluded_duration_n':int(((source.location==loc)&~source.fig3_duration_included).sum()),
          loc+'_mean':float(np.mean(a)) if len(a) else np.nan,
          loc+'_median':float(np.median(a)) if len(a) else np.nan,
          loc+'_q25':float(np.quantile(a,.25)) if len(a) else np.nan,
          loc+'_q75':float(np.quantile(a,.75)) if len(a) else np.nan,
          loc+'_std':float(np.std(a,ddof=1)) if len(a)>1 else (0. if len(a) else np.nan)})
    for metric in ('mean','median'):
        row[metric+'_diff_inside_minus_outside']=row['inside_'+metric]-row['outside_'+metric]
        denominator=row['outside_'+metric]
        row[metric+'_ratio_inside_over_outside']=row['inside_'+metric]/denominator if denominator>0 else np.nan
    row.update(mannwhitney_U=np.nan,mannwhitney_p=np.nan,ks_stat=np.nan,ks_p=np.nan,welch_t=np.nan,welch_p=np.nan)
    if min(map(len,arrays.values()))>=min_n:
        a,b=arrays['inside'],arrays['outside']
        mw=stats.mannwhitneyu(a,b,alternative='two-sided',method='auto',use_continuity=True)
        ks=stats.ks_2samp(a,b,alternative='two-sided',method='auto')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            wt=stats.ttest_ind(a,b,equal_var=False,nan_policy='omit')
        row.update(mannwhitney_U=float(mw.statistic),mannwhitney_p=float(mw.pvalue),
                   ks_stat=float(ks.statistic),ks_p=float(ks.pvalue),welch_t=float(wt.statistic),welch_p=float(wt.pvalue))
    return arrays,row



def _fig3_duration_rank_tests(duration_stats):
    """Two planned unpaired comparisons; original valid positive event samples.

    Reuse the already computed two-sided Mann-Whitney U p-values. This is the
    Wilcoxon rank-sum family, not a paired signed-rank test. No p adjustment.
    """
    rows=[]
    for row in duration_stats.itertuples(index=False):
        p=float(row.mannwhitney_p)
        symbol=('***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns') if np.isfinite(p) else 'NA'
        if not np.isfinite(p):caption=f'Not tested (n < {int(row.min_n_per_group)})'
        else:
            if p==0:ptext=r'$p\approx 0$'
            else:
                coefficient,exponent=f'{p:.2e}'.split('e')
                ptext=rf'$p={coefficient}\times 10^{{{int(exponent)}}}$' 
            caption=f'{ptext} ({symbol})'
        rows.append({'kind':row.kind,'test':'Wilcoxon rank-sum (Mann-Whitney U)',
            'alternative':'two-sided','sampling_unit':'pooled occurrence; no subject clustering',
            'n_inside':int(row.inside_n),'n_outside':int(row.outside_n),
            'U_inside':float(row.mannwhitney_U),
            'rank_sum_inside':float(row.mannwhitney_U)+row.inside_n*(row.inside_n+1)/2 if np.isfinite(row.mannwhitney_U) else np.nan,
            'p_value':p,'p_adjustment':'none','alpha':.05,'significant':bool(p<.05) if np.isfinite(p) else None,
            'symbol':symbol,'caption':caption,'status':'tested' if np.isfinite(p) else 'insufficient valid events',
            'method':'SciPy auto: exact for small untied samples; otherwise tie-corrected asymptotic',
            'use_continuity':True})
    return pd.DataFrame(rows)


def _fig3_n3_boundaries(hypnograms):
    """Original rule: count each observed side of a continuous N3 run separately."""
    if not isinstance(hypnograms,Mapping):raise TypeError('NocHypDict must map recording keys to hypnograms.')
    events=[];counts={side:{s:0 for s in (0,1,2,4)} for side in ('before','after')}
    for key,value in hypnograms.items():
        h=_fig3_vector(value,f'NocHypDict[{key!r}]')
        if not np.all(np.isfinite(h)) or not np.all(np.isin(h,list(_FIG3_STAGE))):
            raise ValueError(f'Hypnogram {key!r}: expected finite stage codes 0..4; no epoch is silently removed.')
        h=h.astype(int);inside=h==3
        if not len(h):continue
        starts=np.flatnonzero(inside&np.r_[True,~inside[:-1]])
        ends=np.flatnonzero(inside&np.r_[~inside[1:],True])
        for start,end in zip(starts,ends):
            before=int(h[start-1]) if start else None
            after=int(h[end+1]) if end+1<len(h) else None
            if before is not None:counts['before'][before]+=1
            if after is not None:counts['after'][after]+=1
            events.append({'recording_key':key,'n3_start_epoch_0based':int(start),'n3_end_epoch_0based_inclusive':int(end),
                           'before_state':before,'after_state':after,'before_observed':before is not None,'after_observed':after is not None})
    rows=[]
    for side in ('before','after'):
        total=sum(counts[side].values())
        for stage,n in counts[side].items():
            rows.append({'side':side,'state':stage,'label':_FIG3_STAGE[stage],'count':n,'denominator':total,
                         'percentage':100*n/total if total else np.nan})
    return pd.DataFrame(rows),pd.DataFrame(events,columns=['recording_key','n3_start_epoch_0based',
        'n3_end_epoch_0based_inclusive','before_state','after_state','before_observed','after_observed'])


def _fig3_no_n3(family_counts):
    needed={'cycle_family','cycle_count','loop_count'}
    if not isinstance(family_counts,pd.DataFrame) or needed-set(family_counts):
        raise ValueError('B requires a full family-count table with cycle_family, cycle_count, loop_count.')
    src=family_counts.copy(deep=True).reset_index(drop=True)
    src['cycle_family']=src.cycle_family.map(_fig3_family)
    src['without_n3']=src.cycle_family.map(lambda v:3 not in v)
    rows=[]
    for name,col in [('Observed hypnocycles','cycle_count'),('Decomposed flux loops','loop_count')]:
        a=pd.to_numeric(src[col],errors='coerce').to_numpy(float)
        if np.any(~np.isfinite(a)) or np.any(a<0) or np.any(a!=np.floor(a)):
            raise ValueError(f'B: {col} must contain finite non-negative integer occurrence counts.')
        src[col]=a.astype(np.int64)
        total=int(src[col].sum());n=int(src.loc[src.without_n3,col].sum())
        rows.append({'type':name,'without_n3_count':n,'total_count':total,'with_n3_count':total-n,
                     'proportion':n/total if total else np.nan,'percentage':100*n/total if total else np.nan})
    return pd.DataFrame(rows),src


def _fig3_pearson(x,y):
    n=len(x)
    row={'n':n,'pearson_r':np.nan,'pearson_p':np.nan,'r_ci_low':np.nan,'r_ci_high':np.nan}
    if n<3 or np.ptp(x)==0 or np.ptp(y)==0:return row
    r,p=map(float,stats.pearsonr(x,y));lo=hi=np.nan
    if n>3:
        if abs(r)>=1-1e-14:lo=hi=r
        else:lo,hi=np.tanh(np.arctanh(r)+np.array([-1,1])*stats.norm.ppf(.975)/np.sqrt(n-3))
    row.update(pearson_r=r,pearson_p=p,r_ci_low=float(lo),r_ci_high=float(hi))
    return row


def _fig3_prepare_hypnogram_example(hypnograms,key=30,*,epoch_seconds=30.,min_nrem_minutes=15.,min_rem_minutes=5.):
    """Select the exact key, retain every epoch, and use the Fig1/Fig2 cycle rule.

    A cycle starts at first sleep, then immediately after the previous accepted
    REM bout. Accumulated NREM before a qualifying uninterrupted REM bout must
    reach the threshold. Wake and short REM bouts do not reset accumulation.
    End epochs are exclusive. An incomplete final segment is not a full cycle.
    """
    for name,value in [('epoch_seconds',epoch_seconds),('min_nrem_minutes',min_nrem_minutes),('min_rem_minutes',min_rem_minutes)]:
        if not np.isfinite(value) or value<=0:raise ValueError(f'{name} must be positive and finite.')
    try:raw=hypnograms[key]
    except (KeyError,IndexError,TypeError) as error:
        raise KeyError(f'Fig3a requires NocHypDict[{key!r}]; no other recording is substituted.') from error
    stage=np.asarray(raw,dtype=float)
    if stage.ndim==2 and 1 in stage.shape:stage=stage.reshape(-1)
    if stage.ndim!=1 or not len(stage):raise ValueError('Fig3a needs a nonempty one-dimensional hypnogram.')
    if not np.isfinite(stage).all() or not np.isin(stage,[0,1,2,3,4]).all():
        raise ValueError('Fig3a needs finite stage codes 0..4; no epoch is removed or bridged.')
    stage=stage.astype(int,copy=True);n=len(stage)
    starts=np.r_[0,np.flatnonzero(np.diff(stage))+1];stops=np.r_[starts[1:],n]
    sleep=np.flatnonzero(stage!=0);begin=int(sleep[0]) if len(sleep) else n
    nrem_min_epochs=int(np.ceil(min_nrem_minutes*60./epoch_seconds))
    rem_min_epochs=int(np.ceil(min_rem_minutes*60./epoch_seconds))
    rows=[]
    for first,last in zip(starts,stops):
        if stage[first]!=4 or last-first<rem_min_epochs:continue
        nrem_epochs=int(np.count_nonzero(np.isin(stage[begin:first],[1,2,3])))
        if nrem_epochs<nrem_min_epochs:continue
        rows.append(dict(cycle_number=len(rows)+1,start_epoch=int(begin),stop_epoch_exclusive=int(last),
            rem_start_epoch=int(first),rem_stop_epoch_exclusive=int(last),
            start_min=begin*epoch_seconds/60.,stop_min=last*epoch_seconds/60.,
            nrem_before_rem_min=nrem_epochs*epoch_seconds/60.,rem_bout_min=(last-first)*epoch_seconds/60.,
            duration_min=(last-begin)*epoch_seconds/60.))
        begin=int(last)
    columns=['cycle_number','start_epoch','stop_epoch_exclusive','rem_start_epoch','rem_stop_epoch_exclusive',
             'start_min','stop_min','nrem_before_rem_min','rem_bout_min','duration_min']
    cycles=pd.DataFrame(rows,columns=columns)
    cycle_index=np.zeros(n,dtype=int);outside=[];cursor=0
    for row in rows:
        first,last=row['start_epoch'],row['stop_epoch_exclusive']
        if first>cursor:outside.append([cursor,first])
        cycle_index[first:last]=row['cycle_number'];cursor=last
    if cursor<n:outside.append([cursor,n])
    source=pd.DataFrame({'epoch_0based':np.arange(n),'time_min':np.arange(n)*epoch_seconds/60.,
                         'stage_code':stage,'cycle_number':cycle_index,'inside_cycle':cycle_index>0})
    metadata={'type':'Recorded hypnogram supplied by user at runtime; no synthetic fallback',
        'hypnogram_key':key.item() if isinstance(key,np.generic) else key,
        'n_epochs':n,'epoch_seconds':float(epoch_seconds),'stage_codes':stage.tolist(),
        'cycle_count':len(cycles),'cycles':rows,'outside_intervals_epochs':outside,
        'min_nrem_minutes':float(min_nrem_minutes),'min_rem_minutes':float(min_rem_minutes),
        'cycle_rule':'First sleep / previous accepted REM end to end of qualifying REM; accumulated NREM; no reset for Wake or short REM',
        'boundary_convention':'0-based epochs, stop exclusive; REM exit epoch not included',
        'time_axis':'Hours from recording start, not inferred lights-off time',
        'classification':'Other panels retain location from supplied occurrence tables; not reclassified here'}
    return {'stages':stage,'cycles':cycles,'source':source,'metadata':metadata}


def _fig3_draw_hypnogram_example(fig,example,W,H,loop_top,font_size,region_colors,extra_height=0.):
    """Plot all epochs and all detected cycles, packing numeric labels if needed."""
    from matplotlib.patches import Rectangle
    fs=lambda n:n*font_size/7.
    ax=fig.add_axes([18/W,(loop_top+38)/H,(W-25)/W,(36.+extra_height)/H],label='inside_outside_hypnogram')
    meta=dict(example['metadata']);stage=example['stages'];seconds=meta['epoch_seconds']
    edges=np.arange(len(stage)+1)*seconds/3600.;mapped=np.array([4,2,1,0,3])[stage]
    ax.set(xlim=(0,float(edges[-1])),ylim=(-.35,6.))
    for first,last in meta['outside_intervals_epochs']:
        ax.add_patch(Rectangle((first*seconds/3600.,-.35),(last-first)*seconds/3600.,4.7,facecolor=region_colors['outside'],alpha=1.,lw=0,zorder=0))
    labels=[]
    for row in example['cycles'].itertuples(index=False):
        first,last=row.start_epoch*seconds/3600.,row.stop_epoch_exclusive*seconds/3600.
        mid=(first+last)/2.
        patch=Rectangle((first,-.35),last-first,4.7,facecolor=region_colors['inside'],alpha=1.,lw=0,zorder=0)
        ax.add_patch(patch)
        patch.set_gid(f'nrem-rem-cycle-{row.cycle_number}')
        ax.plot([first,first,last,last],[4.5,4.7,4.7,4.5],color=region_colors['inside'],lw=.8,zorder=4)
        artist=ax.text(mid,5.25,str(row.cycle_number),ha='center',va='center',fontsize=fs(6.),
                       color=region_colors['inside'],fontweight='bold',gid=f'cycle-number-{row.cycle_number}')
        labels.append(artist)
        # Same Inside fill for every cycle; boundaries identify adjacent cycles.
        ax.plot([last,last],[-.35,4.35],color='white',lw=.55,alpha=.65,ls=(0,(2,2)),zorder=1)
    line,=ax.step(edges,np.r_[mapped,mapped[-1]],where='post',color='white',lw=1.05,zorder=3)
    line.set_gid('recorded-hypnogram')
    # Keep every number. Move neighbouring labels to additional rows as needed.
    fig.canvas.draw();renderer=fig.canvas.get_renderer();lane_ends=[];label_rows=[]
    gap=3.*fig.dpi/72.
    for artist in labels:
        box=artist.get_window_extent(renderer)
        lane=next((i for i,end in enumerate(lane_ends) if box.x0>=end+gap),len(lane_ends))
        if lane==len(lane_ends):lane_ends.append(box.x1)
        else:lane_ends[lane]=box.x1
        artist.set_y(5.25+lane)
        label_rows.append({'cycle_number':int(artist.get_text()),'label_row_0based':lane})
        if lane:
            mid=artist.get_position()[0]
            ax.plot([mid,mid],[4.75,4.95+lane],color=region_colors['inside'],lw=.4,zorder=2)
    n_rows=max(1,len(lane_ends));ax.set_ylim(-.35,5.+n_rows)
    required_mm=(24.+4.8*n_rows)*font_size/10.
    available_mm=(36.+extra_height)/H*fig.get_size_inches()[1]*25.4
    if available_mm+1e-6<required_mm:raise _Fig3LabelSpace(fig,'schematic',required_mm-available_mm)
    if not labels:
        ax.text(.5,.88,'No cycles meet these thresholds',transform=ax.transAxes,ha='center',va='center',fontsize=fs(6.),color=region_colors['inside'])
    ax.set_yticks([4,3,2,1,0],['Wake','REM','N1','N2','N3'])
    ax.set_ylabel('Sleep stage',fontsize=fs(6.),labelpad=3)
    ax.set_xlabel('Time from recording start (h)',fontsize=fs(6.),labelpad=2)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(7))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,p:f'{v:g}'))
    ax.tick_params(axis='both',length=2,width=.5,pad=2,labelsize=fs(5.5))
    ax.spines['left'].set_visible(False)
    meta['cycle_label_rows']=label_rows
    meta['region_fill_colors']=dict(region_colors)
    meta['region_fill_alpha']=1.
    meta['region_fill_y_range']=[-.35,4.35]
    meta['hypnogram_line_color']='#FFFFFF'
    return ax,meta



def prepare_fig3_data(df_loop_occurrence,df_reversible_occurrence,family_counts,NocHypDict,
                      Noc_N2N3_freq,NocAllLoopFlux_N2N3,NocAllSymFlux_N2N3,*,
                      exchange_duration_factor=2.,frequency_factor=2.,min_n_per_group=3,
                      example_hypnogram_key=30,epoch_seconds=30.,min_nrem_minutes=15.,min_rem_minutes=5.):
    """Preserve the supplied events and factors; no new force decomposition."""
    if not np.isfinite(exchange_duration_factor) or exchange_duration_factor<=0:raise ValueError('exchange_duration_factor must be positive.')
    if not np.isfinite(frequency_factor) or frequency_factor<=0:raise ValueError('frequency_factor must be positive.')
    if int(min_n_per_group)!=min_n_per_group or min_n_per_group<2:raise ValueError('min_n_per_group must be an integer >=2.')
    loop,lc=_fig3_occurrences(df_loop_occurrence,'loop',exchange_duration_factor)
    exchange,ec=_fig3_occurrences(df_reversible_occurrence,'exchange',exchange_duration_factor)
    durations={};duration_rows=[]
    for kind,source in [('loop',loop),('exchange',exchange)]:
        durations[kind],row=_fig3_duration_stats(source,kind,int(min_n_per_group));duration_rows.append(row)
    without_n3,bsource=_fig3_no_n3(family_counts)
    example=_fig3_prepare_hypnogram_example(NocHypDict,example_hypnogram_key,epoch_seconds=epoch_seconds,
        min_nrem_minutes=min_nrem_minutes,min_rem_minutes=min_rem_minutes)
    boundaries,episodes=_fig3_n3_boundaries(NocHypDict)
    d={name:_fig3_vector(value,name) for name,value in [('frequency_input',Noc_N2N3_freq),
        ('loop_flux',NocAllLoopFlux_N2N3),('equilibrium',NocAllSymFlux_N2N3)]}
    if len({len(v) for v in d.values()})!=1:raise ValueError(f'D array lengths differ: { {k:len(v) for k,v in d.items()} }')
    ds=pd.DataFrame(d);ds.insert(0,'record_position_0based',np.arange(len(ds)))
    for name in d:
        if np.any(np.isfinite(ds[name])&(ds[name]<0)):raise ValueError(f'D: {name} contains negative values.')
    ds['frequency_per_min']=ds.frequency_input*frequency_factor
    ds['included']=np.isfinite(ds[['frequency_per_min','loop_flux','equilibrium']]).all(axis=1)
    ds['exclusion_reason']=np.where(ds.included,'','non-finite frequency or force')
    included=ds[ds.included]
    if len(included)<3:raise ValueError('D needs at least 3 complete recording triplets.')
    dstat=pd.DataFrame([dict(component=component,**_fig3_pearson(included.frequency_per_min,included[col]))
                       for component,col in [('equilibrium','equilibrium'),('loop_flux','loop_flux')]])
    denominator=ds.equilibrium+ds.loop_flux
    ds['eq_share_defined']=ds.included&(denominator>0)
    ds['equilibrium_share']=np.where(ds.eq_share_defined,ds.equilibrium/denominator.where(denominator>0),np.nan)
    shares=ds.loc[ds.eq_share_defined,'equilibrium_share'].to_numpy(float)
    share={'formula':'DF_EQ^(N2-N3) / (DF_EQ^(N2-N3) + DF_LF^(N2-N3))','n_defined':len(shares),
           'median':float(np.median(shares)) if len(shares) else np.nan,
           'q25':float(np.quantile(shares,.25)) if len(shares) else np.nan,
           'q75':float(np.quantile(shares,.75)) if len(shares) else np.nan}
    return dict(a_example=example,loop_source=loop,exchange_source=exchange,loop_categories=lc,exchange_categories=ec,
                durations=durations,duration_stats=pd.DataFrame(duration_rows),
                duration_rank_tests=_fig3_duration_rank_tests(pd.DataFrame(duration_rows)),without_n3=without_n3,
                family_source=bsource,n3_boundaries=boundaries,n3_episodes=episodes,d_source=ds,d_stats=dstat,
                equilibrium_share=share,metadata={'exchange_duration_factor':float(exchange_duration_factor),
                'exchange_duration_interpretation':'Scaled single-edge unit duration; estimated exchange duration, not measured round-trip time',
                'frequency_factor':float(frequency_factor),'frequency_unit':'per TST minute',
                'A_count_rule':'Every occurrence, including rows without a valid duration',
                'A_percent_rule':'Within each category: inside_count or outside_count divided by their sum; segments sum to 100%',
                'A_region_composition_columns':'Legacy inside_percent/outside_percent describe composition within each region; retained in source CSV, not used as stack labels',
                'B_rule':'Occurrence weighted, full supplied table, no family rotation or intersection',
                'C_cohort':'Nocturnal only, NocHypDict; boundary-censored sides excluded independently',
                'D_cohort':'Nocturnal only; supplied aligned vectors, complete triplets',
                'A_displayed_duration_test':'Two-sided Wilcoxon rank-sum (Mann-Whitney U); two unadjusted pooled-occurrence comparisons',
                'duration_tests':'Original MW, KS and Welch tests; pooled occurrence-level exploratory results, unadjusted',
                'correlation_CI':'95% Fisher-z; assumes independent recordings; no participant IDs supplied'})


def _fig3_rc(font_size):
    q=font_size/7.
    return {'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
      'font.size':7*q,'axes.labelsize':6*q,'xtick.labelsize':5.8*q,'ytick.labelsize':5.8*q,
      'axes.linewidth':.55,'axes.spines.top':False,'axes.spines.right':False,
      'text.color':'#263844','axes.labelcolor':'#263844','xtick.color':'#52606A','ytick.color':'#52606A',
      'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'savefig.facecolor':'white'}


def _fig3_alignment(fig,axes):
    """Rendered geometry gate; categorical rows intentionally adapt to family count."""
    fig.canvas.draw();scale=fig.get_size_inches()*72
    bounds={k:np.array(a.get_position().bounds)*np.array([*scale,*scale]) for k,a in axes.items()}
    errors=[]
    def same(values,name):
        if np.ptp(values)>1.5:errors.append(name)
    for row in ('loop','exchange'):
        first,second=bounds['b_'+row+'_categories'],bounds['b_'+row+'_duration']
        same([first[1],second[1]],row+' bottom');same([first[3],second[3]],row+' height')
    for suffix in ('categories','duration'):
        first,second=bounds['b_loop_'+suffix],bounds['b_exchange_'+suffix]
        same([first[0],second[0]],suffix+' left');same([first[2],second[2]],suffix+' width')
    bottom=[bounds[k] for k in ('c','d','e_equilibrium')]
    for j,name in [(1,'bottom'),(2,'width'),(3,'height')]:same([b[j] for b in bottom],'b-d '+name)
    same([bottom[i+1][0]-bottom[i][0]-bottom[i][2] for i in range(2)],'b-d gutters')
    if not np.allclose(axes['e_equilibrium'].get_position().bounds,axes['e_loop_flux'].get_position().bounds):errors.append('e twin axes')
    renderer=fig.canvas.get_renderer()
    # Full mathematical-label extents include scripts; do not treat internal scripts as different labels.
    dlabels=[axes[k].yaxis.label.get_window_extent(renderer) for k in ('e_equilibrium','e_loop_flux')]
    if dlabels[0].overlaps(dlabels[1]):errors.append('e axis-label overlap')
    for labelbox in dlabels:
        if labelbox.x0<0 or labelbox.y0<0 or labelbox.x1>fig.bbox.x1 or labelbox.y1>fig.bbox.y1:errors.append('e label clipping')
    report={'tolerance_pt':1.5,'bounds_pt':{k:v.tolist() for k,v in bounds.items()},'errors':errors,
      'exemptions':{'a_panel_width':'Category charts intentionally wider than corresponding duration charts',
                    'a_row_height':'Rows adapt to full category counts; no category is removed'}}
    if errors:raise RuntimeError('Fig3 alignment gate failed: '+', '.join(errors))
    return report



class _Fig3LabelSpace(RuntimeError):
    def __init__(self,fig,kind,extra_mm):
        self.figure=fig;self.kind=kind;self.extra_mm=float(extra_mm)
        super().__init__(f'{kind} annotations require {extra_mm:.1f} mm additional row space')


def _fig3_place_stack_labels(fig,ax,table,kind,region_colors,font_size,show_percentages):
    """Label counts and shares within a category; external labels use side leaders.

    Fit is measured with the final renderer. Nothing is sampled or omitted,
    including zero-count components. Only text and axis headroom are adjusted.
    """
    n=len(table)
    if not n:return []
    q=font_size/7.;ink='#263844'
    entries=[];totals=[]
    for j,row in table.iterrows():
        for loc,patch in [('inside',ax.patches[j]),('outside',ax.patches[n+j])]:
            count=int(row[loc+'_count']);pct=float(row[loc+'_within_category_percent'])
            pct_text='NA' if not np.isfinite(pct) else '0%' if pct==0 else '100%' if pct==100 else '<0.1%' if pct<.1 else f'{pct:.1f}%'
            label=f'{count:,}\n{pct_text}' if show_percentages else f'{count:,}'
            artist=ax.annotate(label,(j,patch.get_y()+patch.get_height()/2),xytext=(0,0),
                textcoords='offset points',ha='center',va='center',fontsize=5.5*q,zorder=7,clip_on=False)
            artist.set_gid(f'fig3-stack-{kind}-{loc}-{j}')
            if count==0:artist.set_visible(False)
            entries.append({'kind':kind,'category_position':j,'location':loc,'category_code':row.category_code,
                'category_name':row.category_name,'count':count,'category_total':int(row.total_count),
                'percentage':pct,'denominator_scope':'inside_plus_outside_for_this_category','label':label,
                'artist':artist,'patch':patch})
        top=ax.patches[n+j].get_y()+ax.patches[n+j].get_height()
        artist=ax.annotate(f'n={int(row.total_count):,}',(j,top),xytext=(0,2),textcoords='offset points',
            ha='center',va='bottom',fontsize=5.5*q,color=ink,zorder=7,clip_on=False)
        artist.set_gid(f'fig3-total-{kind}-{j}');totals.append(artist)
    leaders=[]
    for attempt in range(3):
        for line in leaders:line.remove()
        leaders=[]
        # Restore neutral positions before measuring whether each segment can contain its label.
        for entry in entries:
            patch=entry['patch'];entry['artist'].xy=(entry['category_position'],patch.get_y()+patch.get_height()/2)
            entry['artist'].set_position((0,0));entry['artist'].set_va('center')
        fig.canvas.draw();renderer=fig.canvas.get_renderer();pt=fig.dpi/72.
        for j in range(n):
            external=[]
            for entry in [e for e in entries if e['category_position']==j]:
                artist,patch=entry['artist'],entry['patch']
                if entry['count']==0:
                    entry['placement']='zero segment (not labelled)';continue
                box=artist.get_window_extent(renderer);barbox=patch.get_window_extent(renderer)
                fits=(patch.get_height()>0 and barbox.height>=box.height+2*pt and barbox.width>=box.width+1*pt)
                if fits:
                    entry['placement']='inside'
                    artist.set_color('white' if entry['location']=='inside' else ink)
                else:
                    entry['placement']='external';external.append(entry)
                    artist.set_color(region_colors[entry['location']])
            top=totals[j].xy[1];cursor=2.*q
            for entry in external:
                artist=entry['artist'];artist.xy=(j,top);artist.set_va('bottom');artist.set_position((0,cursor))
                box=artist.get_window_extent(renderer);cursor+=box.height/pt+2.*q
            totals[j].set_position((0,cursor+1.*q))
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        all_artists=[e['artist'] for e in entries if e['artist'].get_visible()]+totals
        # Pack external labels in physical coordinates across neighboring columns.
        # The estimates never move; only text offsets and y-axis headroom change.
        fixed=[e['artist'].get_window_extent(renderer).expanded(1.15,1.15)
               for e in entries if e['artist'].get_visible() and e['placement']=='inside']
        placed=[]
        for j in range(n):
            group=[e['artist'] for e in entries if e['category_position']==j
                   and e['placement']=='external']+[totals[j]]
            for _ in range(50):
                # Annotation.get_window_extent updates its positions with this
                # renderer; a full scatter/KDE redraw is unnecessary per move.
                clashes=[(a.get_window_extent(renderer),old) for a in group for old in fixed+placed
                         if a.get_window_extent(renderer).expanded(1.18,1.18).overlaps(old)]
                if not clashes:break
                lift=max(old.y1-box.y0+3*pt for box,old in clashes)
                for artist in group:
                    dx,dy=artist.get_position();artist.set_position((dx,dy+lift/pt))
            placed.extend(a.get_window_extent(renderer).expanded(1.15,1.15) for a in group)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        overflow=max(a.get_window_extent(renderer).y1 for a in all_artists)-ax.get_window_extent(renderer).y1+pt
        if overflow>0:
            if attempt<2:
                lo,hi=ax.get_ylim();ax.set_ylim(lo,hi*1.16);continue
            # Changing ylim cannot create physical space once all labels are external.
            # Ask the enclosing layout to grow this row and re-render every panel.
            raise _Fig3LabelSpace(fig,kind,overflow/fig.dpi*25.4+8*font_size/10.)
        # A side elbow keeps leaders outside the text and all stacked segments.
        for entry in entries:
            if entry['placement']!='external':continue
            box=entry['artist'].get_window_extent(renderer);barbox=entry['patch'].get_window_extent(renderer)
            left=entry['location']=='inside'
            edge=barbox.x0 if left else barbox.x1
            route=min(edge,box.x0)-1.3*pt if left else max(edge,box.x1)+1.3*pt
            target=box.x0-.7*pt if left else box.x1+.7*pt
            mid=(barbox.y0+barbox.y1)/2;labelmid=(box.y0+box.y1)/2
            points=ax.transData.inverted().transform([[edge,mid],[route,mid],[route,labelmid],[target,labelmid]])
            line,=ax.plot(points[:,0],points[:,1],color=region_colors[entry['location']],lw=.4,clip_on=False,zorder=5)
            line.set_gid(f'fig3-leader-{kind}-{entry["location"]}-{entry["category_position"]}')
            leaders.append(line)
        boxes=[a.get_window_extent(renderer) for a in all_artists]
        for i,box in enumerate(boxes):
            if any(box.overlaps(other) for other in boxes[i+1:]):
                raise _Fig3LabelSpace(fig,kind,ax.get_window_extent(renderer).height/fig.dpi*25.4*.5+8)
        return [{key:value for key,value in entry.items() if key not in ('artist','patch')} for entry in entries]
    raise _Fig3LabelSpace(fig,kind,20*font_size/10.)


def _fig3_separate_tick_corners(fig,axes):
    """Resolve x/y tick-corner collisions by measuring, then adding padding."""
    changes=[]
    for key in ('c','d','e_equilibrium'):
        a=axes[key]
        for attempt in range(12):
            fig.canvas.draw();renderer=fig.canvas.get_renderer()
            lo,hi=sorted(a.get_xlim())
            xb=[t.get_window_extent(renderer) for t in a.get_xticklabels()
                if t.get_visible() and lo-1e-10<=t.get_position()[0]<=hi+1e-10]
            yaxes=[a]+([axes['e_loop_flux']] if key=='e_equilibrium' else [])
            yb=[]
            for yaxis in yaxes:
                low,high=sorted(yaxis.get_ylim())
                yb.extend(t.get_window_extent(renderer) for t in yaxis.get_yticklabels()
                          if t.get_visible() and low-1e-10<=t.get_position()[1]<=high+1e-10)
            if not any(x.expanded(1.02,1.08).overlaps(y) for x in xb for y in yb):break
            ticks=a.xaxis.get_major_ticks()
            current=ticks[0].get_pad() if ticks else 6.
            a.tick_params(axis='x',pad=current+2.)
            changes.append({'panel':key,'x_tick_pad_pt':current+2.})
    fig.canvas.draw()
    return changes


def _fig3_text_geometry(fig):
    """Check complete rendered text artists, treating math scripts as one label."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer();items=[];seen=set()
    def keep(t):
        if id(t) in seen or not t.get_visible() or not t.get_text():return
        seen.add(id(t));items.append((t,t.get_window_extent(renderer)))
    for a in fig.axes:
        for t in a.texts:keep(t)
        if not a.axison:continue
        if a.xaxis.get_visible():
            keep(a.xaxis.label)
            lo,hi=sorted(a.get_xlim())
            for t in a.get_xticklabels():
                if lo-1e-10<=t.get_position()[0]<=hi+1e-10:keep(t)
        if a.yaxis.get_visible():
            keep(a.yaxis.label)
            lo,hi=sorted(a.get_ylim())
            for t in a.get_yticklabels():
                if lo-1e-10<=t.get_position()[1]<=hi+1e-10:keep(t)
    for legend in fig.legends:
        for t in legend.get_texts():keep(t)
    errors=[]
    for i,(t,box) in enumerate(items):
        if box.x0<0 or box.y0<0 or box.x1>fig.bbox.x1 or box.y1>fig.bbox.y1:
            errors.append('clipped: '+t.get_text())
        for other,otherbox in items[i+1:]:
            if box.overlaps(otherbox):errors.append(t.get_text()+' / '+other.get_text())
    if errors:raise RuntimeError('Rendered Fig3 text overlaps; enlarge physical canvas or adjust layout: '+'; '.join(errors[:8]))
    return {'result':'PASS','complete_text_artists':len(items),'boxes_pixels':[
        {'text':t.get_text(),'bounds':list(b.bounds)} for t,b in items]}

def _plot_fig3_once(data,*,font_size=10.,fig_width_mm=180.,fig_height_mm=None,b_left_width_mm=120.,transition_pair_label_shift_pt=-3.,b_row_gap_mm=38.,b_to_cde_gap_mm=15.,category_mode='count',
              duration_scale='log',show_points=True,show_regression_ci=True,show_eq_share=True,show_duration_test=True,
              show_category_percentages=True,category_label_rotation=0.,
              random_seed=2026,output_dir='Fig3_output',basename='Fig3_combined',dpi=600,_qa_callback=None,_row_extras=None):
    """One a-e figure. All categories and all valid positive durations are retained.

    A: exact supplied hypnogram with every qualifying NREM-REM cycle numbered.
    B: stacked category counts with within-category inside/outside shares, paired with event duration.
    C: original occurrence-weighted proportions without N3.
    D: original nocturnal N3 boundaries, with one common percentage axis.
    E: original x*2 and supplied forces; no force is recomputed or decomposed.
    """
    if category_mode not in ('percent','count'):raise ValueError('category_mode must be percent or count.')
    if duration_scale not in ('log','linear'):raise ValueError('duration_scale must be log or linear.')
    if not np.isfinite(transition_pair_label_shift_pt):raise ValueError('transition_pair_label_shift_pt must be finite.')
    lc,ec=data['loop_categories'],data['exchange_categories']
    extras={} if _row_extras is None else _row_extras
    loop_h=max(42.,2.7*len(lc))+extras.get('loop',0.)
    exchange_h=max(38.,2.7*len(ec))+extras.get('exchange',0.)
    # Physical row spacing in mm at the default automatic canvas height.
    exchange_bottom=61.+b_to_cde_gap_mm;exchange_top=exchange_bottom+exchange_h
    loop_bottom=exchange_top+b_row_gap_mm;loop_top=loop_bottom+loop_h
    if not np.isfinite(b_left_width_mm) or b_left_width_mm<=0:
        raise ValueError('b_left_width_mm must be positive and finite.')
    count_left=14.;duration_width=27.;column_gap=22.
    duration_left=count_left+b_left_width_mm+column_gap
    duration_centres=[duration_left+duration_width*.25,duration_left+duration_width*.75]
    duration_centre=duration_left+duration_width/2
    W,H=float(fig_width_mm),loop_top+86.+extras.get('schematic',0.)
    if fig_height_mm is None:fig_height_mm=H*max(1.,font_size/10.)
    for name,value in [('font_size',font_size),('fig_width_mm',fig_width_mm),('fig_height_mm',fig_height_mm),('dpi',dpi)]:
        if not np.isfinite(value) or value<=0:raise ValueError(f'{name} must be finite and positive.')
    fs=lambda n:n*font_size/7.
    ink,gray,blue,orange='#263844','#71808A','#0072B2','#D55E00'
    region_colors={'inside':'#496D83','outside':'#879BA9'}
    rng=np.random.default_rng(random_seed);axes={};stack_groups=[];stack_label_rows=[]
    with mpl.rc_context(_fig3_rc(font_size)):
        fig=plt.figure(figsize=(fig_width_mm/25.4,fig_height_mm/25.4),dpi=150,facecolor='white')
        canvas=fig.add_axes([0,0,1,1],label='layout');canvas.set(xlim=(0,W),ylim=(0,H));canvas.axis('off')
        def text(x,y,s,size=7,weight='normal',ha='left',color=ink):
            return canvas.text(x,y,s,fontsize=fs(size),fontweight=weight,ha=ha,va='center',color=color)
        def add(x,y,w,h,key):
            a=fig.add_axes([x/W,y/H,w/W,h/H],label=key);axes[key]=a
            a.tick_params(length=2,width=.5,pad=2,labelsize=fs(5.8));return a
        text(4,H-6,'a',8,'bold')
        region_legend=fig.legend(handles=[Line2D([],[],color=region_colors[loc],lw=3,label=loc.capitalize()) for loc in ('inside','outside')],
                   loc='center',bbox_to_anchor=((W-29)/W,(H-6)/H),ncol=2,fontsize=fs(5.8),handlelength=1.5,columnspacing=1.2)
        for item,loc in zip(region_legend.get_texts(),('inside','outside')):item.set_color(region_colors[loc])
        schematic,schematic_metadata=_fig3_draw_hypnogram_example(
            fig,data['a_example'],W,H,loop_top,font_size,region_colors,extras.get('schematic',0.))
        text(W/2,loop_top+25.5,
             f'{schematic_metadata["cycle_count"]} cycles; NREM ≥{schematic_metadata["min_nrem_minutes"]:g} min; REM ≥{schematic_metadata["min_rem_minutes"]:g} min.',
             5.5,ha='center',color=gray)
        text(4,loop_top+21,'b',8,'bold')
        for kind,table,bottom,height,cat_title,dur_title in [
            ('loop',lc,loop_bottom,loop_h,'Selected loop families','Loop-associated duration'),
            ('exchange',ec,exchange_bottom,exchange_h,'Uncovered transition pairs','Estimated exchange duration')]:
            top=bottom+height
                # Reclaim the previous long-label left margin: x=14, width=106 (was x=42, width=73).
            a=add(count_left,bottom,b_left_width_mm,height,'b_'+kind+'_categories')
            n=[int(table[loc+'_count'].sum()) for loc in ('inside','outside')]
            text(14,top+10,f'Inside: n = {n[0]:,}',5.5,color=region_colors['inside'])
            text(60,top+10,f'Outside: n = {n[1]:,}',5.5,color=region_colors['outside'])
            label_note='Labels: count (% within this category); Inside + Outside = 100%' if show_category_percentages else 'Segment labels: count; column total: n'
            text(14,top+2.4,label_note,5.5,color=gray)
            display_column='within_category_percent' if category_mode=='percent' else 'count'
            inside=table['inside_'+display_column].to_numpy(float)
            outside=table['outside_'+display_column].to_numpy(float)
            a.bar(np.arange(len(table)),np.nan_to_num(inside,nan=0.),width=.88,
                  color=region_colors['inside'],edgecolor='none',zorder=3)
            a.bar(np.arange(len(table)),np.nan_to_num(outside,nan=0.),bottom=np.nan_to_num(inside,nan=0.),width=.88,
                  color=region_colors['outside'],edgecolor='none',zorder=3)
            stack_groups.append((a,table,kind))
            display_labels=table.label.str.replace(' → ','→',regex=False).str.replace(' ↔ ','↔',regex=False)
            if kind=='loop':
                # Keep complete families readable across two/three short, horizontal lines.
                def wrap_family(label):
                    nodes=label.split('→')
                    return '\n'.join(('→' if i else '')+node for i,node in enumerate(nodes))
                display_labels=display_labels.map(wrap_family)
            a.set_xticks(np.arange(len(table)),display_labels,fontsize=fs(5.5),rotation=category_label_rotation,
                         ha='right' if category_label_rotation else 'center',rotation_mode='anchor')
            if kind=='exchange':
                shift=mpl.transforms.ScaledTranslation(float(transition_pair_label_shift_pt)/72.,0.,fig.dpi_scale_trans)
                for tick in a.get_xticklabels():tick.set_transform(tick.get_transform()+shift)
            if len(table):a.set_xlim(-.85,len(table)-.15)
            else:a.text(.5,.5,'No events',ha='center',va='center',transform=a.transAxes,fontsize=fs(6))
            for tick in a.get_xticklabels():
                if tick.get_text() in ('N2↔N3','N3↔N2'):tick.set_fontweight('bold')
            if category_mode=='percent':
                a.set_ylim(0,155);a.set_yticks([0,25,50,75,100]);a.set_ylabel('Within-category share (%)',labelpad=2)
            else:
                maximum=float(table.total_count.max()) if len(table) else 0.
                a.set_ylim(0,max(1.,maximum)*1.55)
                a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4,integer=True));a.set_ylabel('Occurrence count',labelpad=2)
                # Keep large real-data ticks from pushing the ylabel off-page.
                # Bar values and every count annotation remain the exact integers.
                a.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                    lambda value,pos:f'{value/1_000_000:g}M' if value>=1_000_000 else
                    f'{value/1000:g}k' if value>=1000 else f'{value:g}'))
            a.set_xlabel('Loop family' if kind=='loop' else 'Transition pair',labelpad=2)
            a.tick_params(axis='x',length=2,pad=2)
            b=add(duration_left,bottom,duration_width,height,'b_'+kind+'_duration')
            arrays=data['durations'][kind]
            for i,loc in enumerate(('inside','outside')):
                values=arrays[loc];color=region_colors[loc]
                if len(values)>1 and np.ptp(values)>0:
                    if duration_scale=='log':
                        # Density per log-duration unit; all event coordinates stay in minutes.
                        grid=np.linspace(np.log10(values.min()),np.log10(values.max()),120)
                        density=stats.gaussian_kde(np.log10(values))(grid)
                        half_width=.35*density/density.max()
                        b.fill_betweenx(10**grid,i-half_width,i+half_width,color=color,alpha=.27,lw=0)
                    else:
                        violin=b.violinplot([values],positions=[i],widths=.7,showextrema=False,points=100)
                        for body in violin['bodies']:body.set(facecolor=color,edgecolor='none',alpha=.27)
                if len(values):
                    if show_points:
                        b.scatter(i+rng.uniform(-.16,.16,len(values)),values,s=2.0,color=color,alpha=.16,linewidths=0,zorder=2)
                    b.boxplot([values],positions=[i],widths=.18,showfliers=False,manage_ticks=False,
                              boxprops=dict(color=color,linewidth=.7),whiskerprops=dict(color=color,linewidth=.6),
                              capprops=dict(color=color,linewidth=.6),medianprops=dict(color=ink,linewidth=1.))
                    b.scatter([i],[np.mean(values)],s=11,marker='D',color=ink,edgecolors='white',linewidths=.35,zorder=5)
                    mean=f'{np.mean(values):.1f}'
                else:mean='NA'
                text(duration_centres[i],top+11.6,f'n={len(values):,}\nmean\n{mean}',5.5,ha='center',color=color)
            if show_duration_test:
                result_test=data['duration_rank_tests'].set_index('kind').loc[kind]
                text(W-6,top+3,result_test['caption'],7.2,ha='right')
                # Bracket above both groups; neither data nor axis limits are changed.
                canvas.plot([duration_centres[0],duration_centres[0],duration_centres[1],duration_centres[1]],[top-.5,top+.3,top+.3,top-.5],
                            color=ink,lw=.55,solid_capstyle='butt')
            b.set_xlim(-.5,1.5);b.set_xticks([0,1],['Inside','Outside'])
            for tick,loc in zip(b.get_xticklabels(),('inside','outside')):tick.set_color(region_colors[loc])
            pooled=np.concatenate(list(arrays.values()))
            b.set_yscale(duration_scale)
            if len(pooled):
                if duration_scale=='log':b.set_ylim(pooled.min()/1.4,pooled.max()*1.4)
                else:b.set_ylim(0,max(float(pooled.max())*1.12,1.))
            elif duration_scale=='log':b.set_ylim(.1,10)
            b.set_ylabel('Duration (min)',labelpad=2)
            if duration_scale=='log':
                b.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10,numticks=5))
                b.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,p:f'{v:g}'))
                b.yaxis.set_minor_locator(mpl.ticker.NullLocator())
            else:b.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        scale_note='log scale' if duration_scale=='log' else 'linear scale'
        # b: occurrence-weighted proportions, preserving each type's own denominator.
        text(4,61,'c',8,'bold')
        b=add(11,21,32,25,'c');summary=data['without_n3']
        values=summary.percentage.to_numpy(float)
        b.bar([0,1],np.nan_to_num(values,nan=0.),width=.55,color=['#52606A',orange],linewidth=0,zorder=3)
        for i,row in summary.iterrows():
            label=f'{row.percentage:.1f}%\n{row.without_n3_count:,}/\n{row.total_count:,}' if row.total_count else 'NA\n(n = 0)'
            b.annotate(label,(i,float(row.percentage) if row.total_count else 0),xytext=(0,4),textcoords='offset points',
                       ha='center',va='bottom',fontsize=fs(5.5))
        b.set(ylim=(0,118),yticks=[0,25,50,75,100],xticks=[0,1],xticklabels=['Observed\nhypnocycles','Flux\nloops'])
        b.set_ylabel('Without N3 (%)',labelpad=2)
        b.tick_params(axis='x',pad=6)
        # c: original before/after denominators, one shared percentage scale.
        text(56,61,'d',8,'bold')
        c=add(65,21,32,25,'d');boundary=data['n3_boundaries']
        for side,offset,color in [('before',-.19,region_colors['inside']),('after',.19,region_colors['outside'])]:
            part=boundary[boundary.side==side]
            c.bar(np.arange(4)+offset,part.percentage.fillna(0),width=.34,color=color,linewidth=0,zorder=3)
        c.set(ylim=(0,118),yticks=[0,25,50,75,100],xticks=np.arange(4),xticklabels=['W','N1','N2','R'])
        c.set_ylabel('N3 boundary states (%)',labelpad=2);c.set_xlabel('Adjacent stage',labelpad=2)
        c.tick_params(axis='x',pad=6)
        c.get_xticklabels()[2].set_fontweight('bold')
        before=int(boundary.loc[boundary.side=='before','denominator'].iloc[0]);after=int(boundary.loc[boundary.side=='after','denominator'].iloc[0])
        text(81,56.7,f'Before: n = {before:,}; after: n = {after:,}',5.5,ha='center',color=gray)
        boundary_legend=fig.legend(handles=[Line2D([],[],color=region_colors[side],lw=3,label=label) for side,label in [('inside','Before N3'),('outside','After N3')]],
                   loc='center',bbox_to_anchor=(81/W,51.5/H),ncol=2,fontsize=fs(5.5),handlelength=1.,columnspacing=1.)
        for item,loc in zip(boundary_legend.get_texts(),('inside','outside')):item.set_color(region_colors[loc])
        c.grid(axis='y',color='#E7ECEF',lw=.4,zorder=0);c.set_axisbelow(True)
        # d: blue left equilibrium, orange right loop flux, matching Fig. 2.
        text(110,61,'e',8,'bold')
        ds=data['d_source'];part=ds[ds.included];x=part.frequency_per_min.to_numpy(float)
        left=add(119,21,32,25,'e_equilibrium');right=left.twinx();right.set_label('e_loop_flux');right.patch.set_visible(False)
        axes['e_loop_flux']=right
        for component,a,color,ylabel,ypos in [
            ('equilibrium',left,blue,r'EQ ($\mathrm{DF}_{\mathrm{EQ}}^{\mathrm{N2-N3}}$)',56.7),
            ('loop_flux',right,orange,r'LF ($\mathrm{DF}_{\mathrm{LF}}^{\mathrm{N2-N3}}$)',52.6)]:
            y=part[component].to_numpy(float);yr=float(np.ptp(y));pad=.1*yr if yr else max(float(y.max())*.1,.001)
            ylim=(max(0.,float(y.min())-pad),float(y.max())+pad);a.set_ylim(ylim)
            a.scatter(x,y,s=5.5,color=color,alpha=.48,linewidths=0,zorder=3)
            if np.ptp(x)>0:
                grid=np.linspace(x.min(),x.max(),160);slope,intercept=np.polyfit(x,y,1);pred=intercept+slope*grid
                valid=(pred>=ylim[0])&(pred<=ylim[1]);a.plot(grid[valid],pred[valid],color=color,lw=1.05,zorder=4)
                if show_regression_ci and len(x)>2:
                    variance=np.sum((y-(intercept+slope*x))**2)/(len(x)-2)
                    half=stats.t.ppf(.975,len(x)-2)*np.sqrt(variance*(1/len(x)+(grid-x.mean())**2/np.sum((x-x.mean())**2)))
                    a.fill_between(grid,np.clip(pred-half,*ylim),np.clip(pred+half,*ylim),color=color,alpha=.055,lw=0,zorder=1)
            a.set_ylabel(ylabel,color=color,fontsize=fs(7.2),labelpad=1)
            a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            a.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value,pos:f'{value:.3g}'))
            a.tick_params(axis='y',colors=color,labelsize=fs(5.8),length=2,width=.5,pad=2)
            a.spines['left' if component=='equilibrium' else 'right'].set_edgecolor(color)
            rc=data['d_stats'].set_index('component').loc[component]
            caption=f'r = {rc.pearson_r:.2f}; n = {int(rc.n)}'
            text(119,ypos,caption,5.5,color=color)
        right.spines['left'].set_visible(False);right.spines['bottom'].set_visible(False)
        right.spines['top'].set_visible(False);right.spines['right'].set_visible(True)
        pad=.05*np.ptp(x) if np.ptp(x) else max(abs(float(x[0]))*.1,.001)
        left.set_xlim(max(0.,float(x.min())-pad),float(x.max())+pad)
        left.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        left.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,p:f'{v:.3g}'))
        left.set_xlabel('N2–N3 transition frequency (/min)',labelpad=2)
        left.tick_params(axis='x',pad=6)
        if show_eq_share:
            median=data['equilibrium_share']['median']
            text(119,48.5,f'EQ share: median {median:.1%}' if np.isfinite(median) else 'EQ share: undefined',5.5,color=gray)
        for category_ax,table,kind in stack_groups:
            stack_label_rows.extend(_fig3_place_stack_labels(fig,category_ax,table,kind,region_colors,font_size,show_category_percentages))
        tick_corner_layout=_fig3_separate_tick_corners(fig,axes)
        alignment=_fig3_alignment(fig,axes)
        result={'figure':fig,'axes':axes,'data':data,'alignment':alignment,'paths':{},
                'category_stack_labels':pd.DataFrame(stack_label_rows),'schematic_axis':schematic,'schematic_metadata':schematic_metadata,'tick_corner_layout':tick_corner_layout,
                'a_cycle_table':data['a_example']['cycles'].copy(),'a_hypnogram_source':data['a_example']['source'].copy()}
        if _qa_callback is not None:_qa_callback(result)
        result['text_geometry']=_fig3_text_geometry(fig)
        if output_dir is not None:
            output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
            for suffix in ('pdf','svg','png'):
                path=output/f'{basename}.{suffix}'
                fig.savefig(path,dpi=dpi,facecolor='white')
                result['paths'][suffix]=str(path.resolve())
            for key in ('loop_source','exchange_source','loop_categories','exchange_categories','duration_stats',
                        'duration_rank_tests','without_n3','family_source','n3_boundaries','n3_episodes','d_source','d_stats'):
                data[key].to_csv(output/f'{basename}_{key}.csv',index=False)
            (output/f'{basename}_text_geometry.json').write_text(json.dumps(result['text_geometry'],indent=2),encoding='utf-8')
            result['category_stack_labels'].to_csv(output/f'{basename}_category_stack_labels.csv',index=False)
            result['a_cycle_table'].to_csv(output/f'{basename}_a_nremrem_cycles.csv',index=False)
            result['a_hypnogram_source'].to_csv(output/f'{basename}_a_hypnogram_source.csv',index=False)
            metadata=dict(data['metadata'],tick_corner_layout=tick_corner_layout,category_mode=category_mode,duration_scale=duration_scale,
                          duration_density='KDE in log10 minutes' if duration_scale=='log' else 'KDE in minutes',
                          show_points=show_points,show_regression_ci=show_regression_ci,show_eq_share=show_eq_share,show_duration_test=show_duration_test,show_category_percentages=show_category_percentages,
                          category_label_rotation=category_label_rotation,category_orientation='vertical stacked',
                          percentage_annotation_rule='Within each category: segment count / (inside_count + outside_count); two segments sum to 100%',
                          plot_size_mm=(fig.get_size_inches()*25.4).tolist(),font_size_pt=font_size,dpi=dpi,
                          b_left_width_mm=b_left_width_mm,transition_pair_label_shift_pt=transition_pair_label_shift_pt,panel_map={'a':'recorded hypnogram, all qualifying NREM-REM cycles numbered','b':'inside/outside counts and durations','c':'without N3','d':'N3 boundaries','e':'N2-N3 dynamics'},point_jitter_seed=random_seed,inside_outside_schematic=schematic_metadata,equilibrium_share=data['equilibrium_share'])
            (output/f'{basename}_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
            (output/f'{basename}_alignment.json').write_text(json.dumps(alignment,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def plot_fig3(data,*,font_size=10.,fig_width_mm=180.,fig_height_mm=None,b_left_width_mm=120.,transition_pair_label_shift_pt=-3.,b_row_gap_mm=38.,b_to_cde_gap_mm=15.,category_mode='count',
              duration_scale='log',show_points=True,show_regression_ci=True,show_eq_share=True,show_duration_test=True,
              show_category_percentages=True,category_label_rotation=0.,random_seed=2026,
              output_dir='Fig3_output',basename='Fig3_combined',dpi=600,_qa_callback=None):
    """Render all data with fixed font/width and physically adaptive category rows.

    fig_width_mm is the minimum canvas width; b_left_width_mm sets the actual
    width of both panel-b count axes. The canvas expands to preserve the right column.
    b_row_gap_mm controls the gap between panel-b plot areas.
    b_to_cde_gap_mm controls the gap from the lower panel-b plot bottom to
    the c/d/e panel-label row. Defaults reclaim 24 mm of vertical whitespace.
    fig_height_mm is the starting/minimum height. When labels need extra room,
    the affected row and total height grow together; font size never shrinks.
    Metadata records requested and actual dimensions plus all expansion steps.
    """
    for name,value in [('b_row_gap_mm',b_row_gap_mm),('b_to_cde_gap_mm',b_to_cde_gap_mm)]:
        if not np.isfinite(value) or value<=0:
            raise ValueError(name+' must be positive and finite.')
    requested_width_mm=fig_width_mm
    if not np.isfinite(b_left_width_mm) or b_left_width_mm<=0:
        raise ValueError('b_left_width_mm must be positive and finite.')
    # Keep right-column size and the font intact instead of squeezing them.
    fig_width_mm=max(float(fig_width_mm),14.+b_left_width_mm+22.+27.+7.)
    options=dict(font_size=font_size,fig_width_mm=fig_width_mm,b_left_width_mm=b_left_width_mm,transition_pair_label_shift_pt=transition_pair_label_shift_pt,b_row_gap_mm=b_row_gap_mm,b_to_cde_gap_mm=b_to_cde_gap_mm,category_mode=category_mode,
        duration_scale=duration_scale,show_points=show_points,show_regression_ci=show_regression_ci,
        show_eq_share=show_eq_share,show_duration_test=show_duration_test,
        show_category_percentages=show_category_percentages,category_label_rotation=category_label_rotation,
        random_seed=random_seed,output_dir=output_dir,basename=basename,dpi=dpi,_qa_callback=_qa_callback)
    extras={'loop':0.,'exchange':0.,'schematic':0.};history=[];active_height=fig_height_mm
    if fig_height_mm is not None:
        if not np.isfinite(fig_height_mm) or fig_height_mm<=0:
            raise ValueError('fig_height_mm must be positive and finite, or None.')
        base_units=(61.+b_to_cde_gap_mm+max(38.,2.7*len(data['exchange_categories']))+b_row_gap_mm
                    +max(42.,2.7*len(data['loop_categories']))+86.)
        minimum_height=base_units*max(1.,font_size/10.)
        active_height=max(float(fig_height_mm),minimum_height)
        if active_height>fig_height_mm:
            print(f'Fig3: preserving minimum layout height {active_height:.1f} mm for the requested font.')
    for attempt in range(8):
        try:
            result=_plot_fig3_once(data,fig_height_mm=active_height,_row_extras=extras,**options)
            report={'requested_height_mm':fig_height_mm,'actual_height_mm':float(result['figure'].get_size_inches()[1]*25.4),
                    'font_size_unchanged':font_size,'requested_width_mm':requested_width_mm,'actual_width_mm':fig_width_mm,'b_left_width_mm':b_left_width_mm,'b_row_gap_mm':b_row_gap_mm,'b_to_cde_gap_mm':b_to_cde_gap_mm,
                    'row_extra_layout_units':extras.copy(),'expansions':history}
            result['label_layout']=report
            if output_dir is not None:
                path=Path(output_dir)/f'{basename}_label_layout.json'
                path.write_text(json.dumps(report,indent=2),encoding='utf-8')
            if history:print('Fig3 annotation layout expanded automatically:',report)
            return result
        except _Fig3LabelSpace as need:
            fig=need.figure
            current_height=float(fig.get_size_inches()[1]*25.4)
            layout=next(a for a in fig.axes if a.get_label()=='layout')
            unit_mm=current_height/float(layout.get_ylim()[1])
            growth=max(need.extra_mm,12.*font_size/10.)
            extras[need.kind]+=growth/unit_mm
            history.append({'row':need.kind,'added_height_mm':growth})
            print(f'Fig3: expanding {need.kind} annotation row by {growth:.1f} mm; font and chosen column widths unchanged.')
            active_height=current_height+growth
            plt.close(fig)
    raise RuntimeError('Fig3 label layout could not converge after physical row expansion. '+
        'Please provide the category tables for further layout adaptation; no observations were removed.')


# ======================================================================
# Definitions from cleaned notebook cell 58 (zero-based)
# ======================================================================

# Cell 1: def. Dependencies: numpy, pandas, scipy, matplotlib.
from pathlib import Path
from collections.abc import Mapping
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

_FIG4_METRICS = {
    'SOL':('mean_sleep_latency',None,'SOL (min)','greater'),
    'Hypnocycle':('mean_loop_frequency','mean_loop_count','Frequency (/min)','less'),
    'Awakening':('mean_wake_frequency_after_sleep','mean_wake_count_after_sleep','Frequency (/min)','less'),
    'Transition':('mean_state_switch_frequency_after_sleep','mean_state_switch_count_after_sleep','Frequency (/min)','less'),
    'NREM-REM':('mean_nrem_rem_cycle_frequency','mean_nrem_rem_cycle_count','Frequency (/min)','less'),
}


def _fig4_tst_mapping(frame,tst,cohort):
    ids=frame['index'].drop_duplicates().tolist()
    if isinstance(tst,Mapping):
        missing=[x for x in ids if x not in tst]
        if missing:raise KeyError(f'{cohort}: TST mapping lacks IDs {missing[:8]}')
        mapping={x:float(tst[x]) for x in ids};rule='explicit keyed mapping'
    else:
        a=np.asarray(tst,dtype=float)
        if a.ndim==2 and 1 in a.shape:a=a.reshape(-1)
        if a.ndim!=1 or not len(a):raise ValueError(f'{cohort} TST must be a 1D minute vector or ID mapping.')
        direct=False
        try:
            numeric=np.asarray(ids);integer=numeric.astype(int)
            direct=bool(np.all(numeric==integer) and np.all(integer>=0) and np.all(integer<len(a)))
        except (ValueError,TypeError):pass
        if direct:
            mapping={x:float(a[int(x)]) for x in ids};rule='integer index directly indexes TST'
        elif len(ids)==len(a):
            mapping={x:float(a[j]) for j,x in enumerate(ids)};rule='original first-appearance fallback'
            warnings.warn(f'{cohort}: TST mapped in first-appearance ID order, matching your original rule. Check exported tst_mapping.csv or pass an explicit dict.',stacklevel=2)
        else:raise ValueError(f'{cohort}: cannot align {len(ids)} IDs with TST length {len(a)}; supply a keyed TST dict.')
    if any(not np.isfinite(v) or v<=0 for v in mapping.values()):raise ValueError(f'{cohort}: mapped TST must be positive finite minutes.')
    return mapping,pd.DataFrame([{'dataset':cohort,'index':x,'TST_min':value,'mapping_rule':rule} for x,value in mapping.items()])


def _fig4_input_table(frame,tst,cohort):
    if not isinstance(frame,pd.DataFrame):raise TypeError(f'{cohort}: expected a DataFrame of recording-by-k simulation means.')
    if {'index','k'}-set(frame):raise KeyError(f'{cohort}: missing index or k column.')
    d=frame.copy(deep=True).reset_index(drop=True)
    if not len(d) or d['index'].isna().any():raise ValueError(f'{cohort}: input is empty or contains missing recording IDs.')
    d['fig4_source_row_0based']=np.arange(len(d));d['k']=pd.to_numeric(d.k,errors='coerce')
    if np.any(~np.isfinite(d.k)) or np.any(d.k<0):raise ValueError(f'{cohort}: k must be finite and non-negative.')
    if d.duplicated(['index','k']).any():raise ValueError(f'{cohort}: multiple rows per index,k. Supply already aggregated simulation means, not individual replicate rows.')
    if len(d['index'].drop_duplicates())!=len(d['index'].astype(str).drop_duplicates()):raise ValueError(f'{cohort}: recording IDs collide after string conversion.')
    mapping,map_table=_fig4_tst_mapping(d,tst,cohort)
    d['TST']=d['index'].map(mapping).astype(float);d['dataset']=cohort
    d['hypnogram_id']=cohort+'_'+d['index'].astype(str)
    sources={}
    for metric,(key,count,unit,alternative) in _FIG4_METRICS.items():
        for prefix in ('orig','new'):
            column=prefix+'_'+key
            if count is not None and prefix+'_'+count in d:
                value=pd.to_numeric(d[prefix+'_'+count],errors='coerce')
                d[column]=value/d.TST;sources[column]='supplied mean count / recorded TST minutes'
            elif column in d:
                d[column]=pd.to_numeric(d[column],errors='coerce')
                sources[column]='supplied minutes' if metric=='SOL' else 'supplied frequency, assumed /min'
            else:raise KeyError(f'{cohort}: need {column}'+(f' or {prefix}_{count}' if count else ''))
            if np.any(np.isfinite(d[column])&(d[column]<0)):raise ValueError(f'{cohort}.{column}: negative values are invalid.')
    return d,map_table,sources


def _fig4_wilcoxon(x,y,alternative):
    """Retain original paired test: finite pairs, wilcox zeros, allclose-zero => p=1."""
    x=np.asarray(x,float);y=np.asarray(y,float);keep=np.isfinite(x)&np.isfinite(y);x=x[keep];y=y[keep]
    out={'n_pairs':len(x),'W':np.nan,'p_value':np.nan,'alternative':alternative}
    if len(x)<2:return dict(out,status='insufficient pairs')
    if np.allclose(x-y,0):return dict(out,W=0.,p_value=1.,status='all differences numerically zero (original rule)')
    result=stats.wilcoxon(x,y,alternative=alternative,zero_method='wilcox',method='auto')
    return dict(out,W=float(result.statistic),p_value=float(result.pvalue),status='tested')


def _fig4_relative(orig,new):
    orig=np.asarray(orig,float);new=np.asarray(new,float)
    out=np.full(orig.shape,np.nan);valid=np.isfinite(orig)&np.isfinite(new)&(orig>0)
    out[valid]=100*(new[valid]-orig[valid])/orig[valid]
    return out


def _fig4_resample_indices(groups,n_boot,rng):
    groups=np.asarray(groups)
    pieces=[]
    for cohort in pd.unique(groups):
        positions=np.flatnonzero(groups==cohort)
        pieces.append(positions[rng.integers(0,len(positions),size=(n_boot,len(positions)))])
    return np.concatenate(pieces,axis=1)


def _fig4_ci(values,n_boot,seed,groups=None,orig=None,absolute=False):
    """Recording-level paired/stratified percentile bootstrap, no simulation-level resampling."""
    values=np.asarray(values,float)
    if orig is None:valid=np.isfinite(values)
    else:
        orig=np.asarray(orig,float);valid=np.isfinite(values)&np.isfinite(orig)
    values=values[valid]
    if orig is not None:orig=orig[valid]
    groups=np.asarray(groups)[valid] if groups is not None else np.repeat('all',len(values))
    if len(values)<2 or n_boot<=0:return np.nan,np.nan,0
    indices=_fig4_resample_indices(groups,int(n_boot),np.random.default_rng(seed))
    if orig is None:samples=values[indices].mean(axis=1)
    else:
        denominator=orig[indices].mean(axis=1);numerator=values[indices].mean(axis=1)
        samples=np.full(len(indices),np.nan);ok=denominator>0
        samples[ok]=100*(numerator[ok]-denominator[ok])/denominator[ok]
    if absolute:samples=np.abs(samples)
    samples=samples[np.isfinite(samples)]
    if not len(samples):return np.nan,np.nan,0
    lo,hi=np.quantile(samples,[.025,.975]);return float(lo),float(hi),len(samples)


def _fig4_static_rows(all_rows,k_static):
    """Restore the original sorted groupby.first, including first-nonmissing column behavior."""
    d=all_rows.copy()
    if k_static is None:d=d.sort_values(['hypnogram_id','k'])
    else:
        if not np.isfinite(k_static):raise ValueError('k_static must be None or finite.')
        d['fig4_k_distance']=abs(d.k-float(k_static))
        d=d.sort_values(['hypnogram_id','fig4_k_distance'])
    selected=d.groupby('hypnogram_id',as_index=False).first()
    selected=selected.drop(columns=['fig4_k_distance'],errors='ignore')
    selected['fig4_selected_static_k']=selected.k
    checks=[]
    for metric,(key,_,_,_) in _FIG4_METRICS.items():
        if metric=='NREM-REM':continue
        for prefix in ('orig','new'):
            column=prefix+'_'+key
            finite=all_rows[['hypnogram_id',column]].copy();finite[column]=finite[column].where(np.isfinite(finite[column]))
            summary=finite.groupby('hypnogram_id',sort=False)[column].agg(['min','max','count'])
            for identifier,row in summary.iterrows():
                spread=row['max']-row['min']
                checks.append({'hypnogram_id':identifier,'column':column,'n_finite':int(row['count']),
                  'min':row['min'],'max':row['max'],'spread':spread,
                  'varies_across_k':bool(np.isfinite(spread) and not np.isclose(spread,0,rtol=0,atol=1e-12))})
    return selected.reset_index(drop=True),pd.DataFrame(checks)


def _fig4_neglog_p(p):
    if not np.isfinite(p):return np.nan
    return float(-np.log10(p)) if p>0 else np.inf


def prepare_fig4(Nap_df_subject_k,Noc_df_subject_k,NapTST,NocTST,*,k_static=None,n_boot=2000,random_seed=2026):
    """Use supplied simulation means. No new simulations, repeated averaging or flux decomposition.

    A retains the original changes of paired means. All magnitude comparisons
    and the four B curves use the supplied original absolute-change definition.
    TST and both simulated SOL columns are minutes, confirmed by the user.
    """
    if int(n_boot)!=n_boot or n_boot<0:raise ValueError('n_boot must be a non-negative integer.')
    tables=[];maps=[];frequency_sources={}
    for cohort,frame,tst in [('Nap',Nap_df_subject_k,NapTST),('Noc',Noc_df_subject_k,NocTST)]:
        d,mapping,sources=_fig4_input_table(frame,tst,cohort)
        tables.append(d);maps.append(mapping);frequency_sources[cohort]=sources
    all_rows=pd.concat(tables,ignore_index=True);static,variation=_fig4_static_rows(all_rows,k_static)
    a_source=[];a_stats=[]
    for j,metric in enumerate(('SOL','Hypnocycle','Awakening','Transition')):
        key,_,unit,alternative=_FIG4_METRICS[metric]
        d=static[['hypnogram_id','dataset','index','fig4_selected_static_k','orig_'+key,'new_'+key]].copy()
        d=d.rename(columns={'orig_'+key:'original','new_'+key:'flux_removed'})
        d['metric']=metric;d['unit']='min' if metric=='SOL' else '/min'
        d['included']=np.isfinite(d.original)&np.isfinite(d.flux_removed)
        d['exclusion_reason']=np.where(d.included,'','non-finite paired value')
        d['relative_change_percent']=_fig4_relative(d.original,d.flux_removed)
        d['signed_reduction_percent']=-d.relative_change_percent
        a_source.append(d)
        keep=d[d.included];test=_fig4_wilcoxon(keep.flux_removed,keep.original,alternative)
        mean_orig=float(keep.original.mean());mean_new=float(keep.flux_removed.mean())
        pct=100*(mean_new-mean_orig)/mean_orig if mean_orig>0 else np.nan
        lo,hi,nb=_fig4_ci(keep.flux_removed,n_boot,random_seed+j,keep.dataset,orig=keep.original,absolute=metric!='SOL')
        a_stats.append(dict(metric=metric,mean_original=mean_orig,mean_flux_removed=mean_new,
            change_ratio_of_means_percent=pct,displayed_change_percent=pct if metric=='SOL' else abs(pct),
            change_ci_low=lo,change_ci_high=hi,bootstrap_valid=nb,
            bootstrap_unit='recording, paired and stratified by cohort',**test))
    a_source=pd.concat(a_source,ignore_index=True);a_stats=pd.DataFrame(a_stats)
    # Original definition: absolute paired percent change, not signed reduction.
    static_magnitude=static[['hypnogram_id','dataset','index','fig4_selected_static_k']].copy()
    for metric in ('Hypnocycle','Awakening','Transition'):
        key=_FIG4_METRICS[metric][0]
        static_magnitude[metric]=np.abs(_fig4_relative(static['orig_'+key],static['new_'+key]))
    a_contrasts=[]
    for j,(first,second) in enumerate([('Hypnocycle','Transition'),('Hypnocycle','Awakening'),('Awakening','Transition')]):
        d=static_magnitude[[first,second,'dataset']];valid=np.isfinite(d[first])&np.isfinite(d[second]);d=d[valid]
        delta=(d[first]-d[second]).to_numpy(float)
        lo,hi,nb=_fig4_ci(delta,n_boot,random_seed+20+j,d.dataset)
        test=_fig4_wilcoxon(d[first],d[second],'greater')
        a_contrasts.append(dict(first=first,second=second,mean_magnitude_first=float(d[first].mean()),
            mean_magnitude_second=float(d[second].mean()),mean_difference_pp=float(np.mean(delta)) if len(delta) else np.nan,
            ci_low=lo,ci_high=hi,bootstrap_valid=nb,**test))
    noc_static=static_magnitude[static_magnitude.dataset=='Noc'].set_index('hypnogram_id')
    b_sources=[];wide=[];tests=[]
    key=_FIG4_METRICS['NREM-REM'][0]
    for k,dfk in all_rows[all_rows.dataset=='Noc'].groupby('k',sort=True):
        d=dfk[['hypnogram_id','dataset','index','k','orig_'+key,'new_'+key]].copy().rename(columns={'orig_'+key:'original','new_'+key:'flux_removed'})
        d['relative_change_percent']=_fig4_relative(d.original,d.flux_removed)
        d['absolute_change_percent']=abs(d.relative_change_percent)
        d['magnitude_defined']=np.isfinite(d.absolute_change_percent)
        d['exclusion_reason']=np.where(d.magnitude_defined,'',np.where(~np.isfinite(d.original)|~np.isfinite(d.flux_removed),
            'non-finite pair','original baseline is zero; relative change undefined'))
        for metric in ('Hypnocycle','Awakening','Transition'):d[metric]=d.hypnogram_id.map(noc_static[metric])
        frequency=_fig4_wilcoxon(d.flux_removed,d.original,'less')
        pair=d[np.isfinite(d.original)&np.isfinite(d.flux_removed)];mo,mn=pair.original.mean(),pair.flux_removed.mean()
        row={'k':float(k),'n_pairs_frequency':frequency['n_pairs'],'n_pairs_reduction_magnitude':len(d),
             'mean_noneq_nrem_rem_cycle_frequency':mo,'mean_eq_nrem_rem_cycle_frequency':mn,
             'reduction_magnitude_nrem_percent':abs(100*(mn-mo)/mo) if mo>0 else np.nan,
             'p_nrem_EQ_less_NonEQ':frequency['p_value'],'neglogp_nrem_EQ_less_NonEQ':_fig4_neglog_p(frequency['p_value']),
             'n_original_zero':int((d.original==0).sum()),
             'n_zero_original_positive_removed':int(((d.original==0)&(d.flux_removed>0)).sum())}
        tests.append(dict(k=float(k),test_key='frequency',comparison='NREM-REM frequency: flux removed < original',
            quantity='raw paired NREM-REM frequency',neglog10_p=_fig4_neglog_p(frequency['p_value']),**frequency))
        for metric,internal in [('Transition','transition'),('Awakening','awakening'),('Hypnocycle','microcycle')]:
            test=_fig4_wilcoxon(d.absolute_change_percent,d[metric],'greater')
            row['p_red_nrem_greater_'+internal]=test['p_value']
            row['neglogp_red_nrem_greater_'+internal]=_fig4_neglog_p(test['p_value'])
            row['n_pairs_'+internal]=test['n_pairs']
            tests.append(dict(k=float(k),test_key=internal,comparison='NREM-REM magnitude > '+metric,
                quantity='absolute paired percent change',neglog10_p=_fig4_neglog_p(test['p_value']),**test))
        wide.append(row);b_sources.append(d)
    b_source=pd.concat(b_sources,ignore_index=True)
    metadata={'A_cohorts':['Nap','Noc'],'B_cohort':'Noc only','SOL_unit':'minutes in both simulation tables',
      'TST_unit':'recorded TST minutes; same supplied denominator used for original and flux-removed counts',
      'frequency_sources':frequency_sources,'k_static':k_static,
      'static_selection':'Original sorted groupby.first: first nonmissing value per column; missing columns may originate at different k',
      'A_change_definition':'100 * (mean(flux_removed) - mean(original)) / mean(original); frequency labels use its absolute value, matching original',
      'A_magnitude_contrasts':'Original per-record abs(100 * (flux_removed-original) / original), paired and one-sided greater',
      'B_primary_frequency_test':'Paired Wilcoxon: raw flux-removed NREM-REM frequency < original frequency',
      'B_three_magnitude_tests':'Original absolute percent-change magnitude of NREM-REM > Transition / Awakening / Hypnocycle',
      'B_y_axis':'-log10(p), all four original tests; not effect size or additional signed reduction',
      'zero_baseline':'Relative changes undefined when original=0; retained in raw source/frequency tests, excluded only from magnitude comparisons',
      'n_boot':int(n_boot),'random_seed':random_seed,
      'CI':('Not calculated: n_boot=0; no confidence intervals requested' if n_boot==0 else 'A only: 95% percentile bootstrap over recording-level pairs, stratified by cohort; magnitude intervals bootstrap the absolute statistic'),
      'test':'Original one-sided paired Wilcoxon signed-rank, wilcox zero handling and original allclose-zero rule',
      'p_adjustment':'none; all k-wise tests are unadjusted',
      'input_level':'Already supplied recording-by-k simulation means; no further averaging over simulation replicates',
      'model_design':'Stationary occupancies fixed by manuscript construction; not re-verified from summary tables',
      'participant_clustering':'No independent participant mapping supplied; recording-level inference only',
      'canonical_k5_present':bool(np.any(np.isclose(b_source.k,5.,rtol=0,atol=1e-9)))}
    return dict(all_rows=all_rows,static_rows=static,static_variation=variation,tst_mapping=pd.concat(maps,ignore_index=True),
        a_source=a_source,a_stats=a_stats,a_contrasts=pd.DataFrame(a_contrasts),static_magnitude=static_magnitude,
        b_source=b_source,p_by_k=pd.DataFrame(wide).sort_values('k').reset_index(drop=True),
        b_tests=pd.DataFrame(tests),metadata=metadata)


def _fig4_rc(font_size):
    q=font_size/7.
    resolved_font=mpl.font_manager.FontProperties(fname=mpl.font_manager.findfont(
        mpl.font_manager.FontProperties(family=['Arial','Helvetica','DejaVu Sans']))).get_name()
    return {'font.weight':'normal','axes.labelweight':'normal','text.usetex':False,
      'mathtext.fontset':'dejavusans','mathtext.default':'it','font.family':resolved_font,'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
      'font.size':7*q,'axes.labelsize':6*q,'xtick.labelsize':5.5*q,'ytick.labelsize':5.5*q,
      'axes.linewidth':.55,'axes.spines.top':False,'axes.spines.right':False,
      'axes.labelcolor':'#263844','text.color':'#263844','xtick.color':'#52606A','ytick.color':'#52606A',
      'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'savefig.facecolor':'white'}


def _fig4_ptext(p,with_stars=True):
    """Rendered scientific notation with a real superscript, never a visible e-10 string."""
    if not np.isfinite(p):return r'$p = \mathrm{NA}$'
    star='***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
    if p==0:body=r'$p \approx 0$'
    else:
        coefficient,exponent=f'{p:.2e}'.split('e');comparison='='
        rounded=float(f'{p:.2e}')
        # Do not round across a significance boundary and display an inconsistent star.
        if rounded in (.001,.01,.05) and p!=rounded:comparison='<' if p<rounded else '>'
        body=rf'$p {comparison} {coefficient}\times 10^{{{int(exponent)}}}$'
    return body+(' ('+star+')' if with_stars else '')


def _fig4_top_scale(ax,values,requested):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if requested not in ('auto','linear','log','symlog'):raise ValueError('top_scale must be auto, linear, log or symlog.')
    if not len(values):ax.set_ylim(0,1);return 'linear'
    mode=('symlog' if np.any(values==0) else 'log') if requested=='auto' else requested
    if mode=='log' and np.any(values<=0):raise ValueError('Log scale cannot display zero values. Use auto or symlog; no zero is silently removed.')
    positives=values[values>0]
    if mode=='symlog':ax.set_yscale('symlog',linthresh=float(positives.min()/2) if len(positives) else 1.,linscale=.65)
    else:ax.set_yscale(mode)
    transform=ax.yaxis.get_transform();scores=transform.transform(values)
    lo,hi=float(scores.min()),float(scores.max());span=hi-lo
    if span<=0:span=max(abs(hi)*.15,.5 if mode=='log' else 1.)
    lower=lo-.08*span;upper=hi+.12*span
    bounds=transform.inverted().transform([lower,upper]);bounds[0]=max(0.,bounds[0]) if mode!='log' else bounds[0]
    ax.set_ylim(bounds)
    if mode=='linear':ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    else:
        powers=np.array([])
        if len(positives):
            exponents=np.arange(int(np.floor(np.log10(positives.min()))),int(np.ceil(np.log10(positives.max())))+1)
            powers=10.**exponents;powers=powers[(powers>=bounds[0])&(powers<=bounds[1])]
        if len(powers)>3:powers=powers[np.unique(np.linspace(0,len(powers)-1,3).astype(int))]
        if len(powers)<2:
            positions=np.linspace(transform.transform([bounds[0]])[0],transform.transform([bounds[1]])[0],3)
            powers=transform.inverted().transform(positions)
        ticks=np.r_[0.,powers[powers>0]] if mode=='symlog' else powers
        ax.set_yticks(ticks);ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,p:f'{v:.3g}'))
    return mode


def _fig4_alignment(fig,axes):
    fig.canvas.draw();scale=fig.get_size_inches()*72
    bounds={k:np.array(a.get_position().bounds)*np.array([*scale,*scale]) for k,a in axes.items()}
    groups=[['b_SOL','b_Hypnocycle','b_Awakening','b_Transition']]
    errors=[]
    for group in groups:
        items=[bounds[key] for key in group]
        for pos,name in [(1,'bottom'),(2,'width'),(3,'height')]:
            if np.ptp([box[pos] for box in items])>1.5:errors.append(str(group)+' '+name)
        gaps=[items[i+1][0]-items[i][0]-items[i][2] for i in range(len(items)-1)]
        if len(gaps)>1 and np.ptp(gaps)>1.5:errors.append(str(group)+' gutters')
    span=bounds['c_pvalues'];first=bounds['b_SOL'];last=bounds['b_Transition']
    if abs(span[0]-first[0])>1.5 or abs(span[0]+span[2]-last[0]-last[2])>1.5:errors.append('C spanning outer edges')
    renderer=fig.canvas.get_renderer();boxes=[]
    for key,a in axes.items():
        if a.yaxis.label.get_text():
            box=a.yaxis.label.get_window_extent(renderer);boxes.append((key,box))
            if box.x0<0 or box.y0<0 or box.x1>fig.bbox.x1 or box.y1>fig.bbox.y1:errors.append(key+' label clipping')
    for i,(key,box) in enumerate(boxes):
        for other,second in boxes[i+1:]:
            if box.overlaps(second):errors.append(key+' / '+other+' label overlap')
    report={'bounds_pt':{k:b.tolist() for k,b in bounds.items()},'groups':groups,'tolerance_pt':1.5,'errors':errors,
      'exemptions':{'b_vs_c':'Four equal paired-condition panels above one full-width panel of the original four tests'}}
    if errors:raise RuntimeError('Fig4 rendered alignment gate failed: '+', '.join(errors))
    return report



def _fig4_flux_removal_schematic(fig,total_height,font_size):
    """Direct before/after probability-flow comparison; T reconstruction is secondary."""
    from matplotlib.patches import Circle,FancyArrowPatch,Rectangle
    from matplotlib.path import Path as MplPath
    q=font_size/7.;ink='#263844';eq='#0072B2';flux='#D55E00';diagonal='#65747F'
    ax=fig.add_axes([4/180.,147/total_height,172/180.,80/total_height],label='flux_removal_schematic')
    ax.set(xlim=(0,172),ylim=(0,80));ax.set_aspect('equal',adjustable='box');ax.axis('off')
    def label(x,y,value,size=6.,color=ink,weight='normal',gid=None):
        artist=ax.text(x,y,value,ha='center',va='center',fontsize=size*q,color=color,fontweight=weight)
        if gid:artist.set_gid(gid)
        return artist
    def arrow(first,second,color,lw=.8,style='-|>',rad=0.,gid=None):
        patch=FancyArrowPatch(first,second,arrowstyle=style,mutation_scale=7.*q,linewidth=lw,
            color=color,connectionstyle=f'arc3,rad={rad}',zorder=2)
        if gid:patch.set_gid(gid)
        ax.add_patch(patch);return patch
    ax.text(0,77,'a',ha='left',va='center',fontsize=8.*q,fontweight='bold',color=ink,gid='panel-letter-a')
    def coloured_label(cx,y,parts):
        # Measure complete text objects so inline colours stay centred without overlaps.
        fragments=[label(cx,y,value,6.3,color=color,weight='bold') for value,color in parts]
        class InlineLabelLayout(mpl.artist.Artist):
            def draw(self,renderer):
                widths=[artist.get_window_extent(renderer).width for artist in fragments]
                gap=fig.dpi/72.
                cursor=ax.transData.transform((cx,y))[0]-(sum(widths)+gap*(len(parts)-1))/2
                for artist,width in zip(fragments,widths):
                    xx=ax.transData.inverted().transform((cursor,ax.transData.transform((cx,y))[1]))[0]
                    artist.set_ha('left');artist.set_position((xx,y));cursor+=width+gap
                self.stale=False
        positioner=InlineLabelLayout();positioner.set_zorder(2.9);ax.add_artist(positioner)
    coloured_label(37,68,[('Before: ',ink),('symmetric flow',eq),(' + ',ink),('loop flux',flux)])
    coloured_label(135,68,[('After: ',ink),('symmetric flow',eq),(' + ',ink),('added self-flow',diagonal)])
    node_radius=4.3
    relative={'Wake':np.array([-21.,50.]),'N1':np.array([0.,58.]),'N2':np.array([21.,50.]),
              'N3':np.array([13.,33.]),'REM':np.array([-13.,33.])}
    # Examples already present among the manuscript's loop families. Shared edges
    # have the same direction; no bidirectional excess-flow pair is invented.
    cycles=[['Wake','N1','N2','Wake'],['N1','N2','REM','N1'],['Wake','N1','N2','REM','Wake']]
    directed=list(dict.fromkeys((first,second) for cycle in cycles for first,second in zip(cycle[:-1],cycle[1:])))
    # N2-N3 is purely symmetric; REM->Wake is purely directed in this example.
    equilibrium_edges=[edge for edge in directed if edge!=('REM','Wake')]+[('N2','N3')]
    compensated_stages=list(dict.fromkeys(first for first,second in directed))
    networks={}
    def endpoints(first,second):
        vector=second-first;unit=vector/np.linalg.norm(vector)
        return first+unit*(node_radius+.3),second-unit*(node_radius+.3)
    for model,cx in [('original',37.),('reversible',135.)]:
        positions={stage:point+np.array([cx,0.]) for stage,point in relative.items()};networks[model]=positions
        for first,second in equilibrium_edges:
            start,stop=endpoints(positions[first],positions[second])
            arrow(start,stop,eq,lw=.9,style='<->',gid=f'exchange-{model}-{first}-{second}')
        if model=='original':
            for first,second in directed:
                start,stop=endpoints(positions[first],positions[second])
                arrow(start,stop,flux,lw=1.5,rad=-.20,gid=f'flux-{first}-{second}')
        else:
            # Only stages with outgoing loop flux receive diagonal compensation.
            # N3 has zero illustrated J, so its added diagonal flow is also zero.
            centre=np.array([cx,45.])
            for stage in compensated_stages:
                point=positions[stage]
                direction=point-centre;direction=direction/np.linalg.norm(direction)
                tangent=np.array([-direction[1],direction[0]])
                start=point+node_radius*(.55*direction+.835*tangent)
                stop=point+node_radius*(.55*direction-.835*tangent)
                control1=point+(node_radius+4.)*direction+3.5*tangent
                control2=point+(node_radius+4.)*direction-3.5*tangent
                path=MplPath([start,control1,control2,stop],[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4])
                patch=FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=7.*q,lw=1.,color=diagonal,zorder=2)
                patch.set_gid('added-self-flow-'+stage);ax.add_patch(patch)
        for stage,point in positions.items():
            circle=Circle(point,node_radius,facecolor='white',edgecolor='#71808A',lw=.65,zorder=4)
            circle.set_gid(f'node-{model}-{stage}');ax.add_patch(circle)
            label(*point,stage,5.5,gid=f'label-{model}-{stage}').set_zorder(5)
    # Flow is the primary quantity on BOTH sides; the same blue edges and
    # stage coordinates make removal of every orange edge directly visible.
    label(86,56,'Remove all loop flux',6.,color=flux,weight='bold')
    arrow((65,50),(107,50),flux,lw=1.5,gid='remove-all-flux-arrow')
    label(86,43,r"$J'=0$",7.2,color=flux,gid='formula-flux-after-zero')
    # label(86,37,'All loops in C',5.5,color=flux)
    label(37,21,r'$Q^{ss}=Q^{ss,\mathrm{sym}}+J^{ss}$',7.2,gid='formula-flow-before')
    label(135,21,r"$Q'=Q^{ss,\mathrm{sym}}+\mathrm{diag}(J^{ss}\mathbf{1})$",7.2,gid='formula-flow-after')
    label(86,9.5,r"$Q'_{ii}=Q_{ii}^{ss}+\sum_j J_{ij}^{ss}$"+'; stationary occupancies preserved',7.2,gid='formula-diagonal')
    label(86,0.,'All loops removed; representative paths shown. '+r"$T'_{ij}=Q'_{ij}/P_i^{ss}$",7.2,gid='formula-reconstruction')
    metadata={'scope':'Direct before/after flow comparison on identical five-stage networks',
      'removal_scope':'ALL decomposed loops in C, not only the paths drawn',
      'loop_flux_definition':'Jss = sum(c in C) alpha_c chi_c',
      'flow_before':'Qss = Qss_sym + Jss',
      'flow_after':'Qprime = Qss_sym + diag(Jss @ ones); Jprime = 0',
      'primary_quantity':'Probability flow Q on both sides; Tprime reconstruction is secondary',
      'nodes_original':{k:v.tolist() for k,v in networks['original'].items()},
      'nodes_reversible':{k:v.tolist() for k,v in networks['reversible'].items()},
      'equilibrium_edges':equilibrium_edges,'aggregate_flux_edges':directed,'illustrative_flux_cycles':cycles,
      'added_self_flow_stages':compensated_stages,'equilibrium_only_pair':['N2','N3'],'pure_directed_pair':['REM','Wake'],
      'fixed':'Off-diagonal symmetric flow and the stationary occupancy vector',
      'diagonal':"Qprime_ii = Pss_i - sum(j != i) Qprime_ij = Q_ii + sum(j) J_ij",
      'reconstruction':'Tprime_ij = Qprime_ij / Pss_i',
      'not_encoded':'No numerical probabilities or empirical transition matrices; representative paths are not an exhaustive loop list'}
    return ax,metadata


def _fig4_text_geometry(fig,strict=False):
    """Check complete rendered text artists, treating math scripts as one label."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer();items=[];seen=set()
    def keep(t):
        if id(t) in seen or not t.get_visible() or not t.get_text():return
        seen.add(id(t));items.append((t,t.get_window_extent(renderer)))
    for a in fig.axes:
        for t in a.texts:keep(t)
        if not a.axison:continue
        if a.xaxis.get_visible():
            keep(a.xaxis.label)
            lo,hi=sorted(a.get_xlim())
            for t in a.get_xticklabels():
                if lo-1e-10<=t.get_position()[0]<=hi+1e-10:keep(t)
        if a.yaxis.get_visible():
            keep(a.yaxis.label)
            lo,hi=sorted(a.get_ylim())
            for t in a.get_yticklabels():
                if lo-1e-10<=t.get_position()[1]<=hi+1e-10:keep(t)
    for legend in fig.legends:
        for t in legend.get_texts():keep(t)
    errors=[]
    for i,(t,box) in enumerate(items):
        if box.x0<0 or box.y0<0 or box.x1>fig.bbox.x1 or box.y1>fig.bbox.y1:
            errors.append('clipped: '+t.get_text())
        for other,otherbox in items[i+1:]:
            if box.overlaps(otherbox):errors.append(t.get_text()+' / '+other.get_text())
    if errors and strict:raise RuntimeError('Rendered Fig4 text overlaps: '+'; '.join(errors[:8]))
    return {'result':'NEEDS_REVIEW' if errors else 'PASS','errors':errors,'complete_text_artists':len(items),'boxes_pixels':[
        {'text':t.get_text(),'bounds':list(b.bounds)} for t,b in items]}


def _fig4_fit_top_panels(fig,axes,p_labels,contrast_heading):
    """Fit vertical decorations between the p-value band and contrast band.

    Only the four plot rectangles are shortened, equally. Font sizes, data,
    scales, statistics, x positions and physical canvas width are unchanged.
    """
    keys=['b_SOL','b_Hypnocycle','b_Awakening','b_Transition']
    adjustments=[]
    for attempt in range(6):
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        upper=min(t.get_window_extent(renderer).y0 for t in p_labels)-3.*fig.dpi/72.
        lower=contrast_heading.get_window_extent(renderer).y1+3.*fig.dpi/72.
        high=low=0.
        for key in keys:
            axis=axes[key];lo,hi=sorted(axis.get_ylim())
            texts=[axis.yaxis.label]+[t for t in axis.get_yticklabels() if lo<=t.get_position()[1]<=hi]
            for artist in texts:
                box=artist.get_window_extent(renderer)
                high=max(high,box.y1-upper);low=max(low,lower-box.y0)
        if high<=0.5 and low<=0.5:break
        if high>0.5:
            shrink=2.*high/fig.bbox.height
            current=axes[keys[0]].get_position().height
            minimum=8./(fig.get_size_inches()[1]*25.4)
            shrink=min(shrink,max(0.,current-minimum))
            if shrink<=0:break
            for key in keys:
                bounds=axes[key].get_position();axes[key].set_position([bounds.x0,bounds.y0,bounds.width,bounds.height-shrink])
            adjustments.append({'shortened_plot_height_mm':float(shrink*fig.get_size_inches()[1]*25.4)})
        else:break
    return {'adjustments':adjustments,'font_sizes_unchanged':True,
            'final_plot_height_mm':float(axes[keys[0]].get_position().height*fig.get_size_inches()[1]*25.4)}


def _fig4_threshold_notes(ax,threshold,fs,color):
    """Keep both reference notes inside C, above the horizontal p=.05 line.

    X is an axes fraction; Y follows the actual -log10(p) threshold. Point
    offsets preserve readable spacing when the data change the y-axis range.
    """
    criterion=ax.annotate('k = 5: ≥15 min NREM, ≥5 min REM',
        xy=(.98,threshold),xycoords=ax.get_yaxis_transform(),
        xytext=(0,5),textcoords='offset points',ha='right',va='bottom',
        fontsize=fs(5.5),color=color,annotation_clip=False)
    criterion.set_gid('criterion-note-inside-c')
    fig=ax.figure;fig.canvas.draw()
    height_pt=criterion.get_window_extent(fig.canvas.get_renderer()).height*72./fig.dpi
    significance=ax.annotate('Dashed line: '+_fig4_ptext(.05,with_stars=False),
        xy=(.98,threshold),xycoords=ax.get_yaxis_transform(),
        xytext=(0,8+height_pt),textcoords='offset points',ha='right',va='bottom',
        fontsize=fs(7.2),color=color,annotation_clip=False)
    significance.set_gid('p-threshold-note-inside-c')
    # With weak evidence the upper limit may be near the threshold itself.
    # Add display headroom only; raw p values and plotted curve coordinates stay fixed.
    for attempt in range(8):
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        if significance.get_window_extent(renderer).y1+3.*fig.dpi/72.<=ax.bbox.y1:break
        lo,hi=ax.get_ylim();ax.set_ylim(lo,hi*1.25)
    return (significance,criterion)


def plot_fig4(data,*,font_size=10.,fig_width_mm=180.,fig_height_mm=250.,top_scale='auto',show_ci=False,p_plot_floor=1e-300,
              random_seed=2026,output_dir='Fig4_output',basename='Fig4_combined',dpi=600,strict_layout=False,_qa_callback=None):
    """One figure: A flow-removal schematic, B paired plots, C original paired-test p curves."""
    if fig_height_mm is None:fig_height_mm=250.
    for name,value in [('font_size',font_size),('fig_width_mm',fig_width_mm),('fig_height_mm',fig_height_mm),('dpi',dpi)]:
        if not np.isfinite(value) or value<=0:raise ValueError(f'{name} must be positive and finite.')
    if not np.isfinite(p_plot_floor) or not 0<p_plot_floor<1:raise ValueError('p_plot_floor must be between 0 and 1; display only.')
    q=font_size/7.;fs=lambda x:x*q;W,H=180.,230.
    ink,gray,original,removed='#263844','#71808A','#596873','#0072B2'
    test_colors={'frequency':'#765AA6','transition':'#D55E00','awakening':'#699E98','microcycle':'#60839E'}
    axes={};scales={};p_labels=[];rng=np.random.default_rng(random_seed)
    with mpl.rc_context(_fig4_rc(font_size)):
        fig=plt.figure(figsize=(fig_width_mm/25.4,fig_height_mm/25.4),dpi=150,facecolor='white')
        canvas=fig.add_axes([0,0,1,1],label='layout');canvas.set(xlim=(0,W),ylim=(0,H));canvas.axis('off')
        def text(x,y,s,size=7,weight='normal',ha='left',color=ink):
            return canvas.text(x,y,s,fontsize=fs(size),fontweight=weight,ha=ha,va='center',color=color)
        def add(x,y,w,h,key):
            a=fig.add_axes([x/W,y/H,w/W,h/H],label=key);axes[key]=a
            a.tick_params(length=2,width=.5,pad=2,labelsize=fs(5.5));return a
        schematic_axis,schematic_metadata=_fig4_flux_removal_schematic(fig,H,font_size)
        text(4,139,'b',8,'bold')
        handles=[Line2D([],[],color=original,lw=1.2,label='Original'),Line2D([],[],color=removed,lw=1.2,label='Flux removed'),
          Line2D([],[],color=ink,marker='o',lw=0,markersize=2.7,label='Nocturnal'),
          Line2D([],[],color=ink,marker='^',lw=0,markersize=2.7,label='Nap')]
        fig.legend(handles=handles,loc='center',bbox_to_anchor=(133/W,139/H),ncol=4,fontsize=fs(5.5),
                   handlelength=1.3,handletextpad=.35,columnspacing=.8)
        for j,metric in enumerate(('SOL','Hypnocycle','Awakening','Transition')):
            x0=14+41*j;a=add(x0,91,24,14,'b_'+metric)
            source=data['a_source'];part=source[(source.metric==metric)&source.included].copy()
            row=data['a_stats'].set_index('metric').loc[metric]
            text(x0,126,f'n = {len(part)}',5.5,color=gray)
            change=row.displayed_change_percent
            caption=(f'Change: {change:+.1f}%' if metric=='SOL' else f'|Change|: {change:.1f}%') if np.isfinite(change) else 'Change: undefined'
            text(x0,121,caption,6.,'bold',color=removed)
            ci=f'95% CI [{row.change_ci_low:.1f}, {row.change_ci_high:.1f}]%' if np.isfinite(row.change_ci_low) else '95% CI: NA'
            if show_ci:text(x0,136,ci,5.5,color=gray)  # Optional legacy switch; disabled in the run cell.
            p_labels.append(text(x0,116,_fig4_ptext(row.p_value),7.2))
            values=part[['original','flux_removed']].to_numpy(float)
            scales[metric]=_fig4_top_scale(a,values.ravel(),top_scale)
            if len(part):
                jitter=rng.uniform(-.045,.045,len(part))
                # Every visible line links two conditions from the same recording, using the same jitter.
                for i in range(len(part)):
                    a.plot([jitter[i],1+jitter[i]],values[i],color='#A9B5BD',lw=.35,alpha=.20,zorder=1)
                transform=a.yaxis.get_transform()
                for c,color,column in [(0,original,'original'),(1,removed,'flux_removed')]:
                    y=part[column].to_numpy(float);scores=transform.transform(y)
                    if len(y)>1 and np.ptp(scores)>0:
                        grid=np.linspace(scores.min(),scores.max(),120);density=stats.gaussian_kde(scores)(grid)
                        width=.18*density/density.max()
                        a.fill_betweenx(transform.inverted().transform(grid),c-width,c+width,color=color,alpha=.17,lw=0,zorder=0)
                    for cohort,marker in [('Noc','o'),('Nap','^')]:
                        keep=part.dataset.to_numpy()==cohort
                        a.scatter(c+jitter[keep],y[keep],s=4.0,marker=marker,color=color,alpha=.48,linewidths=0,zorder=3)
                    a.plot([c-.12,c+.12],[y.mean(),y.mean()],color=ink,lw=1.1,zorder=5)
            else:a.text(.5,.5,'No finite pairs',ha='center',va='center',transform=a.transAxes,fontsize=fs(6))
            a.set(xlim=(-.35,1.35),xticks=[0,1],xticklabels=['Original','Flux\nremoved'])
            a.set_ylabel(_FIG4_METRICS[metric][2] if metric=='SOL' else metric+'\nfrequency (/min)',labelpad=2)
        # Compact paired contrasts replace crossing brackets spanning the four metrics.
        contrast_heading=None
        for x0,row in zip((14,70,125),data['a_contrasts'].itertuples(index=False)):
            comparison_caption=text(x0,76,f'{row.first} − {row.second}',5.5,'bold')
            if contrast_heading is None:contrast_heading=comparison_caption
            text(x0,71,_fig4_ptext(row.p_value),7.2,color=gray)
        text(4,62,'c',8,'bold')
        pcurves=add(14,17,147,28,'c_pvalues')
        curve_source=data['b_tests'].copy(deep=True)
        curve_source['plot_neglog10_p']=np.nan
        valid=np.isfinite(curve_source.p_value)
        curve_source.loc[valid,'plot_neglog10_p']=-np.log10(np.maximum(curve_source.loc[valid,'p_value'].to_numpy(float),p_plot_floor))
        curve_source['display_censored']=valid&(curve_source.p_value<p_plot_floor)
        labels={'frequency':'NREM–REM frequency: flux removed < original',
                'transition':'Magnitude: NREM–REM > transition',
                'awakening':'Magnitude: NREM–REM > awakening',
                'microcycle':'Magnitude: NREM–REM > hypnocycle'}
        for key in ('frequency','transition','awakening','microcycle'):
            part=curve_source[curve_source.test_key==key].sort_values('k');color=test_colors[key]
            pcurves.plot(part.k,part.plot_neglog10_p,color=color,lw=1.4 if key=='frequency' else 1.,zorder=4 if key=='frequency' else 3)
            clipped=part[part.display_censored]
            if len(clipped):pcurves.scatter(clipped.k,clipped.plot_neglog10_p,marker='^',s=10,color=color,zorder=5)
        handles=[Line2D([],[],color=test_colors[key],lw=1.4 if key=='frequency' else 1.,label=labels[key])
                 for key in ('frequency','transition','awakening','microcycle')]
        fig.legend(handles=handles,loc='center',bbox_to_anchor=(89/W,50.5/H),ncol=2,fontsize=fs(5.5),
                   handlelength=1.7,columnspacing=2.,labelspacing=.4)
        threshold=-np.log10(.05)
        pcurves.axhline(threshold,color='#89959F',lw=.65,ls=(0,(3,2)),zorder=0)
        pcurves.set_xlim(float(curve_source.k.min()),max(5.15,float(curve_source.k.max())))
        finite=curve_source.plot_neglog10_p.to_numpy(float);finite=finite[np.isfinite(finite)]
        pcurves.set_ylim(0,max(2.,float(finite.max())*1.10 if len(finite) else 2.))
        pcurves.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6,integer=True))
        pcurves.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        pcurves.set_xlabel('Minimum REM duration, k (min)',labelpad=2)
        pcurves.set_ylabel(r'$-\log_{10}(p)$',fontsize=fs(7.2),labelpad=2)
        threshold_notes=_fig4_threshold_notes(pcurves,threshold,fs,gray)
        note='↑: values below display p floor; raw p retained.' if curve_source.display_censored.any() else ''
        text(90,7.5,note,5.5,ha='center',color=gray)
        layout_adjustments=_fig4_fit_top_panels(fig,axes,p_labels,contrast_heading)
        # Thin only redundant tick labels on short plots; observations and limits stay unchanged.
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for axis in axes.values():
            lo,hi=sorted(axis.get_ylim())
            ticks=[v for v in axis.get_yticks() if lo<=v<=hi]
            if len(ticks)<3:continue
            labels=[t for t in axis.get_yticklabels() if lo<=t.get_position()[1]<=hi]
            boxes=[t.get_window_extent(renderer) for t in labels]
            keep=[0]
            for index in range(1,len(ticks)-1):
                if boxes[index].y0>=boxes[keep[-1]].y1+3 and boxes[index].y1+3<=boxes[-1].y0:keep.append(index)
            keep.append(len(ticks)-1)
            axis.set_yticks([ticks[i] for i in keep])
        alignment=_fig4_alignment(fig,axes)
        result={'figure':fig,'axes':axes,'data':data,'alignment':alignment,'top_scales':scales,'paths':{},'p_curve_plot_source':curve_source,'schematic_axis':schematic_axis,'schematic_metadata':schematic_metadata}
        if _qa_callback is not None:_qa_callback(result)
        result['text_geometry']=_fig4_text_geometry(fig,strict=strict_layout)
        result['layout_adjustments']=layout_adjustments
        result['threshold_notes']=threshold_notes
        if result['text_geometry']['errors']:
            import warnings
            warnings.warn('Fig4 exported with layout review notes; see text_geometry.json: '+ '; '.join(result['text_geometry']['errors'][:4]),stacklevel=2)
        if output_dir is not None:
            out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
            for fmt in ('pdf','svg','png'):
                path=out/f'{basename}.{fmt}';fig.savefig(path,dpi=dpi,facecolor='white');result['paths'][fmt]=str(path.resolve())
            for key,value in data.items():
                if isinstance(value,pd.DataFrame):value.to_csv(out/f'{basename}_{key}.csv',index=False)
            curve_source.to_csv(out/f'{basename}_p_curve_plot_source.csv',index=False)
            metadata=dict(data['metadata'],font_size_pt=font_size,plot_size_mm=(fig.get_size_inches()*25.4).tolist(),
                          top_scale_policy=top_scale,actual_top_scales=scales,dpi=dpi,p_plot_floor=p_plot_floor,
                          flux_removal_schematic=schematic_metadata,show_confidence_intervals=bool(show_ci),layout_adjustments=layout_adjustments,
                          panel_mapping={'a':'flux-removal schematic','b':'pooled paired metrics','c':'nocturnal p curves'},
                          show_pp_annotations=False,show_k5_vertical_line=False,threshold_notes_location='Inside C, above p=.05 horizontal line',
                          p_display='Scientific notation with superscript exponent, three significant figures; raw full-precision p in CSV',paired_lines='one per recording, identical horizontal jitter for both conditions',
                          top_mean_lines='Arithmetic means of raw values; not means of log-transformed values',
                          plot_layout='One figure: A flow-removal schematic; B four paired plots; C one full-width -log10(p) panel with all four original comparisons')
            (out/f'{basename}_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
            (out/f'{basename}_alignment.json').write_text(json.dumps(alignment,ensure_ascii=False,indent=2),encoding='utf-8')
            (out/f'{basename}_text_geometry.json').write_text(json.dumps(result['text_geometry'],ensure_ascii=False,indent=2),encoding='utf-8')
    return result


# ======================================================================
# Definitions from cleaned notebook cell 62 (zero-based)
# ======================================================================

# Cell 1: def. Dependencies: numpy, matplotlib. No existing helper defs required.
from pathlib import Path
from collections.abc import Mapping
import csv
import json
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _sif1_matrices(data, name, matrix_axes='auto'):
    """Every supplied 5x5 matrix has equal weight; no reconstruction of flow."""
    records = []
    def walk(item, label):
        if isinstance(item, Mapping):
            if not item:
                raise ValueError(label + ' is empty.')
            for k, value in item.items():
                walk(value, f'{label}[{k!r}]')
            return
        try:
            a = np.asarray(item, dtype=float)
        except (ValueError, TypeError):
            if isinstance(item, (list, tuple)) and len(item):
                for k, value in enumerate(item):
                    walk(value, f'{label}[{k}]')
                return
            raise ValueError(label + ': expected a numeric matrix collection.') from None
        if a.shape == (5, 5):
            records.append(a.copy())
            return
        if a.ndim >= 3:
            if isinstance(matrix_axes, str) and matrix_axes == 'auto':
                axes = [(k, k+1) for k in range(a.ndim-1) if a.shape[k:k+2] == (5,5)]
                if len(axes) != 1:
                    raise ValueError(f'{label}: ambiguous matrix axes in {a.shape}; set matrix_axes explicitly.')
                axes = axes[0]
            else:
                if isinstance(matrix_axes, str) or len(matrix_axes) != 2:
                    raise ValueError("matrix_axes must be 'auto' or two integer axes.")
                axes = tuple(int(k) % a.ndim for k in matrix_axes)
                if axes[0] == axes[1] or any(a.shape[k] != 5 for k in axes):
                    raise ValueError(f'{label}: axes {axes} do not select a 5x5 matrix.')
            records.extend(np.moveaxis(a, axes, (-2,-1)).reshape(-1,5,5).copy())
            return
        raise ValueError(f'{label}: expected (5,5), (n,5,5), or (5,5,n); got {a.shape}.')
    walk(data, name)
    if not records:
        raise ValueError(name + ' contains no matrices.')
    stack = np.stack(records)
    _sif1_probability_values(stack, name)
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    mean = np.divide(np.where(finite, stack, 0).sum(axis=0), counts,
                     out=np.full((5,5), np.nan), where=counts>0)
    if not np.isfinite(mean).any():
        raise ValueError(name + ' has no finite entries.')
    return stack, mean, counts


def _sif1_probability_values(a, name):
    if np.isinf(a).any():
        raise ValueError(name + ' contains infinity.')
    finite = a[np.isfinite(a)]
    if np.any((finite < -1e-10) | (finite > 1+1e-10)):
        raise ValueError(name + ' must contain probabilities in [0,1], or NaN.')


def _sif1_probabilities(data, name):
    if isinstance(data, Mapping):
        a = np.stack([np.asarray(v, float).reshape(-1) for v in data.values()])
    else:
        a = np.asarray(data, dtype=float)
    if a.ndim == 1 and a.size == 5:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != 5 or a.shape[0] == 0:
        raise ValueError(f'{name}: expected (n,5) or (5,), got {a.shape}.')
    _sif1_probability_values(a, name)
    good = np.isfinite(a)
    n = good.sum(axis=0)
    mean = np.divide(np.where(good, a, 0).sum(axis=0), n,
                     out=np.full(5, np.nan), where=n>0)
    if not np.all(np.isfinite(mean)):
        raise ValueError(name + ': at least one stage has no finite observations.')
    return a.copy(), mean, n


def _sif1_relative_difference(pss, empirical):
    # Exactly the user's original metric, simplified algebraically.
    denominator = (pss + empirical) / 2
    terms = np.divide(np.abs(pss-empirical), denominator,
                      out=np.full(5, np.nan), where=denominator>0)
    return float(np.nanmean(terms)) if np.isfinite(terms).any() else np.nan, terms


def plot_si_fig1(NocTmat, NapTmat, NocFS, NapFS,
                 NocPss, NocProbs, NapPss, NapProbs, *,
                 font_size=10., panel_label_size=12., fig_width_mm=180., fig_height_mm=252.,
                 stage_labels=('Wake','N1','N2','N3','REM'),
                 t_matrix_axes='auto', q_matrix_axes='auto',
                 cmap='Blues', show_matrix_values=False, show_relative_difference=True,
                 empirical_color='#52606A', stationary_color='#009E9A',
                 output_dir='SI_Fig1_output', basename='SI_Fig1', dpi=600, show=True):
    """Three aligned rows: mean T, mean supplied Q, and mean occupancy comparison.

    Column order: Nocturnal, Nap. Input stage order: Wake,N1,N2,N3,REM.
    Matrix and occupancy means use all finite entries, exactly like np.nanmean.
    Missing entries are not filled; valid counts are exported. No rows are
    renormalized. C connects two cohort means at the same stage, not two people.
    Its relative difference is calculated AFTER averaging, not per recording.
    A 1D occupancy input is treated as an already supplied vector (not n=1).
    No significance tests, equivalence claims, or uncertainty estimates are added.
    """
    for name, value in [('font_size',font_size),('panel_label_size',panel_label_size),
                        ('fig_width_mm',fig_width_mm),('fig_height_mm',fig_height_mm),('dpi',dpi)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive.')
    if len(stage_labels) != 5:
        raise ValueError('stage_labels must contain five labels in input order.')
    names = ('Nocturnal','Nap')
    datasets = {'Nocturnal':(NocTmat,NocFS,NocPss,NocProbs),
                'Nap':(NapTmat,NapFS,NapPss,NapProbs)}
    summaries, matrix_rows, occupancy_rows, error_rows, diagnostics = {}, [], [], [], []
    for cohort,(tin,qin,pssin,empin) in datasets.items():
        ts,t,nt = _sif1_matrices(tin,cohort+'.T',t_matrix_axes)
        qs,q,nq = _sif1_matrices(qin,cohort+'.Q',q_matrix_axes)
        ps,p,nps = _sif1_probabilities(pssin,cohort+'.Pss')
        es,e,nes = _sif1_probabilities(empin,cohort+'.Empirical')
        error, terms = _sif1_relative_difference(p,e)
        summaries[cohort] = dict(T=t,Q=q,Pss=p,Empirical=e,n_T=len(ts),n_Q=len(qs),
                                 relative_difference=error,relative_difference_terms=terms)
        if len(ts) != len(qs):
            diagnostics.append(f'{cohort}: T has {len(ts)} matrices, Q has {len(qs)}; averaged independently.')
        for kind, stack, mean, counts in [('T',ts,t,nt),('Q',qs,q,nq)]:
            finite = np.isfinite(stack)
            diagnostics.append(f'{cohort}.{kind}: {int((~finite).sum())} missing entries; cellwise finite means, no imputation.')
            if kind == 'T':
                sums = stack.sum(axis=2)
                valid_rows = finite.all(axis=2)
                bad = valid_rows & ~(np.isclose(sums,1,atol=1e-5) | np.isclose(sums,0,atol=1e-10))
                if bad.any():
                    warnings.warn(f'{cohort}.T: {bad.sum()} complete rows do not sum to 0 or 1; values retained.')
            for i in range(5):
                for j in range(5):
                    matrix_rows.append(dict(cohort=cohort,quantity=kind,from_stage=stage_labels[i],
                       to_stage=stage_labels[j],mean=mean[i,j],n_valid=counts[i,j],n_matrices=len(stack)))
        for i,stage in enumerate(stage_labels):
            occupancy_rows.append(dict(cohort=cohort,stage=stage,stationary_mean=p[i],empirical_mean=e[i],
                n_stationary_valid=nps[i],n_empirical_valid=nes[i],symmetric_relative_difference=terms[i]))
        error_rows.append(dict(cohort=cohort,mean_symmetric_relative_difference=error,
                               percentage=100*error,n_valid_stages=int(np.isfinite(terms).sum())))
    qmax = max(float(np.nanmax(summaries[k]['Q'])) for k in names)
    if qmax == 0:
        qmax = 1.
    # Absolute font sizes in pt; physical output dimensions are preserved.
    rc = {'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
          'font.size':font_size,'axes.labelsize':font_size,'xtick.labelsize':font_size,
          'ytick.labelsize':font_size,'axes.linewidth':.65,'pdf.fonttype':42,
          'ps.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'text.usetex':False}
    axes, texts, paths = {}, [], {}
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(fig_width_mm/25.4,fig_height_mm/25.4),dpi=150)
        def axis(x,y,w,h,key):
            a=fig.add_axes([x/180,y/252,w/180,h/252],label=key)
            axes[key]=a
            return a
        def text(x,y,s,**kwargs):
            artist=fig.text(x/180,y/252,s,va='center',fontsize=font_size,**kwargs)
            texts.append(artist)
            return artist
        for x,cohort in zip((44,125),names):
            text(x,239,cohort,ha='center',weight='bold')
        for letter,y in [('a',239),('b',162),('c',83)]:
            artist=fig.text(3/180,y/252,letter,fontsize=panel_label_size,weight='bold',va='center')
            texts.append(artist)
        colormap=plt.get_cmap(cmap).copy();colormap.set_bad('#DCE1E5')
        for kind,bottom,vmax in [('T',175,1.),('Q',94,qmax)]:
            images=[]
            for col,(x,cohort) in enumerate(zip((17,98),names)):
                a=axis(x,bottom,54,54,kind+'_'+cohort)
                im=a.imshow(summaries[cohort][kind],cmap=colormap,vmin=0,vmax=vmax,
                            interpolation='nearest',origin='upper',aspect='auto')
                images.append(im)
                a.set_xticks(range(5),stage_labels);a.set_yticks(range(5),stage_labels)
                a.tick_params(length=0,pad=3)
                a.set_xlabel('To stage',labelpad=3)
                if col == 0:
                    a.set_ylabel('From stage',labelpad=3)
                text(x+27,bottom+59,f'n = {summaries[cohort]["n_"+kind]} recordings',ha='center',color='#667781')
                if show_matrix_values:
                    for i in range(5):
                        for j in range(5):
                            v=summaries[cohort][kind][i,j]
                            if np.isfinite(v):
                                a.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=font_size,
                                       color='white' if im.norm(v)>.55 else '#263844')
            ca=axis(158,bottom,3.2,54,'colorbar_'+kind)
            cb=fig.colorbar(images[-1],cax=ca)
            cb.set_label(r'$\langle '+kind+r'_{ij}^{(r)}\rangle_r$',labelpad=5,fontsize=font_size)
            cb.locator=mpl.ticker.MaxNLocator(4);cb.update_ticks()
            cb.ax.tick_params(length=2,pad=3)
        ymax=max(np.max(summaries[k][q]) for k in names for q in ('Pss','Empirical'))*100
        ymax=min(105.,max(10.,np.ceil(ymax/10)*10+10))
        for col,(x,cohort) in enumerate(zip((17,98),names)):
            a=axis(x,20,54,47,'C_'+cohort)
            p=100*summaries[cohort]['Pss'];e=100*summaries[cohort]['Empirical'];idx=np.arange(5)
            for j in idx:
                a.plot([j-.12,j+.12],[e[j],p[j]],color='#A6B3BB',lw=.9,zorder=2)
            a.scatter(idx-.12,e,s=28,marker='o',color=empirical_color,zorder=3)
            a.scatter(idx+.12,p,s=30,marker='D',color=stationary_color,zorder=3)
            a.set(xticks=idx,xticklabels=stage_labels,xlim=(-.5,4.5),ylim=(0,ymax))
            a.set_xlabel('Sleep stage',labelpad=4)
            if col == 0:
                a.set_ylabel('Occupancy (%)',labelpad=4)
            a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            a.spines[['top','right']].set_visible(False)
            a.tick_params(length=2.5,pad=3)
            a.grid(axis='y',color='#E7ECEF',linewidth=.6,zorder=0);a.set_axisbelow(True)
            if show_relative_difference:
                value=summaries[cohort]['relative_difference']
                label=f'{value:.2%}' if np.isfinite(value) else 'undefined'
                text(x+27,76,'Mean symmetric relative difference\n'+label,ha='center',color='#52606A')
        legend=fig.legend(handles=[Line2D([],[],ls='',marker='o',color=empirical_color,label='Empirical'),
                    Line2D([],[],ls='',marker='D',color=stationary_color,label='Stationary')],
                   loc='center',bbox_to_anchor=(.5,3.5/252),ncol=2,fontsize=font_size)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        complete_text=[t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
        clipped=[]
        for t in complete_text:
            box=t.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1+1 or box.y1 > fig.bbox.y1+1:
                clipped.append(t.get_text())
        if clipped:
            raise ValueError('Text extends outside canvas; increase physical dimensions: '+repr(clipped[:5]))
        # Exact repeated axes geometry (four heatmaps; C shares their columns).
        bounds={k:list(a.get_position().bounds) for k,a in axes.items()}
        for quantity in ('T','Q','C'):
            left=np.array(bounds[quantity+'_Nocturnal']);right=np.array(bounds[quantity+'_Nap'])
            if not np.allclose(left[1:],right[1:],atol=1e-9):
                raise RuntimeError('Panel alignment failed: '+quantity)
        alignment={'bounds_fraction':bounds,'status':'PASS','font_size_pt':font_size,
                   'canvas_mm':[fig_width_mm,fig_height_mm]}
        if output_dir is not None:
            folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
            for extension in ('pdf','svg','png'):
                path=folder/(basename+'.'+extension)
                fig.savefig(path,format=extension,dpi=dpi,facecolor='white')
                paths[extension]=str(path.resolve())
            for suffix,rows in [('matrix_means',matrix_rows),('occupancy_means',occupancy_rows),('relative_difference',error_rows)]:
                path=folder/(basename+'_'+suffix+'.csv')
                with path.open('w',newline='',encoding='utf-8-sig') as handle:
                    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                paths[suffix]=str(path.resolve())
            metadata={'column_order':names,'stage_order':stage_labels,'aggregation':'Equal record weights, finite mean per entry; no renormalization.',
                      'relative_difference_definition':'mean over defined stages of 2*abs(mean(Pss)-mean(Pemp))/(mean(Pss)+mean(Pemp)); computed after averaging.',
                      'both_zero_rule':'Undefined term excluded from this mean, matching original code.',
                      'panel_c':'Cohort stage means; connectors join empirical and stationary means, not participant pairs.',
                      'diagnostics':diagnostics,'alignment':alignment}
            (folder/(basename+'_metadata.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        if show:
            plt.show()
    return {'figure':fig,'axes':axes,'summaries':summaries,'alignment':alignment,'paths':paths,
            'matrix_rows':matrix_rows,'occupancy_rows':occupancy_rows,'relative_difference_rows':error_rows}


# ======================================================================
# Definitions from cleaned notebook cell 65 (zero-based)
# ======================================================================

# Cell 1: def. Dependencies: numpy, scipy, matplotlib. Standalone helpers only.
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats


def _sif2_vector(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or len(a) == 0:
        raise ValueError(f'{name}: expected a nonempty 1D vector, got {a.shape}.')
    return a.copy()


def prepare_si_fig2(cohorts, *, nap_sol_epoch_seconds=30.):
    """Preserve supplied EPR/TP values. Normalize counts by recorded TST minutes.

    Required per-cohort fields: TST, TransNum, ToWakeNum, loop_counts, EPR, TP,
    TPWake, SOL, TPSOL. All vectors except SOL/TPSOL must have matching lengths
    within each cohort. SOL/TPSOL use their own matched subset and need not
    match the full-record vectors. Alignment is by supplied order, never sorted.
    Both cohorts' TPSOL definitions were confirmed identical by the user.
    Nap SOL epochs -> minutes; nocturnal SOL already in minutes.
    Each panel uses one common finite mask for its displayed relationships.
    """
    if not np.isfinite(nap_sol_epoch_seconds) or nap_sol_epoch_seconds <= 0:
        raise ValueError('nap_sol_epoch_seconds must be positive.')
    full = ('TST','TransNum','ToWakeNum','loop_counts','EPR','TP','TPWake')
    panels = {k:[] for k in 'abcd'}
    for cohort in ('Nocturnal','Nap'):
        if cohort not in cohorts:
            raise KeyError('Missing cohort: '+cohort)
        d={k:_sif2_vector(cohorts[cohort][k],cohort+'.'+k) for k in full+('SOL','TPSOL')}
        lengths={k:len(d[k]) for k in full}
        if len(set(lengths.values())) != 1:
            raise ValueError(f'{cohort}: full-record vector lengths differ: {lengths}')
        if len(d['SOL']) != len(d['TPSOL']):
            raise ValueError(f"{cohort}: SOL ({len(d['SOL'])}) and TPSOL ({len(d['TPSOL'])}) must match.")
        valid_tst=np.isfinite(d['TST']) & (d['TST']>0)
        rates={k:np.divide(d[k],d['TST'],out=np.full(len(d[k]),np.nan),where=valid_tst)
               for k in ('TransNum','ToWakeNum','loop_counts')}
        sol=d['SOL']*(nap_sol_epoch_seconds/60 if cohort=='Nap' else 1.)
        inputs={'a':(rates['TransNum'],d['EPR'],d['TP']),
                'b':(rates['ToWakeNum'],d['EPR'],d['TPWake']),
                'c':(sol,None,d['TPSOL']),
                'd':(rates['loop_counts'],d['EPR'],d['TP'])}
        for panel,(x,epr,tp) in inputs.items():
            for i in range(len(x)):
                problems=[]
                if panel!='c' and not valid_tst[i]:problems.append('TST non-finite or non-positive')
                if not np.isfinite(x[i]):problems.append('non-finite x')
                elif x[i]<0:problems.append('negative sleep metric')
                if epr is not None and not np.isfinite(epr[i]):problems.append('non-finite EPR')
                if not np.isfinite(tp[i]):problems.append('non-finite TP')
                panels[panel].append(dict(panel=panel,cohort=cohort,source_position=i,
                    x=float(x[i]),EPR=float(epr[i]) if epr is not None else np.nan,
                    TP=float(tp[i]),included=not problems,exclusion_reason='; '.join(problems)))
    return {'panels':panels,'metadata':{
        'cohort_order':['Nocturnal','Nap'],'TST_unit':'minutes',
        'SOL_units':{'Nocturnal':'minutes','Nap':f'{nap_sol_epoch_seconds:g}-second epochs, converted to minutes'},
        'EPR_unit':'Original supplied units; not rescaled.',
        'TP_input_rule':'Use supplied TP values without recomputing or normalizing.',
        'SOL_TP_definition':'Identical early-window definition in both cohorts, confirmed by user.',
        'alignment':'Supplied within-cohort order. Panel c indexes an independent SOL subset; do not join its positions to full-record panels.',
        'mask':'Panel-wise common finite mask for x and every displayed y; no outlier or percentile removal.',
        'statistics':'Pooled Pearson correlations, two-sided and unadjusted; no formal cross-method correlation comparison.',
        'frequency_definition':'Supplied transition, awakening or hypnocycle count / TST minutes.'}}


def _sif2_ptext(p):
    if not np.isfinite(p):return r'$p=\mathrm{NA}$'
    if p == 0:return r'$p\approx0$'
    exponent=int(np.floor(np.log10(p)))
    coefficient=round(p/10.**exponent,2)
    if coefficient >= 10:
        coefficient/=10;exponent+=1
    return rf'$p={coefficient:.2f}\times10^{{{exponent}}}$'


def _sif2_regression(x,y):
    n=len(x)
    out={'n':n,'pearson_r':np.nan,'pearson_p':np.nan,'slope':np.nan,'intercept':np.nan}
    if n>=2 and np.ptp(x)>0:
        out['slope'],out['intercept']=map(float,np.polyfit(x,y,1))
        if np.ptp(y)>0:
            test=stats.pearsonr(x,y)
            out.update(pearson_r=float(test.statistic),pearson_p=float(test.pvalue))
    return out


def plot_si_fig2(data, *, font_size=10., panel_label_size=12.,
                 fig_width_mm=180.,fig_height_mm=184.,show_regression_ci=True,
                 epr_color='#765AA6',tp_color='#699E98',epr_axis_label='EPR',
                 point_size=10.,point_alpha=.45,output_dir='SI_Fig2_output',
                 basename='SI_Fig2',dpi=600,show=True):
    """Four equal panels, seven relationships; x always denotes a sleep metric.

    a: transition frequency vs EPR/TP; b: awakening frequency vs EPR/TPWake;
    c: SOL minutes vs TPEarly; d: hypnocycle frequency vs EPR/TP.
    Fits are OLS y-on-x, so orientation follows the new sleep-metric x axes.
    Optional bands are 95% CIs of the fitted mean, not prediction intervals.
    Correlation labels show n/r/p without correlation confidence intervals.
    EPR units must be supplied via epr_axis_label if a unit label is desired.
    """
    for key,v in [('font_size',font_size),('panel_label_size',panel_label_size),
                  ('fig_width_mm',fig_width_mm),('fig_height_mm',fig_height_mm),('dpi',dpi)]:
        if not np.isfinite(v) or v<=0:raise ValueError(key+' must be positive and finite.')
    axes={};results=[];paths={};counts=[]
    rc={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':font_size,'axes.labelsize':font_size,'xtick.labelsize':font_size,
        'ytick.labelsize':font_size,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','legend.frameon':False,'text.usetex':False}
    positions={'a':(18,109,167),'b':(109,109,167),'c':(18,21,79),'d':(109,21,79)}
    xlabels={'a':'Transition frequency\n(/min)','b':'Awakening frequency\n(/min)',
             'c':'Sleep onset latency\n(min)','d':'Hypnocycle frequency\n(/min)'}
    tplabels={'a':r'$\mathrm{TP}$','b':r'$\mathrm{TP}^{\mathrm{Wake}}$',
              'c':r'$\mathrm{TP}^{\mathrm{Early}}$','d':r'$\mathrm{TP}$'}
    with mpl.rc_context(rc):
        fig=plt.figure(figsize=(fig_width_mm/25.4,fig_height_mm/25.4),dpi=150)
        def text(x,y,s,**kwargs):
            return fig.text(x/180,y/184,s,fontsize=font_size,va='center',**kwargs)
        handles=[Line2D([],[],color=epr_color,label='EPR'),Line2D([],[],color=tp_color,label='TP'),
                 Line2D([],[],ls='',marker='o',markersize=4,color='#52606A',label='Nocturnal'),
                 Line2D([],[],ls='',marker='^',markersize=4,color='#52606A',label='Nap')]
        fig.legend(handles=handles,loc='center',bbox_to_anchor=(.5,179/184),ncol=4,
                   fontsize=font_size,handlelength=1.2,columnspacing=1.1,handletextpad=.4)
        for panel,(left,bottom,header) in positions.items():
            a=fig.add_axes([left/180,bottom/184,48/180,38/184],label=panel)
            axes[panel]=a
            rows=data['panels'][panel];used=[r for r in rows if r['included']]
            for cohort in ('Nocturnal','Nap'):
                source=[r for r in rows if r['cohort']==cohort]
                counts.append(dict(panel=panel,cohort=cohort,n_input=len(source),
                    n_included=sum(r['included'] for r in source),n_excluded=sum(not r['included'] for r in source)))
            fig.text((left-15)/180,header/184,panel,fontsize=panel_label_size,weight='bold',va='center')
            text(left,header,f'n = {len(used)}',color='#71808A')
            x=np.array([r['x'] for r in used],float)
            if panel!='c':
                twin=a.twinx();axes[panel+'_TP']=twin
                twin.patch.set_visible(False)
                groups=[('EPR',a,epr_color,epr_axis_label,header-7),
                        ('TP',twin,tp_color,tplabels[panel],header-13)]
                twin.spines[['top','left','bottom']].set_visible(False)
            else:
                groups=[('TP',a,tp_color,tplabels[panel],header-7)]
            for component,axis,color,ylabel,label_y in groups:
                y=np.array([r[component] for r in used],float)
                result=_sif2_regression(x,y)
                results.append(dict(panel=panel,component=component,**result))
                for cohort,marker in [('Nocturnal','o'),('Nap','^')]:
                    keep=np.array([r['cohort']==cohort for r in used],bool)
                    axis.scatter(x[keep],y[keep],s=point_size,marker=marker,color=color,
                                 alpha=point_alpha,linewidths=0,zorder=3)
                if np.isfinite(result['slope']):
                    grid=np.linspace(x.min(),x.max(),160)
                    prediction=result['intercept']+result['slope']*grid
                    axis.plot(grid,prediction,color=color,lw=1.2,zorder=4)
                    if show_regression_ci and len(x)>2:
                        variance=np.sum((y-(result['intercept']+result['slope']*x))**2)/(len(x)-2)
                        half=stats.t.ppf(.975,len(x)-2)*np.sqrt(variance*(1/len(x)+(grid-x.mean())**2/np.sum((x-x.mean())**2)))
                        axis.fill_between(grid,prediction-half,prediction+half,color=color,alpha=.10,lw=0,zorder=1)
                if len(y):
                    span=np.ptp(y);pad=.08*span if span else max(.001,abs(y[0])*.15)
                    axis.set_ylim(max(0.,float(y.min())-pad) if np.all(y>=0) else float(y.min())-pad,float(y.max())+pad)
                axis.set_ylabel(ylabel,color=color,labelpad=4)
                axis.tick_params(axis='y',colors=color,length=2.5,pad=3)
                side='right' if component=='TP' and panel!='c' else 'left'
                axis.spines[side].set_color(color)
                axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
                axis.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.3g}'))
                rr=f"{result['pearson_r']:.2f}" if np.isfinite(result['pearson_r']) else 'NA'
                text(left,label_y,f'r = {rr}; '+_sif2_ptext(result['pearson_p']),color=color)
            a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
            a.set_xlabel(xlabels[panel],labelpad=4)
            a.tick_params(axis='x',length=2.5,pad=3)
            a.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            a.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.3g}'))
            if len(x):
                pad=.04*np.ptp(x) if np.ptp(x) else max(.01,abs(x[0])*.1)
                a.set_xlim(max(0.,float(x.min())-pad),float(x.max())+pad)
        fig.canvas.draw()
        # Complete text objects, including math scripts, must stay inside the canvas.
        renderer=fig.canvas.get_renderer()
        visible=[t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
        clipped=[]
        for t in visible:
            b=t.get_window_extent(renderer)
            if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.x1+.5 or b.y1>fig.bbox.y1+.5:clipped.append(t.get_text())
        if clipped:raise ValueError('Text outside canvas; enlarge physical dimensions: '+repr(clipped[:5]))
        bounds={k:list(a.get_position().bounds) for k,a in axes.items()}
        for key in 'bcd':
            if not np.allclose(np.array(bounds[key])[2:],np.array(bounds['a'])[2:]):raise RuntimeError('Unequal plot sizes.')
        for key in ('a','b','d'):
            if not np.allclose(bounds[key],bounds[key+'_TP']):raise RuntimeError('Dual-axis bounds differ.')
        alignment={'status':'PASS','bounds_fraction':bounds,'canvas_mm':[fig_width_mm,fig_height_mm]}
        if output_dir is not None:
            folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
            for extension in ('pdf','svg','png'):
                path=folder/(basename+'.'+extension)
                fig.savefig(path,format=extension,dpi=dpi,facecolor='white');paths[extension]=str(path.resolve())
            for suffix,records in [('statistics',results),('source',[r for p in 'abcd' for r in data['panels'][p]]),('sample_counts',counts)]:
                path=folder/(basename+'_'+suffix+'.csv')
                with path.open('w',newline='',encoding='utf-8-sig') as h:
                    w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
                paths[suffix]=str(path.resolve())
            metadata={**data['metadata'],'alignment':alignment,
                      'regression_band':'95% confidence interval for fitted mean' if show_regression_ci else 'Not shown',
                      'input_mapping':'c: NocAllNoDiagT_ForSOL and NapAllNoDiagT; no substitution.'}
            (folder/(basename+'_metadata.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        if show:plt.show()
    return {'figure':fig,'axes':axes,'statistics':results,'sample_counts':counts,'data':data,'alignment':alignment,'paths':paths}


# ======================================================================
# Definitions from cleaned notebook cell 68 (zero-based)
# ======================================================================

# Cell 1: def. Dependencies: numpy, pandas, matplotlib. No original plotting defs needed.
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch


def _sif3_loop_label(code):
    """Decode names only; do not rotate, close, merge or otherwise alter families."""
    names={'0':'W','1':'N1','2':'N2','3':'N3','4':'R','WAKE':'W','W':'W',
           'REM':'R','R':'R','N1':'N1','N2':'N2','N3':'N3'}
    if isinstance(code,(list,tuple,np.ndarray)):
        tokens=[str(x) for x in code]
    else:
        raw=str(code).strip()
        if re.fullmatch('[0-4]+',raw):tokens=list(raw)
        else:tokens=re.findall(r'N[123]|WAKE|REM|W|R|[0-4]',raw.upper())
    if not tokens or any(t.upper() not in names for t in tokens):
        raise ValueError('Cannot decode loop column; supply loop_labels mapping: '+str(code))
    stages=[names[t.upper()] for t in tokens]
    return '\n'.join([stages[0]]+['→'+s for s in stages[1:]])


def prepare_si_fig3(Noc_df_occ,Nap_df_occ,Noc_df_steady,Nap_df_steady,Noc_loop_matrix,*,
                    duration_col='duration_min',matrix_type='orig',epoch_seconds=30.,
                    target_min=10.,curve_max_min=30.,order_by='occurrence',sort_records=True,
                    loop_labels=None):
    """A pools positive durations; B uses supplied TV trajectories; C uses REM-containing loop flux.

    Matrix includes REM + NREM loops from the original selection. No flux is decomposed or reweighted.
    B is a pointwise descriptive median/IQR, not a confidence interval. Trajectories
    are never padded or extrapolated. Both series must supply the plotted time point,
    matching the original distance-at-time extraction rule. No fixed TV threshold is used.
    """
    if not np.isfinite(epoch_seconds) or epoch_seconds<=0:raise ValueError('epoch_seconds must be positive.')
    step=target_min*60/epoch_seconds
    if target_min<0 or not np.isclose(step,round(step)):raise ValueError('target_min must be an exact non-negative Markov step.')
    target_step=int(round(step))
    if curve_max_min is not None and (not np.isfinite(curve_max_min) or curve_max_min<target_min):
        raise ValueError('curve_max_min must be None or >= target_min.')
    duration_frames=[]
    for cohort,frame in [('Nocturnal',Noc_df_occ),('Nap',Nap_df_occ)]:
        if frame is None or not len(frame):continue
        if duration_col not in frame:raise KeyError(cohort+': missing '+duration_col)
        tmp=frame.copy(deep=True)
        tmp['si_cohort']=cohort;tmp['si_source_index']=[str(i) for i in frame.index]
        values=pd.to_numeric(frame[duration_col],errors='coerce').to_numpy(float)
        tmp['si_duration_min']=values
        tmp['si_included']=np.isfinite(values)&(values>0)
        tmp['si_exclusion_reason']=np.where(~np.isfinite(values),'non-finite duration',np.where(values<=0,'non-positive duration',''))
        duration_frames.append(tmp)
    if not duration_frames:raise ValueError('No occurrence rows supplied.')
    duration_source=pd.concat(duration_frames,ignore_index=True)
    durations=duration_source.loc[duration_source.si_included,'si_duration_min'].to_numpy(float)
    if not len(durations):raise ValueError('No valid positive durations.')
    duration_summary=dict(n_occurrences=len(durations),mean_min=float(durations.mean()),
        median_min=float(np.median(durations)),std_min=float(np.std(durations,ddof=1)) if len(durations)>1 else np.nan,
        n_excluded=int((~duration_source.si_included).sum()))
    trajectories=[];trajectory_audit=[]
    for cohort,frame in [('Nocturnal',Noc_df_steady),('Nap',Nap_df_steady)]:
        required={'distance_series','probability_series'}|({'matrix_type'} if matrix_type is not None else set())
        if required-set(frame.columns):raise KeyError(cohort+': missing '+str(sorted(required-set(frame.columns))))
        for position,(idx,row) in enumerate(frame.iterrows()):
            record=dict(cohort=cohort,source_position=position,source_index=str(idx),included_curve=False,
                        included_target=False,distance_at_target=np.nan,reason='',n_joint_steps=0)
            if matrix_type is not None and row['matrix_type']!=matrix_type:
                record['reason']='other matrix_type';trajectory_audit.append(record);continue
            try:
                distance=np.asarray(row['distance_series'],float)
                probability=np.asarray(row['probability_series'],float)
                if distance.ndim!=1 or probability.ndim!=2 or probability.shape[1]!=5:
                    raise ValueError('expected distance (time,) and probability (time,5)')
                if np.isinf(distance).any() or np.any((distance[np.isfinite(distance)]<0)|(distance[np.isfinite(distance)]>1+1e-10)):
                    raise ValueError('TV values must lie in [0,1], with NaN allowed')
                length=min(len(distance),len(probability))
                if length==0 or not np.isfinite(distance[:length]).any():raise ValueError('no finite joint time points')
            except (ValueError,TypeError) as exc:
                record['reason']=str(exc);trajectory_audit.append(record);continue
            record['n_joint_steps']=length;record['included_curve']=True
            if len(distance)!=len(probability):record['reason']='unequal series lengths; only joint time indices used'
            if target_step<length and np.isfinite(distance[target_step]):
                record['included_target']=True;record['distance_at_target']=float(distance[target_step])
            else:record['reason']+='; target time absent or non-finite'
            trajectories.append(dict(cohort=cohort,source_position=position,source_index=str(idx),distance=distance[:length].copy()))
            trajectory_audit.append(record)
    if not trajectories:raise ValueError('No valid original TV trajectories.')
    audit=pd.DataFrame(trajectory_audit)
    target_values=audit.loc[audit.included_target,'distance_at_target'].to_numpy(float)
    if not len(target_values):raise ValueError('No valid TV distance at target time.')
    max_step=max(len(r['distance']) for r in trajectories)-1
    if curve_max_min is not None:max_step=min(max_step,int(np.floor(curve_max_min*60/epoch_seconds)))
    curves=np.full((len(trajectories),max_step+1),np.nan)
    for i,r in enumerate(trajectories):
        length=min(len(r['distance']),max_step+1);curves[i,:length]=r['distance'][:length]
    summary_rows=[]
    for k in range(max_step+1):
        vals=curves[:,k];vals=vals[np.isfinite(vals)]
        summary_rows.append(dict(time_min=k*epoch_seconds/60,n=len(vals),
            mean=float(vals.mean()) if len(vals) else np.nan,median=float(np.median(vals)) if len(vals) else np.nan,
            q25=float(np.quantile(vals,.25)) if len(vals) else np.nan,q75=float(np.quantile(vals,.75)) if len(vals) else np.nan))
    if not isinstance(Noc_loop_matrix,pd.DataFrame) or Noc_loop_matrix.empty:
        raise ValueError('Noc_loop_matrix must be a nonempty record-by-loop DataFrame.')
    if not Noc_loop_matrix.index.is_unique or not Noc_loop_matrix.columns.is_unique:
        raise ValueError('Loop matrix index and columns must be unique; duplicates cannot be silently merged.')
    matrix=Noc_loop_matrix.astype(float).copy(deep=True);values=matrix.to_numpy()
    if np.isinf(values).any() or np.any(values[np.isfinite(values)]<0):raise ValueError('Loop flux must be nonnegative; NaN denotes missing.')
    counts=(values>0).sum(axis=0);n_known=np.isfinite(values).sum(axis=0)
    means=np.divide(np.nansum(values,axis=0),n_known,out=np.full(values.shape[1],np.nan),where=n_known>0)
    if order_by=='occurrence':order=sorted(range(matrix.shape[1]),key=lambda j:(-counts[j],-np.nan_to_num(means[j],nan=-np.inf),j))
    elif order_by=='mean_flux':order=sorted(range(matrix.shape[1]),key=lambda j:(-np.nan_to_num(means[j],nan=-np.inf),-counts[j],j))
    elif order_by=='given':order=list(range(matrix.shape[1]))
    else:raise ValueError('order_by must be occurrence, mean_flux or given.')
    matrix=matrix.iloc[:,order]
    if sort_records:
        row_order=np.argsort(-np.nansum(matrix.to_numpy(),axis=1),kind='stable')
        matrix=matrix.iloc[row_order]
    labels=[]
    for code in matrix.columns:
        labels.append(str(loop_labels[code]) if loop_labels is not None and code in loop_labels else _sif3_loop_label(code))
    loop_summary=pd.DataFrame({'loop':[str(c) for c in matrix.columns],
        'n_records_present':(matrix.to_numpy()>0).sum(axis=0),
        'n_records_known':np.isfinite(matrix.to_numpy()).sum(axis=0),
        'mean_flux':matrix.mean(axis=0).to_numpy(),'display_label':labels})
    mapping=pd.DataFrame({'display_row':np.arange(1,len(matrix)+1),'original_record_id':[str(i) for i in matrix.index]})
    curve_source=pd.DataFrame([dict(cohort=r['cohort'],source_position=r['source_position'],
        source_index=r['source_index'],time_min=k*epoch_seconds/60,TV=float(value))
        for r in trajectories for k,value in enumerate(r['distance'][:max_step+1])])
    return dict(durations=durations,duration_source=duration_source,duration_summary=duration_summary,
        curve_source=curve_source,
        curves=curves,curve_summary=pd.DataFrame(summary_rows),trajectory_audit=audit,
        target_values=target_values,loop_matrix=matrix,loop_summary=loop_summary,record_mapping=mapping,
        metadata={'epoch_seconds':epoch_seconds,'target_min':target_min,'target_step':target_step,
        'curve_max_min':curve_max_min,'matrix_type':matrix_type,'order_by':order_by,'sort_records':sort_records,
        'loop_scope':'REM-containing flux loops with at least one NREM stage, from sum_nrem_rem_flux; no cycle-coverage filtering.',
        'a_unit':'Hypnocycle occurrences, pooled Nocturnal and Nap; positive finite durations only.',
        'b_unit':'Original per-record trajectories, pooled Nocturnal and Nap; existing TV values, not recomputed.',
        'b_band':'Pointwise 25th-75th percentiles, not a confidence interval; n may vary by time.',
        'c_count':'Number of nocturnal records with strictly positive REM-containing loop flux; not number of loop occurrences.',
        'c_missing':'Zero=absent, NaN=unknown; neither is assigned a positive pseudocount.'})


def plot_si_fig3(data,*,font_size=10.,panel_label_size=12.,fig_width_mm=180.,fig_height_mm=236.,
                 duration_bins=40,show_individual_curves=True,heatmap_cmap='viridis',
                 output_dir='SI_Fig3_output',basename='SI_Fig3',dpi=600,show=True):
    """Two quantitative panels above one full-width prevalence/flux panel.
    All supplied loop columns and records retained; only displayed row tick labels
    are thinned. Width/height auto-expand when full loop names require more room.
    """
    for name,v in [('font_size',font_size),('panel_label_size',panel_label_size),('fig_width_mm',fig_width_mm),('fig_height_mm',fig_height_mm),('dpi',dpi)]:
        if not np.isfinite(v) or v<=0:raise ValueError(name+' must be positive and finite.')
    loops=data['loop_summary'];matrix=data['loop_matrix'];nrow,ncol=matrix.shape
    labels=loops.display_label.tolist();nlines=max(len(s.splitlines()) for s in labels)
    width=max(float(fig_width_mm),36+ncol*9*font_size/10)
    label_margin=max(42.,(nlines*font_size*1.2+20)*25.4/72+10)
    extra=max(0.,label_margin-42.);height=max(float(fig_height_mm),236.+extra)
    # Heights in physical mm, leaving text size independent of canvas dimensions.
    hbottom=42.+extra;bar_bottom=height-111.;heat_top=bar_bottom-5
    if heat_top-hbottom<55:height+=55-(heat_top-hbottom);bar_bottom=height-111.;heat_top=bar_bottom-5
    axes={};paths={};blue='#496D83';ink='#263844';gray='#71808A';accent='#D55E00';median_color='#765AA6'
    rc={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':font_size,'axes.labelsize':font_size,'xtick.labelsize':font_size,'ytick.labelsize':font_size,
        'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.usetex':False,'legend.frameon':False}
    with mpl.rc_context(rc):
        fig=plt.figure(figsize=(width/25.4,height/25.4),dpi=150)
        def add(x,y,w,h,key):
            a=fig.add_axes([x/width,y/height,w/width,h/height],label=key);axes[key]=a;return a
        def text(x,y,s,**kwargs):return fig.text(x/width,y/height,s,fontsize=font_size,va='center',**kwargs)
        left=17.;plot_width=(width-60)/2;right=width-15-plot_width
        for letter,x,y in [('a',3,height-10),('b',right-14,height-10),('c',3,height-87)]:
            fig.text(x/width,y/height,letter,fontsize=panel_label_size,weight='bold',va='center')
        a=add(left,height-72,plot_width,40,'a');ds=data['duration_summary']
        a.hist(data['durations'],bins=duration_bins,color=blue,edgecolor='white',linewidth=.35)
        a.axvline(ds['mean_min'],color=accent,lw=1.3)
        a.axvline(ds['median_min'],color=ink,lw=1.1,ls='--')
        a.set(xlabel='Hypnocycle duration (min)',ylabel='Occurrence count')
        a.set_xlim(left=0)
        text(left,height-21,f"n = {ds['n_occurrences']:,} occurrences\nMean = {ds['mean_min']:.2f} min\nMedian = {ds['median_min']:.2f} min",color=ink)
        b=add(right,height-72,plot_width,40,'b');summary=data['curve_summary'];time=summary.time_min.to_numpy()
        if show_individual_curves:
            for curve in data['curves']:b.plot(time,curve,color='#A8B5BD',alpha=.18,lw=.4,zorder=1)
        b.fill_between(time,summary.q25.to_numpy(),summary.q75.to_numpy(),color=median_color,alpha=.15,lw=0,zorder=2)
        b.plot(time,summary['median'].to_numpy(),color=median_color,lw=1.5,zorder=3)
        target=data['metadata']['target_min'];tv=data['target_values'];mean_target=float(tv.mean())
        b.axvline(target,color=gray,lw=.9,ls='--',zorder=2)
        b.scatter([target],[mean_target],s=28,color=accent,marker='D',zorder=4)
        b.set(xlabel='Time from Wake initialization (min)',ylabel=r'$D_{\mathrm{TV}}(t)$',xlim=(0,time[-1]),ylim=(0,min(1.02,float(np.nanmax(data['curves']))*1.06)))
        text(right,height-21,f'n at {target:g} min = {len(tv)}\nMean at {target:g} min = {mean_target:.4f}\nMedian curve; shaded IQR',color=ink)
        for aa in (a,b):
            aa.spines[['top','right']].set_visible(False)
            aa.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4));aa.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            aa.tick_params(length=2.5,pad=3)
        cbar=add(left,bar_bottom,width-42,18,'c_count')
        heat=add(left,hbottom,width-42,heat_top-hbottom,'c_heat')
        x=np.arange(ncol);counts=loops.n_records_present.to_numpy()
        cbar.bar(x,counts,color=blue,width=.8,linewidth=0)
        cbar.set(xlim=(-.5,ncol-.5),ylabel='Records')
        cbar.tick_params(axis='x',bottom=False,labelbottom=False)
        cbar.tick_params(axis='y',length=2.5,pad=3)
        cbar.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3,integer=True))
        cbar.spines[['top','right']].set_visible(False)
        values=matrix.to_numpy();positive=values[np.isfinite(values)&(values>0)]
        background=np.ones((nrow,ncol,4));background[values==0]=mpl.colors.to_rgba('#E8ECEF')
        heat.imshow(background,aspect='auto',interpolation='nearest',origin='upper')
        cmap=plt.get_cmap(heatmap_cmap).copy();cmap.set_bad((0,0,0,0))
        lo=float(positive.min()) if len(positive) else 1.;hi=float(positive.max()) if len(positive) else 10.
        if hi==lo:hi=lo*10
        masked=np.ma.masked_where(~np.isfinite(values)|(values<=0),values)
        im=heat.imshow(masked,aspect='auto',interpolation='nearest',origin='upper',cmap=cmap,norm=LogNorm(lo,hi))
        heat.set_xlim(-.5,ncol-.5);heat.set_xticks(x,labels)
        heat.tick_params(axis='x',length=0,pad=4)
        ytick=np.unique(np.r_[np.linspace(0,nrow-1,min(7,nrow),dtype=int),nrow-1])
        heat.set_yticks(ytick,[str(i+1) for i in ytick]);heat.tick_params(axis='y',length=2.5,pad=3)
        heat.set_ylabel('Recording rank' if data['metadata']['sort_records'] else 'Recording row',labelpad=4)
        heat.set_xlabel('REM-containing flux-loop family',labelpad=5)
        cbax=add(width-22, hbottom,2.8,heat_top-hbottom,'colorbar')
        cb=fig.colorbar(im,cax=cbax);cb.set_label('Loop flux (log scale)',fontsize=font_size,labelpad=4)
        cb.ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10,numticks=4))
        cb.ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        cb.ax.tick_params(labelsize=font_size,length=2,pad=2)
        if not len(positive):cb.set_ticks([]);cb.set_label('No positive loop flux',fontsize=font_size)
        handles=[Patch(facecolor='#E8ECEF',edgecolor='#A8B5BD',label='Absent (0)')]
        if np.isnan(values).any():handles.append(Patch(facecolor='white',edgecolor='#A8B5BD',label='Missing'))
        fig.legend(handles=handles,loc='center',bbox_to_anchor=(.46,7/height),ncol=len(handles),fontsize=font_size,handlelength=1)
        text(width-3,7,'W: Wake; R: REM',ha='right',color=gray)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        clipped=[]
        for artist in fig.findobj(mpl.text.Text):
            if artist.get_visible() and artist.get_text().strip():
                box=artist.get_window_extent(renderer)
                if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.x1+.5 or box.y1>fig.bbox.y1+.5:clipped.append(artist.get_text())
        if clipped:raise ValueError('Text outside canvas; enlarge physical canvas: '+repr(clipped[:6]))
        bounds={k:list(ax.get_position().bounds) for k,ax in axes.items()}
        if not np.allclose(np.array(bounds['a'])[1:],np.array(bounds['b'])[1:]):raise RuntimeError('Upper-panel alignment failed.')
        if not np.allclose(np.array(bounds['c_count'])[[0,2]],np.array(bounds['c_heat'])[[0,2]]):raise RuntimeError('Count/heatmap columns misaligned.')
        alignment={'status':'PASS','bounds_fraction':bounds,'canvas_mm':[width,height]}
        if output_dir is not None:
            folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
            for extension in ('pdf','svg','png'):
                path=folder/(basename+'.'+extension);fig.savefig(path,format=extension,dpi=dpi,facecolor='white');paths[extension]=str(path.resolve())
            tables={'duration_source':data['duration_source'],'curve_summary':summary,'curve_source':data['curve_source'],'trajectory_audit':data['trajectory_audit'],
                    'loop_summary':loops,'record_mapping':data['record_mapping'],'loop_flux_matrix':matrix}
            for suffix,table in tables.items():
                path=folder/(basename+'_'+suffix+'.csv');table.to_csv(path,index=suffix=='loop_flux_matrix',encoding='utf-8-sig');paths[suffix]=str(path.resolve())
            meta={**data['metadata'],'duration_summary':ds,'n_at_target':len(tv),'mean_TV_at_target':mean_target,
                  'median_TV_at_target':float(np.median(tv)),'alignment':alignment,
                  'a_reference_lines':'Orange solid: mean; dark dashed: median.',
                  'b_target_marker':'Orange diamond: mean TV at target; purple curve: pointwise median.'}
            (folder/(basename+'_metadata.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        if show:plt.show()
    return dict(figure=fig,axes=axes,data=data,paths=paths,alignment=alignment)


# ======================================================================
# Definitions from cleaned notebook cell 71 (zero-based)
# ======================================================================

# Cell 1: def. Dependencies: numpy, pandas, matplotlib. Standalone plotting code.
from pathlib import Path
from collections.abc import Mapping
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def _sif4_stack(data,name,matrix_axes='auto'):
    ids=None
    if isinstance(data,Mapping):
        ids=list(data)
        if not ids:raise ValueError(name+' is empty.')
        a=np.stack([np.asarray(data[k],float) for k in ids])
        if a.shape[1:]!=(5,5):raise ValueError(name+': each dictionary value must be one 5x5 matrix.')
    else:
        a=np.asarray(data,float)
        if a.shape==(5,5):a=a[None,:,:]
        elif a.ndim==3:
            if isinstance(matrix_axes,str) and matrix_axes=='auto':
                candidates=[(i,i+1) for i in range(2) if a.shape[i:i+2]==(5,5)]
                if len(candidates)!=1:raise ValueError(f'{name}: ambiguous axes in {a.shape}; set matrix_axes explicitly.')
                axes=candidates[0]
            else:
                axes=tuple(int(i)%3 for i in matrix_axes)
                if len(axes)!=2 or axes[0]==axes[1] or any(a.shape[i]!=5 for i in axes):raise ValueError(name+': invalid matrix axes.')
            a=np.moveaxis(a,axes,(-2,-1))
        else:raise ValueError(f'{name}: expected (n,5,5), (5,5,n) or a dict of matrices; got {a.shape}.')
    if len(a)==0:raise ValueError(name+' contains no matrices.')
    finite=a[np.isfinite(a)]
    if np.isinf(a).any() or np.any((finite < -1e-10)|(finite > 1+1e-10)):
        raise ValueError(name+': entries must be probabilities in [0,1], or NaN.')
    return a.copy(),ids


def _sif4_pss(data,n,ids,name):
    if data is None:return None,'not supplied; individual occupancy/balance checks skipped'
    if isinstance(data,Mapping):
        if ids is None:raise ValueError(name+': provide stationary vectors in the matrix order or ID-labelled matrices.')
        if set(data)!=set(ids):raise ValueError(name+': stationary vector IDs and paired matrix IDs differ.')
        p=np.stack([np.asarray(data[k],float).reshape(-1) for k in ids])
    else:
        p=np.asarray(data,float)
        if p.shape==(5,) and n==1:p=p[None,:]
        elif p.shape==(5,) and n>1:return None,'only one group-level vector supplied; no per-record validation performed'
    if p.shape!=(n,5):raise ValueError(f'{name}: expected {(n,5)} stationary vectors in original matrix order; got {p.shape}.')
    if np.isinf(p).any() or np.any(p[np.isfinite(p)] < -1e-10):raise ValueError(name+': invalid stationary probabilities.')
    return p.copy(),'supplied per-record stationary vectors; matrix order retained'


def prepare_si_fig4(NocTmat,NapTmat,Noc_T_eq_list,Nap_T_eq_list,*,NocPss=None,NapPss=None,
                    original_matrix_axes='auto',removed_matrix_axes='auto',
                    noc_removed_ids=None,nap_removed_ids=None,validation_tolerance=1e-8):
    """Use supplied original/removed matrices; no reconstruction or renormalization.
    Both ID-labelled collections -> align by IDs. Otherwise assume existing order.
    Optional *_removed_ids label removed lists if their order differs from original.
    Pairwise finite cells use the same records in T/Tprime/Delta means; no cell filled.
    Diagnostics use individual matrices and supplied original stationary vectors,
    never the stationary vector of a group-average transition matrix.
    """
    groups={};audit=[];source=[];cells=[];messages=[]
    if not np.isfinite(validation_tolerance) or validation_tolerance<=0:raise ValueError('validation_tolerance must be positive.')
    inputs=[('Nocturnal',NocTmat,Noc_T_eq_list,NocPss,noc_removed_ids),('Nap',NapTmat,Nap_T_eq_list,NapPss,nap_removed_ids)]
    for cohort,before,after,pss,after_ids in inputs:
        t,ids=_sif4_stack(before,cohort+'.T',original_matrix_axes)
        eq,eq_ids=_sif4_stack(after,cohort+'.Tprime',removed_matrix_axes)
        if after_ids is not None:
            if eq_ids is not None:raise ValueError(cohort+': removed IDs already present as dictionary keys.')
            eq_ids=list(after_ids)
            if len(eq_ids)!=len(eq) or len(set(eq_ids))!=len(eq_ids):raise ValueError(cohort+': removed IDs must be unique and match list length.')
        if len(t)!=len(eq):raise ValueError(f'{cohort}: original {len(t)} vs removed {len(eq)} matrices; cannot pair.')
        if ids is not None and eq_ids is not None:
            if set(ids)!=set(eq_ids):raise ValueError(cohort+': original and removed record ID sets differ.')
            lookup={k:i for i,k in enumerate(eq_ids)};eq=eq[[lookup[k] for k in ids]]
            pairing='ID-aligned to original matrix keys'
        else:pairing='positional pairing in original supplied order; equal lengths checked'
        record_ids=ids if ids is not None else (eq_ids if eq_ids is not None else list(range(len(t))))
        p,p_note=_sif4_pss(pss,len(t),record_ids if ids is not None or eq_ids is not None else None,cohort+'.Pss')
        mask=np.isfinite(t)&np.isfinite(eq);n=mask.sum(axis=0)
        def mean(a):return np.divide(np.where(mask,a,0).sum(axis=0),n,out=np.full((5,5),np.nan),where=n>0)
        delta=eq-t
        if not np.isfinite(mean(delta)).any():raise ValueError(cohort+': no finite paired entries.')
        groups[cohort]=dict(T=mean(t),Tprime=mean(eq),Delta=mean(delta),n_pairs=len(t),n_valid=n,
                            pairing=pairing,pss_validation=p_note)
        for i in range(len(t)):
            complete=bool(mask[i].all())
            rec=dict(cohort=cohort,record_id=str(record_ids[i]),n_valid_cells=int(mask[i].sum()),
                complete_matrix_pair=complete,original_row_sum_error=np.nan,removed_row_sum_error=np.nan,
                min_diagonal_change=np.nan,max_offdiagonal_change=np.nan,stationary_vector_valid=False,
                original_stationarity_residual=np.nan,removed_stationarity_residual=np.nan,
                removed_detailed_balance_residual=np.nan,stationary_TV_after_one_step=np.nan,
                stationary_vector_sum_error=np.nan,validation_status='not audited: missing matrix/vector')
            if complete:
                rec.update(original_row_sum_error=float(np.max(np.abs(t[i].sum(axis=1)-1))),
                           removed_row_sum_error=float(np.max(np.abs(eq[i].sum(axis=1)-1))),
                           min_diagonal_change=float(np.diag(delta[i]).min()),
                           max_offdiagonal_change=float(delta[i][~np.eye(5,dtype=bool)].max()))
            if complete and p is not None and np.isfinite(p[i]).all():
                vector=p[i];rowflow=vector[:,None]*eq[i]
                rec.update(stationary_vector_valid=True,stationary_vector_sum_error=float(abs(vector.sum()-1)),
                    original_stationarity_residual=float(np.max(np.abs(vector@t[i]-vector))),
                    removed_stationarity_residual=float(np.max(np.abs(vector@eq[i]-vector))),
                    removed_detailed_balance_residual=float(np.max(np.abs(rowflow-rowflow.T))),
                    stationary_TV_after_one_step=float(.5*np.abs(vector@eq[i]-vector).sum()))
                checks=[rec[k] for k in ('removed_row_sum_error','stationary_vector_sum_error',
                        'original_stationarity_residual','removed_stationarity_residual','removed_detailed_balance_residual')]
                rec['validation_status']='PASS' if max(checks)<=validation_tolerance else 'CHECK'
            audit.append(rec)
            for r in range(5):
                for c in range(5):source.append(dict(cohort=cohort,record_id=str(record_ids[i]),from_index=r,to_index=c,
                    T=t[i,r,c],Tprime=eq[i,r,c],Delta=delta[i,r,c],included_mean=bool(mask[i,r,c])))
        for r in range(5):
            for c in range(5):cells.append(dict(cohort=cohort,from_index=r,to_index=c,n_valid=n[r,c],
                T=groups[cohort]['T'][r,c],Tprime=groups[cohort]['Tprime'][r,c],Delta=groups[cohort]['Delta'][r,c]))
        messages.append(cohort+': '+pairing+'; '+p_note)
    audits=pd.DataFrame(audit)
    if (audits.validation_status=='CHECK').any():
        warnings.warn('Some supplied matrix pairs fail the reconstruction checks; see validation CSV. Values have not been altered.')
    return dict(groups=groups,audit=audits,source=pd.DataFrame(source),mean_cells=pd.DataFrame(cells),
        metadata={'messages':messages,'stage_order':['Wake','N1','N2','N3','REM'],
          'delta_definition':'Per-record Tprime-T, then equal-record-weight finite paired-cell mean.',
          'missing_rule':'T/Tprime/Delta for a cell share the finite pair mask; valid n exported for each cell.',
          'pairing_warning':'Positional pairing assumes same record order; equal lengths alone do not establish identity.',
          'validation_tolerance':validation_tolerance,
          'stationary_check':'Use original supplied Pss; report max(abs(Pss@Tprime-Pss)).',
          'detailed_balance_check':'max(abs(diag(Pss)@Tprime-(diag(Pss)@Tprime).T)) per record.',
          'TV_check':'0.5*sum(abs(Pss@Tprime-Pss)); one-step invariance diagnostic, not a separately solved stationary distribution.'})


def plot_si_fig4(data,*,font_size=10.,panel_label_size=12.,fig_width_mm=180.,fig_height_mm=148.,
                 output_dir='SI_Fig4_output',basename='SI_Fig4',dpi=600,show=True):
    """2 cohorts x 3 quantities. Columns T/Tprime share [0,1]; Delta shares a zero-centred scale.
    No new steady-state distribution is inferred; diagnostics are exported separately.
    """
    for k,v in [('font_size',font_size),('panel_label_size',panel_label_size),('fig_width_mm',fig_width_mm),('fig_height_mm',fig_height_mm),('dpi',dpi)]:
        if not np.isfinite(v) or v<=0:raise ValueError(k+' must be positive and finite.')
    delta_max=max(float(np.nanmax(np.abs(g['Delta']))) for g in data['groups'].values())
    delta_max=delta_max if delta_max>0 else .001
    rc={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':font_size,
        'axes.labelsize':font_size,'xtick.labelsize':font_size,'ytick.labelsize':font_size,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.usetex':False,'axes.linewidth':.65}
    axes={};paths={}
    with mpl.rc_context(rc):
        fig=plt.figure(figsize=(fig_width_mm/25.4,fig_height_mm/25.4),dpi=150)
        def text(x,y,s,**kw):return fig.text(x/180,y/148,s,fontsize=font_size,va='center',**kw)
        headers=[r'Original $T$',r'Flux removed $T^{\prime}$',r'$\Delta T=T^{\prime}-T$']
        for x,title in zip((35,90,145),headers):text(x,140,title,ha='center',weight='bold')
        maps=[plt.get_cmap('Blues').copy(),plt.get_cmap('RdBu_r').copy()]
        for cm in maps:cm.set_bad('#DCE1E5')
        prob_image=delta_image=None
        for row,(cohort,bottom) in enumerate([('Nocturnal',94),('Nap',43)]):
            group=data['groups'][cohort]
            text(3,bottom+18,f"{cohort}, n = {group['n_pairs']}",rotation=90,ha='center')
            for col,(quantity,left) in enumerate(zip(('T','Tprime','Delta'),(17,72,127))):
                letter='abcdef'[row*3+col]
                fig.text(left/180,(bottom+41)/148,letter,fontsize=panel_label_size,va='center',weight='bold')
                ax=fig.add_axes([left/180,bottom/148,36/180,36/148],label=letter);axes[letter]=ax
                if quantity=='Delta':
                    image=ax.imshow(group[quantity],cmap=maps[1],norm=TwoSlopeNorm(vmin=-delta_max,vcenter=0,vmax=delta_max),interpolation='nearest',aspect='auto')
                    delta_image=image
                else:
                    image=ax.imshow(group[quantity],cmap=maps[0],vmin=0,vmax=1,interpolation='nearest',aspect='auto');prob_image=image
                ax.set_xticks(range(5),['W','N1','N2','N3','R']);ax.set_yticks(range(5),['W','N1','N2','N3','R'])
                ax.tick_params(length=0,pad=2)
                if row==1:ax.set_xlabel('To stage',labelpad=3)
        cp=fig.add_axes([17/180,12/148,91/180,2.5/148]);cd=fig.add_axes([127/180,12/148,36/180,2.5/148])
        pcb=fig.colorbar(prob_image,cax=cp,orientation='horizontal',ticks=[0,.5,1])
        dcb=fig.colorbar(delta_image,cax=cd,orientation='horizontal',ticks=[-delta_max,0,delta_max])
        dcb.formatter=mpl.ticker.FormatStrFormatter('%.2g');dcb.update_ticks()
        for cb,title in [(pcb,'Transition probability'),(dcb,'Change in probability')]:
            cb.set_label(title,labelpad=4,fontsize=font_size);cb.ax.xaxis.set_label_position('top')
            cb.ax.tick_params(length=2,pad=2,labelsize=font_size)
        text(90,1.8,'Rows: from stage; W: Wake; R: REM',ha='center',color='#71808A')
        fig.canvas.draw();renderer=fig.canvas.get_renderer();clipped=[]
        for t in fig.findobj(mpl.text.Text):
            if t.get_visible() and t.get_text().strip():
                box=t.get_window_extent(renderer)
                if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.x1+.5 or box.y1>fig.bbox.y1+.5:clipped.append(t.get_text())
        if clipped:raise ValueError('Text outside canvas; enlarge physical dimensions: '+repr(clipped[:6]))
        bounds={k:list(ax.get_position().bounds) for k,ax in axes.items()}
        for k in 'bcdef':
            if not np.allclose(np.array(bounds[k])[2:],np.array(bounds['a'])[2:]):raise RuntimeError('Matrix sizes are not aligned.')
        alignment={'status':'PASS','bounds_fraction':bounds,'canvas_mm':[fig_width_mm,fig_height_mm]}
        if output_dir is not None:
            folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
            for ext in ('pdf','svg','png'):
                path=folder/(basename+'.'+ext);fig.savefig(path,format=ext,dpi=dpi,facecolor='white');paths[ext]=str(path.resolve())
            for suffix,table in [('paired_source',data['source']),('mean_cells',data['mean_cells']),('validation',data['audit'])]:
                path=folder/(basename+'_'+suffix+'.csv');table.to_csv(path,index=False,encoding='utf-8-sig');paths[suffix]=str(path.resolve())
            metadata={**data['metadata'],'alignment':alignment,'delta_colour_limit':delta_max}
            (folder/(basename+'_metadata.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        if show:plt.show()
    return dict(figure=fig,axes=axes,data=data,alignment=alignment,paths=paths)


def configure_external_helpers(original_code_module):
    """Bind the original Code implementation used by early-sleep calculations.

    The module neither substitutes Codes.py nor reads the notebook's globals.
    No data arrays or simulation results are copied into this namespace.
    """
    helper = getattr(original_code_module, 'compute_J_matrix', None)
    if not callable(helper):
        raise ImportError('The original Code module must provide compute_J_matrix.')
    globals()['compute_J_matrix'] = helper



# Drug data preparation, paired comparisons, and final Fig5 / SI Fig5

def _build_drug_loading_helpers():
    # Telemetry original cell 2, line 6.
    from pathlib import Path

    # Telemetry original cell 2, line 7.
    from typing import Union, List, Tuple, Optional

    # Telemetry original cell 2, line 9.
    import datetime

    # Telemetry original cell 2, line 10.
    import re

    # Telemetry original cell 2, line 12.
    import numpy as np

    # Telemetry original cell 2, line 13.
    import pandas as pd

    # Telemetry original cell 2, line 14.
    import mne

    # Telemetry original cell 2, line 21.
    def parse_st_filename(
        filename: Union[str, Path]
    ) -> Tuple[int, int]:
        """
        从 Sleep-EDF Telemetry 文件名解析 subject 和 night。

        示例
        ----
        ST7011JP-Hypnogram.edf -> subject=1, night=1
        ST7012JP-Hypnogram.edf -> subject=1, night=2
        ST7021JM-Hypnogram.edf -> subject=2, night=1
        """
        name = Path(filename).name

        match = re.match(
            r"^ST7(?P<subject>\d{2})(?P<night>[12])",
            name,
            flags=re.IGNORECASE,
        )

        if match is None:
            raise ValueError(
                f"无法从 ST 文件名解析 subject/night：{name}"
            )

        subject = int(match.group("subject"))
        night = int(match.group("night"))

        return subject, night

    # Telemetry original cell 2, line 56.
    def _to_python_datetime(
        value
    ) -> Optional[datetime.datetime]:
        """
        将不同 MNE 版本返回的日期统一转换成 datetime。

        支持：
            datetime.datetime
            pandas.Timestamp
            numpy.datetime64
            (Unix seconds, microseconds)
            Unix timestamp
            字符串日期
        """
        if value is None:
            return None

        if isinstance(value, datetime.datetime):
            return value

        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()

        if isinstance(value, np.datetime64):
            return pd.Timestamp(value).to_pydatetime()

        # 旧版 MNE 可能使用 (seconds, microseconds)
        if isinstance(value, (tuple, list, np.ndarray)):
            if len(value) == 0:
                return None

            try:
                seconds = float(value[0])

                microseconds = (
                    float(value[1])
                    if len(value) > 1
                    else 0.0
                )

                timestamp = seconds + microseconds * 1e-6

                return datetime.datetime.fromtimestamp(
                    timestamp,
                    tz=datetime.timezone.utc,
                )

            except (TypeError, ValueError):
                return None

        if isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            if not np.isfinite(float(value)):
                return None

            return datetime.datetime.fromtimestamp(
                float(value),
                tz=datetime.timezone.utc,
            )

        # 最后尝试字符串解析
        try:
            parsed = pd.to_datetime(
                value,
                errors="coerce",
                utc=True,
            )

            if pd.isna(parsed):
                return None

            return parsed.to_pydatetime()

        except Exception:
            return None

    # Telemetry original cell 2, line 144.
    def _read_raw_edf_compat(
        edf_path: Union[str, Path]
    ):
        """
        兼容可能不接受 verbose 参数的旧版 MNE。
        """
        edf_path = str(edf_path)

        try:
            return mne.io.read_raw_edf(
                edf_path,
                preload=False,
                verbose=False,
            )

        except TypeError as exc:
            if "verbose" not in str(exc):
                raise

            return mne.io.read_raw_edf(
                edf_path,
                preload=False,
            )

    # Telemetry original cell 2, line 169.
    def _get_recording_start(
        edf_path: Union[str, Path],
        annotations,
    ) -> datetime.datetime:
        """
        记录绝对起始时间优先级：

            annotations.orig_time
            raw.info["meas_date"]
        """
        start_dt = _to_python_datetime(
            getattr(
                annotations,
                "orig_time",
                None,
            )
        )

        if start_dt is not None:
            return start_dt

        raw = None

        try:
            raw = _read_raw_edf_compat(
                edf_path
            )

            start_dt = _to_python_datetime(
                raw.info.get(
                    "meas_date",
                    None,
                )
            )

        except Exception as exc:
            raise ValueError(
                f"无法取得 EDF 起始时间：{Path(edf_path).name}\n"
                f"annotations.orig_time 为空，且读取 meas_date 失败：{exc}"
            ) from exc

        finally:
            if raw is not None:
                close_function = getattr(
                    raw,
                    "close",
                    None,
                )

                if callable(close_function):
                    close_function()

        if start_dt is None:
            raise ValueError(
                f"EDF 中没有有效的绝对起始时间：{Path(edf_path).name}"
            )

        return start_dt

    # Telemetry original cell 2, line 233.
    def _stage_description_to_code(
        description: str
    ) -> int:
        """
        五状态编码：

            Wake  -> 0
            N1    -> 1
            N2    -> 2
            N3/N4 -> 3
            REM   -> 4

        Movement time、未知阶段 -> -1
        """
        text = str(description).strip().lower()

        mapping = {
            "sleep stage w": 0,
            "sleep stage 1": 1,
            "sleep stage 2": 2,
            "sleep stage 3": 3,
            "sleep stage 4": 3,
            "sleep stage r": 4,
            "sleep stage ?": -1,
            "movement time": -1,
        }

        return mapping.get(text, -1)

    # Telemetry original cell 2, line 263.
    def _stage_code_to_name(
        stage_code: int
    ) -> str:
        mapping = {
            0: "Wake",
            1: "N1",
            2: "N2",
            3: "N3",
            4: "REM",
            -1: "Unknown",
        }

        return mapping.get(
            int(stage_code),
            "Unknown",
        )

    # Telemetry original cell 2, line 285.
    def _parse_single_hypnogram(
        edf_path: Union[str, Path],
        target_epoch_sec: float = 30.0,
    ) -> pd.DataFrame:
        """
        读取单个 Sleep-EDF Hypnogram。

        关键处理：
        1. 从记录起点 t=0 建立完整的 30 秒时间轴；
        2. 第一条 annotation 之前的空白区间填为 Unknown；
        3. annotation 之间的空隙也填为 Unknown；
        4. 因此 DataFrame 行号就是从记录起点计算的 30 秒 epoch index。

        睡眠阶段编码：
            Wake    -> 0
            N1      -> 1
            N2      -> 2
            N3/N4   -> 3
            REM     -> 4
            Unknown -> -1
        """
        edf_path = Path(edf_path)

        if not edf_path.exists():
            raise FileNotFoundError(
                f"找不到 EDF 文件：{edf_path}"
            )

        # 兼容旧版 MNE：不传 verbose
        annotations = mne.read_annotations(
            str(edf_path)
        )

        if annotations is None or len(annotations) == 0:
            raise RuntimeError(
                f"{edf_path.name} 中没有 annotation。"
            )

        start_dt = _get_recording_start(
            edf_path,
            annotations,
        )

        annotation_onsets = np.asarray(
            annotations.onset,
            dtype=float,
        )

        annotation_durations = np.asarray(
            annotations.duration,
            dtype=float,
        )

        valid_annotation = (
            np.isfinite(annotation_onsets)
            & np.isfinite(annotation_durations)
            & (annotation_durations > 0)
        )

        if not np.any(valid_annotation):
            raise RuntimeError(
                f"{edf_path.name} 中没有有效 annotation。"
            )

        # 记录长度取所有 annotation 的最大结束时刻
        annotation_ends = (
            annotation_onsets[valid_annotation]
            + annotation_durations[valid_annotation]
        )

        duration_s = float(
            np.max(annotation_ends)
        )

        if duration_s <= 0:
            raise RuntimeError(
                f"{edf_path.name} 的记录时长无效：{duration_s}"
            )

        # --------------------------------------------------------
        # 从记录起点 t=0 建立完整的固定长度时间轴
        # --------------------------------------------------------

        n_epochs = int(
            np.ceil(
                duration_s
                / float(target_epoch_sec)
            )
        )

        epoch_starts = (
            np.arange(
                n_epochs,
                dtype=float,
            )
            * float(target_epoch_sec)
        )

        epoch_ends = np.minimum(
            epoch_starts
            + float(target_epoch_sec),
            duration_s,
        )

        epoch_durations = (
            epoch_ends
            - epoch_starts
        )

        # 默认全部为未评分
        stage_codes = np.full(
            n_epochs,
            -1,
            dtype=int,
        )

        stage_names = np.full(
            n_epochs,
            "Unknown",
            dtype=object,
        )

        annotation_names = np.full(
            n_epochs,
            "Unscored",
            dtype=object,
        )

        # --------------------------------------------------------
        # 将 annotation 覆盖到完整时间轴
        # --------------------------------------------------------

        for onset, duration, description in zip(
            annotations.onset,
            annotations.duration,
            annotations.description,
        ):
            try:
                onset = float(onset)
                duration = float(duration)
            except (TypeError, ValueError):
                continue

            if (
                not np.isfinite(onset)
                or not np.isfinite(duration)
                or duration <= 0
            ):
                continue

            description = str(
                description
            ).strip()

            description_lower = (
                description.lower()
            )

            is_sleep_stage = (
                description_lower.startswith(
                    "sleep stage"
                )
            )

            is_movement = (
                description_lower.startswith(
                    "movement time"
                )
            )

            if not (
                is_sleep_stage
                or is_movement
            ):
                continue

            annotation_end = (
                onset + duration
            )

            stage_code = (
                _stage_description_to_code(
                    description
                )
            )

            stage_name = (
                _stage_code_to_name(
                    stage_code
                )
            )

            # 找出与当前 annotation 有重叠的所有 epoch
            overlap_mask = (
                (epoch_starts < annotation_end)
                & (epoch_ends > onset)
            )

            overlap_indices = np.flatnonzero(
                overlap_mask
            )

            if len(overlap_indices) == 0:
                continue

            stage_codes[
                overlap_indices
            ] = int(stage_code)

            stage_names[
                overlap_indices
            ] = stage_name

            annotation_names[
                overlap_indices
            ] = description

        # --------------------------------------------------------
        # 生成 DataFrame
        # --------------------------------------------------------

        df = pd.DataFrame(
            {
                "epoch_idx": np.arange(
                    n_epochs,
                    dtype=int,
                ),
                "t_start": epoch_starts,
                "t_end": epoch_ends,
                "epoch_sec": epoch_durations,
                "stage": stage_codes,
                "sleep_stage": stage_codes.copy(),
                "stage_name": stage_names,
                "annotation": annotation_names,
            }
        )

        end_dt = (
            start_dt
            + datetime.timedelta(
                seconds=duration_s
            )
        )

        df.attrs["start_dt"] = start_dt
        df.attrs["duration_s"] = duration_s
        df.attrs["end_dt"] = end_dt
        df.attrs["target_epoch_sec"] = float(
            target_epoch_sec
        )

        return df

    # Telemetry original cell 2, line 543.
    def load_sleepstageage(
        root_dir: Union[str, Path] = "",
        pattern: str = "*-Hypnogram.edf",
    ) -> List[pd.DataFrame]:
        """
        加载 SleepEDF_Telemetry 中的全部 Hypnogram。

        root_dir 可以是：

            ""
            "SleepEDF_Telemetry"
            Path("SleepEDF_Telemetry")

        排列顺序：
            subject 升序
            同一 subject 内 night 升序
        """
        root_text = str(
            root_dir
        ).strip()

        if root_text == "":
            root = Path(".").resolve()
        else:
            root = Path(
                root_dir
            ).expanduser().resolve()

        # 用户直接传入 SleepEDF_Telemetry 文件夹
        if (
            root.is_dir()
            and root.name.lower()
            == "sleepedf_telemetry"
        ):
            st_dir = root

        # 用户传入当前项目根目录
        elif (
            root
            / "SleepEDF_Telemetry"
        ).is_dir():
            st_dir = (
                root
                / "SleepEDF_Telemetry"
            )

        # 用户传入其他名称的数据目录，但里面直接有文件
        elif (
            root.is_dir()
            and any(
                root.glob(pattern)
            )
        ):
            st_dir = root

        else:
            raise FileNotFoundError(
                "找不到 SleepEDF_Telemetry 文件夹。\n"
                f"当前检查路径：{root}"
            )

        files = list(
            st_dir.glob(pattern)
        )

        if len(files) == 0:
            # 兼容大小写差异
            files = [
                fp
                for fp in st_dir.glob("*.edf")
                if "hypnogram" in fp.name.lower()
            ]

        if len(files) == 0:
            raise FileNotFoundError(
                f"在 {st_dir} 中没有找到 Hypnogram EDF 文件。"
            )

        # 按 subject、night 显式排序
        files = sorted(
            files,
            key=lambda fp: (
                parse_st_filename(
                    fp.name
                )
            ),
        )

        all_dfs = []

        for edf_path in files:
            subject, night = (
                parse_st_filename(
                    edf_path.name
                )
            )

            try:
                df = _parse_single_hypnogram(
                    edf_path=edf_path,
                    target_epoch_sec=30.0,
                )

            except Exception as exc:
                raise RuntimeError(
                    f"读取文件失败：{edf_path.name}\n"
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            df.insert(
                0,
                "subject",
                int(subject),
            )

            df.insert(
                1,
                "night",
                int(night),
            )

            df.insert(
                2,
                "file_name",
                edf_path.name,
            )

            df.insert(
                3,
                "file_path",
                str(
                    edf_path.resolve()
                ),
            )

            df.attrs["subject"] = int(
                subject
            )

            df.attrs["night"] = int(
                night
            )

            df.attrs["file_name"] = (
                edf_path.name
            )

            df.attrs["file_path"] = str(
                edf_path.resolve()
            )

            all_dfs.append(
                df
            )

        if len(all_dfs) == 0:
            raise RuntimeError(
                "找到了 EDF 文件，但没有成功读取任何 Hypnogram。"
            )

        return all_dfs

    # Telemetry original cell 2, line 710.
    def _normalize_header_text(
        value
    ) -> str:
        if pd.isna(value):
            return ""

        return re.sub(
            r"[^a-z0-9]",
            "",
            str(value).strip().lower(),
        )

    # Telemetry original cell 2, line 723.
    def read_st_subjects(
        xls_path: Union[str, Path]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        按 ST-subjects.xls 的实际宽表结构读取元数据。

        表格结构：
            subject | age | sex |
            placebo night | placebo lights off |
            temazepam night | temazepam lights off

        返回
        ----
        subject_df:
            每名受试者一行。

        night_df:
            每名受试者每晚一行，包含 condition。
        """
        xls_path = Path(
            xls_path
        )

        if not xls_path.exists():
            raise FileNotFoundError(
                f"找不到 Excel 文件：{xls_path}"
            )

        try:
            raw_excel = pd.read_excel(
                xls_path,
                sheet_name="Blad1",
                header=None,
            )

        except ImportError as exc:
            raise ImportError(
                "读取 .xls 文件需要 xlrd。\n"
                "请在当前 Python 环境运行：\n"
                "pip install xlrd"
            ) from exc

        # 去除完全空白的行列
        raw_excel = (
            raw_excel
            .dropna(
                axis=0,
                how="all",
            )
            .dropna(
                axis=1,
                how="all",
            )
            .reset_index(drop=True)
        )

        if raw_excel.shape[1] < 7:
            raise ValueError(
                "ST-subjects.xls 的有效列数少于 7。\n"
                f"当前 shape={raw_excel.shape}"
            )

        # 自动寻找第二层表头：
        # Nr, age, M1/F2, night nr, lights off, night nr, lights off
        header_row = None

        for row_index, row in raw_excel.iterrows():
            normalized = [
                _normalize_header_text(
                    value
                )
                for value in row.tolist()
            ]

            contains_age = (
                "age" in normalized
            )

            contains_sex = any(
                value in {
                    "m1f2",
                    "sex",
                    "gender",
                }
                for value in normalized
            )

            night_nr_count = sum(
                value == "nightnr"
                for value in normalized
            )

            if (
                contains_age
                and contains_sex
                and night_nr_count >= 2
            ):
                header_row = int(
                    row_index
                )
                break

        # 若自动识别失败，按官方表格的第三行开始读取
        if header_row is None:
            if raw_excel.shape[0] < 3:
                raise ValueError(
                    "无法识别 ST-subjects.xls 的表头。"
                )

            header_row = 1

        subject_df = raw_excel.iloc[
            header_row + 1:,
            :7,
        ].copy()

        subject_df.columns = [
            "subject",
            "age",
            "sex_code",
            "placebo_night",
            "placebo_lights_off",
            "temazepam_night",
            "temazepam_lights_off",
        ]

        # 只保留 subject 为数字的数据行
        subject_df["subject"] = pd.to_numeric(
            subject_df["subject"],
            errors="coerce",
        )

        subject_df = subject_df.dropna(
            subset=["subject"]
        ).copy()

        subject_df["subject"] = (
            subject_df["subject"]
            .astype(int)
        )

        subject_df["age"] = pd.to_numeric(
            subject_df["age"],
            errors="coerce",
        )

        subject_df["sex_code"] = pd.to_numeric(
            subject_df["sex_code"],
            errors="coerce",
        )

        for column in [
            "placebo_night",
            "temazepam_night",
        ]:
            subject_df[column] = pd.to_numeric(
                subject_df[column],
                errors="coerce",
            )

            bad_rows = subject_df[
                column
            ].isna()

            if bad_rows.any():
                bad_subjects = subject_df.loc[
                    bad_rows,
                    "subject",
                ].tolist()

                raise ValueError(
                    f"{column} 为空或无法转成数字，"
                    f"对应 subject：{bad_subjects}"
                )

            subject_df[column] = (
                subject_df[column]
                .astype(int)
            )

        # night 只能为 1 或 2
        invalid_night = (
            ~subject_df[
                "placebo_night"
            ].isin([1, 2])
            |
            ~subject_df[
                "temazepam_night"
            ].isin([1, 2])
        )

        if invalid_night.any():
            raise ValueError(
                "发现 night 编号不是 1 或 2：\n"
                + subject_df.loc[
                    invalid_night,
                    [
                        "subject",
                        "placebo_night",
                        "temazepam_night",
                    ],
                ].to_string(
                    index=False
                )
            )

        same_night = (
            subject_df[
                "placebo_night"
            ]
            ==
            subject_df[
                "temazepam_night"
            ]
        )

        if same_night.any():
            bad_subjects = subject_df.loc[
                same_night,
                "subject",
            ].tolist()

            raise ValueError(
                "以下 subject 的 Placebo 和 Temazepam "
                f"被标成同一晚：{bad_subjects}"
            )

        duplicated_subjects = (
            subject_df[
                "subject"
            ].duplicated(
                keep=False
            )
        )

        if duplicated_subjects.any():
            raise ValueError(
                "ST-subjects.xls 中存在重复 subject：\n"
                + subject_df.loc[
                    duplicated_subjects,
                    ["subject"],
                ].to_string(
                    index=False
                )
            )

        subject_df = subject_df.sort_values(
            "subject",
            kind="stable",
        ).reset_index(drop=True)

        # --------------------------------------------------------
        # Placebo 转成长表
        # --------------------------------------------------------

        placebo_df = subject_df[
            [
                "subject",
                "age",
                "sex_code",
                "placebo_night",
                "placebo_lights_off",
            ]
        ].copy()

        placebo_df = placebo_df.rename(
            columns={
                "placebo_night": "night",
                "placebo_lights_off": "lights_off",
            }
        )

        placebo_df["condition"] = (
            "Placebo"
        )

        # --------------------------------------------------------
        # Temazepam 转成长表
        # --------------------------------------------------------

        temazepam_df = subject_df[
            [
                "subject",
                "age",
                "sex_code",
                "temazepam_night",
                "temazepam_lights_off",
            ]
        ].copy()

        temazepam_df = (
            temazepam_df.rename(
                columns={
                    "temazepam_night": "night",
                    "temazepam_lights_off": "lights_off",
                }
            )
        )

        temazepam_df["condition"] = (
            "Temazepam"
        )

        night_df = pd.concat(
            [
                placebo_df,
                temazepam_df,
            ],
            ignore_index=True,
        )

        night_df["night"] = (
            night_df["night"]
            .astype(int)
        )

        night_df = night_df[
            [
                "subject",
                "night",
                "condition",
                "age",
                "sex_code",
                "lights_off",
            ]
        ]

        duplicated_nights = (
            night_df.duplicated(
                subset=[
                    "subject",
                    "night",
                ],
                keep=False,
            )
        )

        if duplicated_nights.any():
            raise ValueError(
                "Excel 中存在重复的 subject/night：\n"
                + night_df.loc[
                    duplicated_nights,
                    [
                        "subject",
                        "night",
                        "condition",
                    ],
                ].to_string(
                    index=False
                )
            )

        night_df = night_df.sort_values(
            [
                "subject",
                "night",
            ],
            kind="stable",
        ).reset_index(drop=True)

        return subject_df, night_df

    # Telemetry original cell 2, line 1090.
    def _parse_lights_off_clock(
        value
    ) -> datetime.time:
        """
        将 Excel 中的 lights off 转成 datetime.time。

        支持：
            datetime.time
            datetime.datetime
            pandas.Timestamp
            datetime.timedelta
            pandas.Timedelta
            Excel 小数时间
            "22:30"
            "22:30:00"
        """
        if value is None or pd.isna(value):
            raise ValueError(
                "lights off 为空。"
            )

        if isinstance(
            value,
            datetime.time,
        ):
            return value.replace(
                tzinfo=None
            )

        if isinstance(
            value,
            datetime.datetime,
        ):
            return value.time().replace(
                tzinfo=None
            )

        if isinstance(
            value,
            pd.Timestamp,
        ):
            return value.time().replace(
                tzinfo=None
            )

        if isinstance(
            value,
            datetime.timedelta,
        ):
            total_seconds = int(
                round(
                    value.total_seconds()
                )
            ) % (24 * 3600)

            return datetime.time(
                hour=total_seconds // 3600,
                minute=(
                    total_seconds % 3600
                ) // 60,
                second=total_seconds % 60,
            )

        if isinstance(
            value,
            pd.Timedelta,
        ):
            total_seconds = int(
                round(
                    value.total_seconds()
                )
            ) % (24 * 3600)

            return datetime.time(
                hour=total_seconds // 3600,
                minute=(
                    total_seconds % 3600
                ) // 60,
                second=total_seconds % 60,
            )

        # Excel 时间可能以一天的小数保存
        if isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            value_float = float(
                value
            )

            if (
                np.isfinite(value_float)
                and 0 <= value_float < 1
            ):
                total_seconds = int(
                    round(
                        value_float
                        * 24
                        * 3600
                    )
                ) % (24 * 3600)

                return datetime.time(
                    hour=total_seconds // 3600,
                    minute=(
                        total_seconds % 3600
                    ) // 60,
                    second=total_seconds % 60,
                )

        text = str(
            value
        ).strip()

        for time_format in [
            "%H:%M:%S",
            "%H:%M",
        ]:
            try:
                return datetime.datetime.strptime(
                    text,
                    time_format,
                ).time()

            except ValueError:
                pass

        parsed = pd.to_datetime(
            text,
            errors="coerce",
        )

        if pd.isna(parsed):
            raise ValueError(
                f"无法解析 lights off：{value!r}"
            )

        return parsed.time().replace(
            tzinfo=None
        )

    # Telemetry original cell 2, line 1245.
    def _resolve_lights_off_datetime(
        start_dt: datetime.datetime,
        end_dt: datetime.datetime,
        lights_off_value,
        boundary_tolerance_sec: float = 60.0,
    ) -> datetime.datetime:
        """
        Excel 中 lights off 通常只精确到分钟，而 EDF 起始时间
        可能位于 xx:xx:30，因此允许最多 boundary_tolerance_sec
        的边界误差。

        例如：
            lights off = 00:00:00
            EDF start  = 00:00:30

        差 30 秒时，将 lights off 归到 EDF 起始时间。
        """
        start_dt = _to_python_datetime(start_dt)
        end_dt = _to_python_datetime(end_dt)

        if start_dt is None:
            raise ValueError("start_dt 无效。")

        if end_dt is None:
            raise ValueError("end_dt 无效。")

        if end_dt < start_dt:
            raise ValueError(
                f"end_dt 早于 start_dt："
                f"start={start_dt}, end={end_dt}"
            )

        clock_time = _parse_lights_off_clock(
            lights_off_value
        )

        # 关灯时刻可能在记录开始日的前一天、当天或下一天
        candidates = []

        for day_offset in (-1, 0, 1, 2):
            candidate_date = (
                start_dt.date()
                + datetime.timedelta(days=day_offset)
            )

            candidate_dt = datetime.datetime.combine(
                candidate_date,
                clock_time,
            )

            # 保持与 EDF 时间相同的 timezone
            if start_dt.tzinfo is not None:
                candidate_dt = candidate_dt.replace(
                    tzinfo=start_dt.tzinfo
                )

            candidates.append(candidate_dt)

        # --------------------------------------------------------
        # 1. 优先选择严格位于记录区间内的候选值
        # --------------------------------------------------------

        inside_candidates = [
            candidate
            for candidate in candidates
            if start_dt <= candidate <= end_dt
        ]

        if len(inside_candidates) == 1:
            return inside_candidates[0]

        if len(inside_candidates) > 1:
            return min(
                inside_candidates,
                key=lambda candidate: abs(
                    (candidate - start_dt).total_seconds()
                ),
            )

        # --------------------------------------------------------
        # 2. 允许关灯时间略早于 EDF 起点
        # --------------------------------------------------------

        before_start = [
            candidate
            for candidate in candidates
            if candidate < start_dt
        ]

        if before_start:
            nearest_before = max(before_start)

            difference_sec = (
                start_dt - nearest_before
            ).total_seconds()

            if difference_sec <= boundary_tolerance_sec:
                # Excel 时间按分钟记录，而 EDF 可能从 xx:xx:30 开始。
                # 将这种情况归到记录的第一个 epoch。
                return start_dt

        # --------------------------------------------------------
        # 3. 允许关灯时间略晚于 EDF 终点
        # --------------------------------------------------------

        after_end = [
            candidate
            for candidate in candidates
            if candidate > end_dt
        ]

        if after_end:
            nearest_after = min(after_end)

            difference_sec = (
                nearest_after - end_dt
            ).total_seconds()

            if difference_sec <= boundary_tolerance_sec:
                return end_dt

        raise ValueError(
            "lights off 无法放入该 EDF 的记录范围。\n"
            f"record start = {start_dt}\n"
            f"record end   = {end_dt}\n"
            f"lights off   = {lights_off_value!r}\n"
            f"tolerance    = {boundary_tolerance_sec} s\n"
            f"candidates   = {candidates}"
        )

    # Telemetry original cell 2, line 1376.
    def lights_off_epoch_idx_by_tstart_end(
        df_i: pd.DataFrame,
        start_dt: datetime.datetime,
        lights_off_value,
        boundary_tolerance_sec: float = 60.0,
    ) -> int:
        """
        返回 lights off 对应的 30 秒 epoch index，0-based。

        由于 df_i 现在从记录起点 t=0 建立完整时间轴，
        所以可以直接根据 elapsed_seconds 计算 epoch index。

        例如：
            recording start = 23:49:30
            lights off      = 23:53:00

            elapsed = 210 秒
            index   = floor(210 / 30) = 7
        """
        if len(df_i) == 0:
            raise ValueError(
                "epoch DataFrame 为空。"
            )

        start_dt = _to_python_datetime(
            start_dt
        )

        if start_dt is None:
            raise ValueError(
                "start_dt 为空或格式无效。"
            )

        required_columns = {
            "epoch_idx",
            "t_start",
            "t_end",
            "epoch_sec",
        }

        missing_columns = (
            required_columns
            - set(df_i.columns)
        )

        if missing_columns:
            raise KeyError(
                "epoch DataFrame 缺少字段："
                f"{sorted(missing_columns)}"
            )

        end_dt = _to_python_datetime(
            df_i.attrs.get(
                "end_dt",
                None,
            )
        )

        if end_dt is None:
            end_dt = (
                start_dt
                + datetime.timedelta(
                    seconds=float(
                        df_i["t_end"].max()
                    )
                )
            )

        lights_off_dt = (
            _resolve_lights_off_datetime(
                start_dt=start_dt,
                end_dt=end_dt,
                lights_off_value=lights_off_value,
                boundary_tolerance_sec=boundary_tolerance_sec,
            )
        )

        elapsed_seconds = float(
            (
                lights_off_dt
                - start_dt
            ).total_seconds()
        )

        # Excel 按分钟记录，EDF 可能从 xx:xx:30 开始
        if (
            elapsed_seconds < 0
            and abs(elapsed_seconds)
            <= boundary_tolerance_sec
        ):
            elapsed_seconds = 0.0

        recording_duration = float(
            df_i["t_end"].max()
        )

        if elapsed_seconds < 0:
            raise ValueError(
                "lights off 早于 EDF 起始时间，"
                "且超出允许误差：\n"
                f"elapsed_seconds={elapsed_seconds}"
            )

        if elapsed_seconds > recording_duration:
            excess = (
                elapsed_seconds
                - recording_duration
            )

            if excess <= boundary_tolerance_sec:
                elapsed_seconds = np.nextafter(
                    recording_duration,
                    0,
                )
            else:
                raise ValueError(
                    "lights off 晚于 EDF 结束时间：\n"
                    f"elapsed_seconds={elapsed_seconds}\n"
                    f"duration={recording_duration}"
                )

        # 每个正常 epoch 为 30 秒
        target_epoch_sec = float(
            df_i.attrs.get(
                "target_epoch_sec",
                30.0,
            )
        )

        epoch_index = int(
            np.floor(
                elapsed_seconds
                / target_epoch_sec
            )
        )

        # 恰好位于记录终点时归入最后一个 epoch
        epoch_index = min(
            epoch_index,
            len(df_i) - 1,
        )

        epoch_index = max(
            epoch_index,
            0,
        )

        return int(
            epoch_index
        )

    # Telemetry original cell 2, line 1537.
    def build_st_matched_table(
        all_epochs: List[pd.DataFrame],
        night_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        根据 subject/night 匹配：

            all_epochs
            ST-subjects.xls

        并计算每晚 lights off epoch。
        """
        rows = []

        for epoch_list_index, df_i in enumerate(
            all_epochs
        ):
            if len(df_i) == 0:
                raise ValueError(
                    f"all_epochs[{epoch_list_index}] 为空。"
                )

            subject = int(
                df_i.attrs.get(
                    "subject",
                    df_i["subject"].iloc[0],
                )
            )

            night = int(
                df_i.attrs.get(
                    "night",
                    df_i["night"].iloc[0],
                )
            )

            file_name = str(
                df_i.attrs.get(
                    "file_name",
                    df_i[
                        "file_name"
                    ].iloc[0],
                )
            )

            file_path = str(
                df_i.attrs.get(
                    "file_path",
                    df_i[
                        "file_path"
                    ].iloc[0],
                )
            )

            start_dt = _to_python_datetime(
                df_i.attrs.get(
                    "start_dt",
                    None,
                )
            )

            end_dt = _to_python_datetime(
                df_i.attrs.get(
                    "end_dt",
                    None,
                )
            )

            if start_dt is None:
                raise ValueError(
                    f"{file_name} 缺少 start_dt。"
                )

            if end_dt is None:
                end_dt = (
                    start_dt
                    + datetime.timedelta(
                        seconds=float(
                            df_i[
                                "t_end"
                            ].max()
                        )
                    )
                )

            rows.append(
                {
                    "epoch_list_index": int(
                        epoch_list_index
                    ),
                    "file": file_name,
                    "file_path": file_path,
                    "subject": subject,
                    "night": night,
                    "start_utc": start_dt,
                    "duration_s": float(
                        (
                            end_dt
                            - start_dt
                        ).total_seconds()
                    ),
                    "end_utc": end_dt,
                }
            )

        rows_df = pd.DataFrame(
            rows
        )

        duplicated_edf = (
            rows_df.duplicated(
                subset=[
                    "subject",
                    "night",
                ],
                keep=False,
            )
        )

        if duplicated_edf.any():
            raise ValueError(
                "多个 EDF 对应同一个 subject/night：\n"
                + rows_df.loc[
                    duplicated_edf,
                    [
                        "file",
                        "subject",
                        "night",
                    ],
                ].to_string(
                    index=False
                )
            )

        # 使用 outer，既检查缺少 Excel 行，也检查缺少 EDF 文件
        matched_df = rows_df.merge(
            night_df,
            on=[
                "subject",
                "night",
            ],
            how="outer",
            validate="one_to_one",
            indicator=True,
            sort=False,
        )

        unmatched = matched_df.loc[
            matched_df["_merge"] != "both"
        ].copy()

        if len(unmatched) > 0:
            raise ValueError(
                "EDF 与 ST-subjects.xls 没有完全匹配：\n"
                + unmatched[
                    [
                        "file",
                        "subject",
                        "night",
                        "condition",
                        "_merge",
                    ]
                ].to_string(
                    index=False
                )
            )

        matched_df = matched_df.drop(
            columns="_merge"
        )

        matched_df[
            "epoch_list_index"
        ] = matched_df[
            "epoch_list_index"
        ].astype(int)

        matched_df = matched_df.sort_values(
            [
                "subject",
                "night",
            ],
            kind="stable",
        ).reset_index(drop=True)

        lights_off_indices = []

        for row in matched_df.itertuples(
            index=False
        ):
            df_i = all_epochs[
                int(
                    row.epoch_list_index
                )
            ]

            try:
                lights_off_idx = (
                    lights_off_epoch_idx_by_tstart_end(
                        df_i=df_i,
                        start_dt=row.start_utc,
                        lights_off_value=row.lights_off,
                    )
                )

            except Exception as exc:
                raise RuntimeError(
                    "计算 lights off epoch 失败：\n"
                    f"file      = {row.file}\n"
                    f"subject   = {row.subject}\n"
                    f"night     = {row.night}\n"
                    f"condition = {row.condition}\n"
                    f"lights off= {row.lights_off!r}\n"
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            lights_off_indices.append(
                int(lights_off_idx)
            )

            # 把匹配结果保存在每个 DataFrame 的 attrs 中
            df_i.attrs["condition"] = str(
                row.condition
            )

            df_i.attrs["age"] = (
                row.age
            )

            df_i.attrs["sex_code"] = (
                row.sex_code
            )

            df_i.attrs["lights_off"] = (
                row.lights_off
            )

            df_i.attrs[
                "lights_off_epoch_30s"
            ] = int(
                lights_off_idx
            )

        matched_df[
            "lights_off_epoch_30s"
        ] = np.asarray(
            lights_off_indices,
            dtype=int,
        )

        return rows_df, matched_df

    return {'load_sleepstageage': load_sleepstageage, 'read_st_subjects': read_st_subjects, 'build_st_matched_table': build_st_matched_table}

_drug_loading_helpers = _build_drug_loading_helpers()
load_telemetry_hypnograms = _drug_loading_helpers['load_sleepstageage']
read_telemetry_subjects = _drug_loading_helpers['read_st_subjects']
match_telemetry_records = _drug_loading_helpers['build_st_matched_table']


def _build_drug_metric_helpers():
    # Telemetry original cell 6, line 1.
    import warnings

    # Telemetry original cell 7, line 9.
    import pandas as pd

    # Telemetry original cell 19, line 1.
    import numpy as np

    # Telemetry original cell 6, line 5.
    def generate_group_sleep_indices(
        epochs_group,
        lights_off_group,
        subjects_group,
        group_name,
    ):
        """
        为一个条件组生成：

            Labels:
                完整 hypnogram 标签。

            FirstIdx:
                入睡第一个时刻相对于 lights off 的 epoch 偏移。

            LastIdx:
                最终醒来第一个时刻相对于 lights off 的 epoch 偏移。

            FirstAbsIdx:
                入睡第一个时刻在完整 hypnogram 中的绝对 index。

            LastAbsIdx:
                最终醒来第一个时刻在完整 hypnogram 中的绝对 index。

        睡眠阶段编码：
            Wake = 0
            N1   = 1
            N2   = 2
            N3   = 3
            REM  = 4
            Unknown = -1
        """
        if len(epochs_group) != len(lights_off_group):
            raise ValueError(
                f"{group_name}：Epochs 和 LOffIdx 数量不同："
                f"{len(epochs_group)} vs {len(lights_off_group)}"
            )

        if len(epochs_group) != len(subjects_group):
            raise ValueError(
                f"{group_name}：Epochs 和 Subjects 数量不同："
                f"{len(epochs_group)} vs {len(subjects_group)}"
            )

        labels_group = []

        first_relative_group = []
        last_relative_group = []

        first_absolute_group = []
        last_absolute_group = []

        sleep_codes = np.array(
            [1, 2, 3, 4],
            dtype=int,
        )

        for j, df_i in enumerate(epochs_group):

            if "stage" not in df_i.columns:
                raise KeyError(
                    f"{group_name} 第 {j} 条记录没有 stage 列。"
                )

            labels = df_i[
                "stage"
            ].to_numpy(dtype=int)

            labels_group.append(labels)

            lights_off_idx = int(
                lights_off_group[j]
            )

            if not (
                0 <= lights_off_idx < len(labels)
            ):
                raise ValueError(
                    f"{group_name} subject={subjects_group[j]}："
                    f"lights off index 超出范围。\n"
                    f"lights_off_idx={lights_off_idx}, "
                    f"hypnogram length={len(labels)}"
                )

            # ====================================================
            # 入睡：lights off 后第一个睡眠 epoch
            # ====================================================

            sleep_after_lights_off = np.flatnonzero(
                np.isin(
                    labels[lights_off_idx:],
                    sleep_codes,
                )
            )

            if len(sleep_after_lights_off) == 0:
                raise ValueError(
                    f"{group_name} subject={subjects_group[j]}："
                    "lights off 之后没有找到睡眠阶段。"
                )

            first_sleep_abs = (
                lights_off_idx
                + int(sleep_after_lights_off[0])
            )

            # ====================================================
            # 最终醒来：
            # 找到入睡后最后一个睡眠 epoch，
            # 再寻找其后的第一个 Wake epoch
            # ====================================================

            all_sleep_after_onset = np.flatnonzero(
                np.isin(
                    labels[first_sleep_abs:],
                    sleep_codes,
                )
            )

            last_sleep_abs = (
                first_sleep_abs
                + int(all_sleep_after_onset[-1])
            )

            wake_after_last_sleep = np.flatnonzero(
                labels[
                    last_sleep_abs + 1:
                ] == 0
            )

            if len(wake_after_last_sleep) > 0:
                first_final_wake_abs = (
                    last_sleep_abs
                    + 1
                    + int(wake_after_last_sleep[0])
                )

            else:
                # 极少数记录可能在最后一个睡眠 epoch 后
                # 没有正式标注 Wake。此时使用记录末尾作为边界，
                # 同时给出警告，避免程序直接中断。
                first_final_wake_abs = (
                    len(labels) - 1
                )

                warnings.warn(
                    f"{group_name} subject={subjects_group[j]}："
                    "最后一个睡眠 epoch 后没有找到 Wake，"
                    "暂时使用 hypnogram 最后一个 epoch 作为结束位置。"
                )

            # 相对于 lights off 的偏移
            first_sleep_relative = (
                first_sleep_abs
                - lights_off_idx
            )

            first_final_wake_relative = (
                first_final_wake_abs
                - lights_off_idx
            )

            first_absolute_group.append(
                int(first_sleep_abs)
            )

            last_absolute_group.append(
                int(first_final_wake_abs)
            )

            first_relative_group.append(
                int(first_sleep_relative)
            )

            last_relative_group.append(
                int(first_final_wake_relative)
            )

        return (
            labels_group,
            np.asarray(
                first_relative_group,
                dtype=int,
            ),
            np.asarray(
                last_relative_group,
                dtype=int,
            ),
            np.asarray(
                first_absolute_group,
                dtype=int,
            ),
            np.asarray(
                last_absolute_group,
                dtype=int,
            ),
        )

    # Telemetry original cell 7, line 19.
    def compute_sleep_metrics_from_labels(
        labels_list,
        epoch_duration_sec=30.0,
        wake_state=0,
        sleep_states=(1, 2, 3, 4),
        missing_state=-1,
    ):
        """
        根据 hypnogram 计算睡眠指标。

        状态编码
        --------
        -1 : 未记录
         0 : Wake
         1 : N1
         2 : N2
         3 : N3
         4 : REM

        指标定义
        --------
        TST_min:
            实际睡眠总时长，只统计 sleep_states。

        scored_TIB_min:
            整个 hypnogram 中有有效标注的时长，排除 -1。

        scored_SPT_min:
            从首次睡眠 epoch 到最后一个睡眠 epoch，
            排除 -1 后的有效时长。

        WASO_min:
            从首次睡眠到最后一次睡眠之间的 Wake 总时长。

        SPE_percent:
            TST / scored SPT × 100%。

        SE_percent:
            TST / scored TIB × 100%。

        awakening_count:
            sleep period 中相邻的 Sleep -> Wake 转移次数。
            不跨越 -1 统计转移。

        awakening_frequency_per_hour_sleep:
            awakening_count / TST_hours，
            单位为每小时实际睡眠的觉醒次数。
        """

        epoch_duration_min = epoch_duration_sec / 60.0

        allowed_states = np.asarray(
            (missing_state, wake_state) + tuple(sleep_states),
            dtype=int,
        )

        rows = []

        for subject_index, labels in enumerate(labels_list):

            labels = np.asarray(labels).ravel()

            if labels.size == 0:
                raise ValueError(
                    f"第 {subject_index} 条 hypnogram 为空。"
                )

            # 转换成数值
            try:
                labels_float = labels.astype(float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"第 {subject_index} 条 hypnogram "
                    "包含不能转换为数值的内容。"
                ) from error

            if np.any(~np.isfinite(labels_float)):
                raise ValueError(
                    f"第 {subject_index} 条 hypnogram "
                    "包含 NaN 或 inf。"
                )

            # 确保状态值均为整数
            if not np.allclose(
                labels_float,
                np.round(labels_float),
            ):
                raise ValueError(
                    f"第 {subject_index} 条 hypnogram "
                    "包含非整数状态编码。"
                )

            labels = labels_float.astype(int)

            invalid_states = np.setdiff1d(
                np.unique(labels),
                allowed_states,
            )

            if invalid_states.size > 0:
                raise ValueError(
                    f"第 {subject_index} 条 hypnogram "
                    f"包含未定义状态：{invalid_states.tolist()}。\n"
                    f"允许状态：{allowed_states.tolist()}"
                )

            # ----------------------------------------------------
            # 基本状态 mask
            # ----------------------------------------------------

            missing_mask = (
                labels == missing_state
            )

            sleep_mask = np.isin(
                labels,
                sleep_states,
            )

            scored_mask = ~missing_mask

            sleep_indices = np.flatnonzero(
                sleep_mask
            )

            # ----------------------------------------------------
            # 整段记录指标
            # ----------------------------------------------------

            total_elapsed_epochs = len(labels)

            scored_tib_epochs = int(
                scored_mask.sum()
            )

            missing_epochs = int(
                missing_mask.sum()
            )

            tst_epochs = int(
                sleep_mask.sum()
            )

            total_elapsed_min = (
                total_elapsed_epochs
                * epoch_duration_min
            )

            scored_tib_min = (
                scored_tib_epochs
                * epoch_duration_min
            )

            missing_min = (
                missing_epochs
                * epoch_duration_min
            )

            tst_min = (
                tst_epochs
                * epoch_duration_min
            )

            tst_hours = (
                tst_min / 60.0
            )

            # ----------------------------------------------------
            # 没有任何睡眠 epoch
            # ----------------------------------------------------

            if sleep_indices.size == 0:

                rows.append(
                    {
                        "subject_index": subject_index,
                        "total_elapsed_min": total_elapsed_min,
                        "scored_TIB_min": scored_tib_min,
                        "missing_min": missing_min,
                        "TST_min": 0.0,
                        "scored_SPT_min": 0.0,
                        "missing_within_SPT_min": 0.0,
                        "WASO_min": 0.0,
                        "SPE_percent": np.nan,
                        "SE_percent": (
                            0.0
                            if scored_tib_epochs > 0
                            else np.nan
                        ),
                        "awakening_count": 0,
                        "awakening_frequency_per_hour_sleep": np.nan,
                    }
                )

                continue

            # ----------------------------------------------------
            # 从首次睡眠到最后一次睡眠
            # 保留原始位置，不能先删除 -1
            # ----------------------------------------------------

            sleep_onset_idx = int(
                sleep_indices[0]
            )

            final_sleep_idx = int(
                sleep_indices[-1]
            )

            sleep_period_labels = labels[
                sleep_onset_idx:
                final_sleep_idx + 1
            ]

            sleep_period_missing_mask = (
                sleep_period_labels == missing_state
            )

            sleep_period_scored_mask = (
                ~sleep_period_missing_mask
            )

            sleep_period_sleep_mask = np.isin(
                sleep_period_labels,
                sleep_states,
            )

            sleep_period_wake_mask = (
                sleep_period_labels == wake_state
            )

            # ----------------------------------------------------
            # SPT
            # ----------------------------------------------------

            scored_spt_epochs = int(
                sleep_period_scored_mask.sum()
            )

            missing_within_spt_epochs = int(
                sleep_period_missing_mask.sum()
            )

            scored_spt_min = (
                scored_spt_epochs
                * epoch_duration_min
            )

            missing_within_spt_min = (
                missing_within_spt_epochs
                * epoch_duration_min
            )

            # ----------------------------------------------------
            # WASO
            # ----------------------------------------------------

            waso_epochs = int(
                sleep_period_wake_mask.sum()
            )

            waso_min = (
                waso_epochs
                * epoch_duration_min
            )

            # ----------------------------------------------------
            # SPE
            # ----------------------------------------------------

            sleep_epochs_within_spt = int(
                sleep_period_sleep_mask.sum()
            )

            spe_percent = (
                100.0
                * sleep_epochs_within_spt
                / scored_spt_epochs
                if scored_spt_epochs > 0
                else np.nan
            )

            # ----------------------------------------------------
            # SE
            # ----------------------------------------------------

            se_percent = (
                100.0
                * tst_epochs
                / scored_tib_epochs
                if scored_tib_epochs > 0
                else np.nan
            )

            # ----------------------------------------------------
            # 觉醒次数
            # 严格统计相邻的 Sleep -> Wake
            #
            # Sleep, -1, Wake 不算一次觉醒，
            # 因为真实转移未被连续观测到。
            # ----------------------------------------------------

            previous_is_sleep = np.isin(
                sleep_period_labels[:-1],
                sleep_states,
            )

            current_is_wake = (
                sleep_period_labels[1:]
                == wake_state
            )

            awakening_count = int(
                np.sum(
                    previous_is_sleep
                    & current_is_wake
                )
            )

            # ----------------------------------------------------
            # 觉醒频率：次 / 小时睡眠
            # ----------------------------------------------------

            awakening_frequency_per_hour_sleep = (
                awakening_count / tst_hours
                if tst_hours > 0
                else np.nan
            )

            rows.append(
                {
                    "subject_index": subject_index,
                    "total_elapsed_min": total_elapsed_min,
                    "scored_TIB_min": scored_tib_min,
                    "missing_min": missing_min,
                    "TST_min": tst_min,
                    "scored_SPT_min": scored_spt_min,
                    "missing_within_SPT_min": (
                        missing_within_spt_min
                    ),
                    "WASO_min": waso_min,
                    "SPE_percent": spe_percent,
                    "SE_percent": se_percent,
                    "awakening_count": awakening_count,
                    "awakening_frequency_per_hour_sleep": (
                        awakening_frequency_per_hour_sleep
                    ),
                }
            )

        return pd.DataFrame(rows)

    # Telemetry original cell 19, line 4.
    def calculate_tmat_ignore_invalid(
        hypnogram,
        n_states=5,
        invalid_value=-1,
    ):
        """
        根据一条 hypnogram 计算转移矩阵。

        只统计相邻且均为有效状态 0, 1, 2, 3, 4 的转移。
        任何包含 -1 的相邻对都不计入统计。

        例如：
            [0, 0, -1, 2, 2]

        只统计：
            0 -> 0
            2 -> 2

        不会统计：
            0 -> -1
            -1 -> 2
            0 -> 2

        返回
        ----
        Tmat : ndarray, shape (5, 5)
            行归一化转移概率，row -> column。

        count_mat : ndarray, shape (5, 5)
            归一化前的转移次数。

        valid_pair_count : int
            被计入统计的有效相邻转移总数。
        """
        hypnogram = np.asarray(
            hypnogram
        ).reshape(-1)

        if len(hypnogram) < 2:
            return (
                np.zeros(
                    (n_states, n_states),
                    dtype=float,
                ),
                np.zeros(
                    (n_states, n_states),
                    dtype=int,
                ),
                0,
            )

        state_from = hypnogram[:-1]
        state_to = hypnogram[1:]

        # 前后两个状态都必须属于 0,...,n_states-1
        valid_pairs = (
            (state_from != invalid_value)
            & (state_to != invalid_value)
            & (state_from >= 0)
            & (state_from < n_states)
            & (state_to >= 0)
            & (state_to < n_states)
        )

        valid_from = state_from[
            valid_pairs
        ].astype(int)

        valid_to = state_to[
            valid_pairs
        ].astype(int)

        count_mat = np.zeros(
            (n_states, n_states),
            dtype=int,
        )

        if len(valid_from) > 0:
            np.add.at(
                count_mat,
                (valid_from, valid_to),
                1,
            )

        row_sum = count_mat.sum(
            axis=1,
            keepdims=True,
        )

        Tmat = np.divide(
            count_mat,
            row_sum,
            out=np.zeros_like(
                count_mat,
                dtype=float,
            ),
            where=row_sum > 0,
        )

        return (
            Tmat,
            count_mat,
            int(valid_pairs.sum()),
        )

    # Telemetry original cell 19, line 110.
    def calculate_pss_ignore_invalid(
        hypnogram,
        n_states=5,
        invalid_value=-1,
    ):
        """
        计算五个有效睡眠状态的经验占据概率。

        -1 不计入任何状态，也不计入归一化分母。
        """
        hypnogram = np.asarray(
            hypnogram
        ).reshape(-1)

        valid_mask = (
            (hypnogram != invalid_value)
            & (hypnogram >= 0)
            & (hypnogram < n_states)
        )

        valid_states = hypnogram[
            valid_mask
        ].astype(int)

        counts = np.bincount(
            valid_states,
            minlength=n_states,
        ).astype(float)

        total = counts.sum()

        if total == 0:
            return np.zeros(
                n_states,
                dtype=float,
            )

        return counts / total

    # Telemetry original cell 19, line 150.
    def calculate_tmat_list_ignore_invalid(
        hypnogram_list,
        n_states=5,
        invalid_value=-1,
    ):
        """
        对一个 hypnogram list 中的每条序列分别计算：

            Pss
            Tmat
            transition count matrix
            有效转移总数

        返回值均按照 hypnogram_list 的顺序排列。
        """
        Pss_list = []
        Tmat_list = []
        CountMat_list = []
        ValidTransitionCount = []

        for subject_index, hypnogram in enumerate(
            hypnogram_list
        ):
            pss = calculate_pss_ignore_invalid(
                hypnogram=hypnogram,
                n_states=n_states,
                invalid_value=invalid_value,
            )

            (
                tmat,
                count_mat,
                valid_pair_count,
            ) = calculate_tmat_ignore_invalid(
                hypnogram=hypnogram,
                n_states=n_states,
                invalid_value=invalid_value,
            )

            Pss_list.append(
                pss
            )

            Tmat_list.append(
                tmat
            )

            CountMat_list.append(
                count_mat
            )

            ValidTransitionCount.append(
                valid_pair_count
            )

        return (
            Pss_list,
            Tmat_list,
            CountMat_list,
            np.asarray(
                ValidTransitionCount,
                dtype=int,
            ),
        )

    return {'generate_group_sleep_indices': generate_group_sleep_indices, 'compute_sleep_metrics_from_labels': compute_sleep_metrics_from_labels, 'calculate_tmat_list_ignore_invalid': calculate_tmat_list_ignore_invalid}

_drug_metric_helpers = _build_drug_metric_helpers()
generate_group_sleep_indices = _drug_metric_helpers['generate_group_sleep_indices']
compute_sleep_metrics_from_labels = _drug_metric_helpers['compute_sleep_metrics_from_labels']
calculate_tmat_list_ignore_invalid = _drug_metric_helpers['calculate_tmat_list_ignore_invalid']


def _build_drug_force_helpers():
    # Telemetry original cell 58, line 1.
    import numpy as np

    # Telemetry original cell 58, line 2.
    import pandas as pd

    # Telemetry original cell 58, line 5.
    STATE_NAME_MAP = {
        0: "Wake",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM",
    }

    # Telemetry original cell 58, line 14.
    def to_1d_hypnogram_force(hypnogram):
        h0 = np.asarray(hypnogram)
        h = np.squeeze(h0)

        if h.ndim == 1:
            return h.astype(int)

        if h.ndim == 2 and 1 in h.shape:
            return h.reshape(-1).astype(int)

        raise ValueError(
            f"Invalid hypnogram shape: original={h0.shape}, squeezed={h.shape}"
        )

    # Telemetry original cell 58, line 29.
    def iter_hypnograms_force(data):
        if isinstance(data, dict):
            items = sorted(data.items(), key=lambda kv: kv[0])
            return [(i, k, v) for i, (k, v) in enumerate(items)]

        arr = np.asarray(data)

        if arr.ndim == 1:
            return [(0, 0, data)]

        if arr.ndim == 2 and arr.dtype != object:
            return [(i, i, arr[i]) for i in range(arr.shape[0])]

        return [(i, i, hyp) for i, hyp in enumerate(data)]

    # Telemetry original cell 58, line 45.
    def find_runs_force(x):
        x = np.asarray(x)

        if len(x) == 0:
            return (
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )

        change = np.where(np.diff(x) != 0)[0] + 1
        starts = np.r_[0, change]
        ends = np.r_[change - 1, len(x) - 1]
        states = x[starts]

        return states.astype(int), starts.astype(int), ends.astype(int)

    # Telemetry original cell 58, line 63.
    def compress_consecutive_force(seq):
        seq = list(seq)

        if len(seq) == 0:
            return []

        out = [int(seq[0])]

        for x in seq[1:]:
            x = int(x)
            if x != out[-1]:
                out.append(x)

        return out

    # Telemetry original cell 58, line 79.
    def structure_to_code_force(seq):
        if seq is None:
            return ""

        if isinstance(seq, str):
            return (
                seq.replace(" ", "")
                   .replace(",", "")
                   .replace("-", "")
                   .replace("_", "")
                   .replace("→", "")
            )

        return "".join(str(int(x)) for x in seq)

    # Telemetry original cell 58, line 95.
    def structure_to_tuple_force(seq):
        if seq is None:
            return tuple()

        if isinstance(seq, str):
            s = structure_to_code_force(seq)
            return tuple(int(ch) for ch in s if ch.isdigit())

        return tuple(int(x) for x in seq)

    # Telemetry original cell 58, line 106.
    def structure_to_name_force(seq, state_name_map=None, arrow="→"):
        if state_name_map is None:
            state_name_map = STATE_NAME_MAP

        return arrow.join(state_name_map.get(int(x), str(x)) for x in seq)

    # Telemetry original cell 58, line 113.
    def close_loop_force(loop):
        loop = structure_to_tuple_force(loop)

        if len(loop) == 0:
            return tuple()

        if loop[0] != loop[-1]:
            loop = loop + (loop[0],)

        return loop

    # Telemetry original cell 58, line 125.
    def canonical_rotation_loop_force(loop):
        loop = close_loop_force(loop)

        if len(loop) <= 1:
            return loop

        core = loop[:-1]
        rotations = []

        for k in range(len(core)):
            r = core[k:] + core[:k]
            rotations.append(r + (r[0],))

        return sorted(rotations, key=structure_to_code_force)[0]

    # Telemetry original cell 58, line 141.
    def get_key_for_dict_force(hyp_index, subject, d):
        if subject in d:
            return subject
        if hyp_index in d:
            return hyp_index
        return None

    # Telemetry original cell 58, line 149.
    def get_flux_key_from_row_force(row, AllCycle):
        hyp_index = int(row["hyp_index"])
        subject = row.get("subject", hyp_index)

        if subject in AllCycle:
            return subject

        if hyp_index in AllCycle:
            return hyp_index

        return None

    # Telemetry original cell 58, line 162.
    def get_strength_list_for_hypnogram(hyp_index, subject, NocAllStrength):
        key = get_key_for_dict_force(hyp_index, subject, NocAllStrength)

        if key is None:
            return []

        return list(NocAllStrength.get(key, []))

    # Telemetry original cell 58, line 171.
    def get_sym_matrix_for_hypnogram(hyp_index, subject, NocFlux_sym_list):
        if isinstance(NocFlux_sym_list, dict):
            if subject in NocFlux_sym_list:
                return np.asarray(NocFlux_sym_list[subject], dtype=float)
            if hyp_index in NocFlux_sym_list:
                return np.asarray(NocFlux_sym_list[hyp_index], dtype=float)
            return np.zeros((5, 5), dtype=float)

        return np.asarray(NocFlux_sym_list[hyp_index], dtype=float)

    # Telemetry original cell 58, line 182.
    def extract_raw_nremrem_cycles_for_force(
        data,
        epoch_sec=30,
        wake_state=0,
        rem_state=4,
        nrem_states=(1, 2, 3),
        min_nrem_min=15,
        min_rem_min=5,
        include_post_rem_state=True,
        include_wake_in_structure=False,
    ):
        """
        Extract raw NREM-REM cycles. No reversible transitions are removed.
        """

        all_rows = []

        min_nrem_epochs = int(np.ceil(min_nrem_min * 60 / epoch_sec))
        min_rem_epochs = int(np.ceil(min_rem_min * 60 / epoch_sec))

        for hyp_index, subject, hyp in iter_hypnograms_force(data):
            h = to_1d_hypnogram_force(hyp)

            sleep_states = set(nrem_states) | {rem_state}
            sleep_onset_candidates = np.where(np.isin(h, list(sleep_states)))[0]

            if len(sleep_onset_candidates) == 0:
                continue

            sleep_onset = int(sleep_onset_candidates[0])

            run_states, run_starts, run_ends = find_runs_force(h)

            rem_runs = []
            for s, st, ed in zip(run_states, run_starts, run_ends):
                if int(s) == rem_state:
                    rem_runs.append((int(st), int(ed)))

            cycle_start = sleep_onset
            cycle_index = 0

            for rem_start, rem_end in rem_runs:
                if rem_end < cycle_start:
                    continue

                pre_rem_segment = h[cycle_start:rem_start]

                nrem_count = int(np.sum(np.isin(pre_rem_segment, nrem_states)))
                rem_count = int(rem_end - rem_start + 1)

                if nrem_count >= min_nrem_epochs and rem_count >= min_rem_epochs:
                    segment_start = int(cycle_start)
                    segment_end = int(rem_end)

                    if include_post_rem_state and rem_end + 1 < len(h):
                        structure_end = int(rem_end + 1)
                    else:
                        structure_end = int(rem_end)

                    segment = h[segment_start:structure_end + 1]

                    if include_wake_in_structure:
                        structure = compress_consecutive_force(segment)
                    else:
                        segment_no_wake = [
                            x for x in segment if int(x) != wake_state
                        ]
                        structure = compress_consecutive_force(segment_no_wake)

                    structure = tuple(structure)

                    duration_epochs = int(segment_end - segment_start + 1)
                    duration_min = duration_epochs * epoch_sec / 60.0

                    all_rows.append({
                        "hyp_index": int(hyp_index),
                        "subject": subject,
                        "cycle_index": int(cycle_index),

                        "cycle_start_epoch": int(segment_start),
                        "cycle_end_epoch": int(segment_end),
                        "structure_end_epoch": int(structure_end),

                        "rem_start_epoch": int(rem_start),
                        "rem_end_epoch": int(rem_end),

                        "nrem_epochs_before_rem": int(nrem_count),
                        "rem_epochs": int(rem_count),

                        "duration_epochs": int(duration_epochs),
                        "duration_min": float(duration_min),

                        "structure": structure,
                        "structure_code": structure_to_code_force(structure),
                        "structure_name": structure_to_name_force(structure),
                    })

                    cycle_index += 1
                    cycle_start = rem_end + 1

        return pd.DataFrame(all_rows)

    # Telemetry original cell 58, line 285.
    def all_directed_subpaths_from_closed_loop_force(loop):
        loop = close_loop_force(loop)

        if len(loop) <= 1:
            return []

        core = loop[:-1]
        m = len(core)

        subpaths = []

        for edge_len in range(1, m + 1):
            for start in range(m):
                path = []
                for k in range(edge_len + 1):
                    path.append(core[(start + k) % m])
                subpaths.append(tuple(path))

        return subpaths

    # Telemetry original cell 58, line 306.
    def build_flux_loop_segment_catalog_with_strength(
        flux_loops,
        flux_strengths,
    ):
        """
        Build directed segment catalog from one hypnogram's flux loops.

        Each support has:
            loop_index
            loop_strength
            segment
        """

        catalog = {}

        flux_strengths = list(flux_strengths)

        for loop_index, loop in enumerate(flux_loops):
            loop_closed = close_loop_force(loop)

            if len(loop_closed) <= 1:
                continue

            loop_canonical = canonical_rotation_loop_force(loop_closed)
            loop_code = structure_to_code_force(loop_canonical)
            loop_name = structure_to_name_force(loop_canonical)
            loop_n_edges = len(loop_canonical) - 1

            if loop_index < len(flux_strengths):
                loop_strength = float(flux_strengths[loop_index])
            else:
                loop_strength = 0.0

            subpaths = all_directed_subpaths_from_closed_loop_force(loop_canonical)

            for segment in subpaths:
                used_edges = len(segment) - 1

                if used_edges <= 0:
                    continue

                info = {
                    "loop_index": int(loop_index),
                    "loop": loop_canonical,
                    "loop_code": loop_code,
                    "loop_name": loop_name,
                    "loop_n_edges": int(loop_n_edges),

                    "loop_strength": float(loop_strength),

                    "segment": segment,
                    "segment_code": structure_to_code_force(segment),
                    "segment_name": structure_to_name_force(segment),

                    "used_edges": int(used_edges),
                }

                catalog.setdefault(segment, []).append(info)

        for segment in catalog:
            catalog[segment] = sorted(
                catalog[segment],
                key=lambda x: (
                    x["loop_strength"],
                    x["used_edges"],
                    -x["loop_n_edges"],
                    x["loop_code"],
                ),
                reverse=True,
            )

        return catalog

    # Telemetry original cell 58, line 380.
    def decompose_cycle_by_flux_loop_segments_force(
        target_structure,
        catalog,
        min_segment_edges=1,
        allow_loop_reuse=True,
    ):
        """
        Partial decomposition is allowed.

        It maximizes:
            1. number of covered edges
            2. fewer segments
            3. larger total selected loop strength
        """

        target = tuple(int(x) for x in target_structure)

        if len(target) <= 1:
            return [], tuple(), 0.0

        target_edges = [(target[i], target[i + 1]) for i in range(len(target) - 1)]
        n_target_edges = len(target_edges)
        n_states = len(target)

        memo = {}

        def better(a, b):
            if a is None:
                return b
            if b is None:
                return a

            key_a = (a[0], a[1], a[2])
            key_b = (b[0], b[1], b[2])

            if key_b > key_a:
                return b
            return a

        def solve(pos, used_loop_indices):
            if pos >= n_states - 1:
                return (0, 0, 0.0, [])

            if allow_loop_reuse:
                key = (pos, ())
            else:
                key = (pos, tuple(sorted(used_loop_indices)))

            if key in memo:
                return memo[key]

            best = None

            skipped = solve(pos + 1, used_loop_indices)
            best = better(best, skipped)

            for next_pos in range(pos + 1, n_states):
                segment = tuple(target[pos:next_pos + 1])
                segment_edges = len(segment) - 1

                if segment_edges < min_segment_edges:
                    continue

                if segment not in catalog:
                    continue

                for support in catalog[segment]:
                    loop_index = int(support["loop_index"])

                    if (not allow_loop_reuse) and (loop_index in used_loop_indices):
                        continue

                    if allow_loop_reuse:
                        new_used = used_loop_indices
                    else:
                        new_used = set(used_loop_indices)
                        new_used.add(loop_index)
                        new_used = frozenset(new_used)

                    rest = solve(next_pos, new_used)

                    if rest is None:
                        continue

                    piece = dict(support)
                    piece["target_start_state_index"] = pos
                    piece["target_end_state_index"] = next_pos
                    piece["target_start_edge_index"] = pos
                    piece["target_end_edge_index"] = next_pos - 1

                    candidate_pieces = [piece] + rest[3]

                    candidate = (
                        segment_edges + rest[0],
                        -len(candidate_pieces),
                        support["loop_strength"] + rest[2],
                        candidate_pieces,
                    )

                    best = better(best, candidate)

            memo[key] = best
            return best

        solution = solve(0, frozenset())

        if solution is None:
            pieces = []
        else:
            pieces = solution[3]

        covered_edge_indices = set()

        for p in pieces:
            for eidx in range(
                int(p["target_start_edge_index"]),
                int(p["target_end_edge_index"]) + 1,
            ):
                covered_edge_indices.add(eidx)

        missing_edges = tuple(
            edge for idx, edge in enumerate(target_edges)
            if idx not in covered_edge_indices
        )

        coverage_fraction = (
            len(covered_edge_indices) / n_target_edges
            if n_target_edges > 0 else 0.0
        )

        return pieces, missing_edges, coverage_fraction

    # Telemetry original cell 58, line 512.
    def analyze_nremrem_loop_flux_plus_sym_force_by_hypnogram(
        data,
        AllCycle,
        AllStrength,
        Flux_sym_list,
        temporal_cycle_occurrence_df=None,
        epoch_sec=30,
        wake_state=0,
        rem_state=4,
        nrem_states=(1, 2, 3),
        min_nrem_min=15,
        min_rem_min=5,
        include_post_rem_state=True,
        include_wake_in_structure=False,
        min_segment_edges=1,
        allow_loop_reuse=True,
        count_each_loop_once_per_cycle=True,
        sum_unique_missing_edges=False,

        # 新增选项：
        # False: 每个 loop 贡献 = loop flux strength
        # True : 每个 loop 贡献 = loop flux strength × loop 边数
        multiply_loop_flux_by_n_edges=False,
    ):
        """
        For each raw NREM-REM cycle:
            force_estimate =
                sum of selected loop flux strengths
                +
                sum of symmetric flux values for uncovered directed edges.

        If multiply_loop_flux_by_n_edges=True:
            each loop contribution becomes:
                loop_strength * number_of_edges_in_loop

        Important:
            loop flux is not weighted by used-edge fraction.
            If count_each_loop_once_per_cycle=True, each loop contributes once per NREM-REM cycle,
            even if used multiple times.
        """

        df_raw_occ = extract_raw_nremrem_cycles_for_force(
            data=data,
            epoch_sec=epoch_sec,
            wake_state=wake_state,
            rem_state=rem_state,
            nrem_states=nrem_states,
            min_nrem_min=min_nrem_min,
            min_rem_min=min_rem_min,
            include_post_rem_state=include_post_rem_state,
            include_wake_in_structure=include_wake_in_structure,
        )

        if len(df_raw_occ) == 0:
            raise ValueError("No NREM-REM cycles were found.")

        catalog_cache = {}

        occurrence_rows = []
        loop_piece_rows = []
        sym_edge_rows = []

        for _, row in df_raw_occ.iterrows():
            row_dict = row.to_dict()

            hyp_index = int(row["hyp_index"])
            subject = row.get("subject", hyp_index)
            cycle_index = int(row.get("cycle_index", -1))

            flux_key = get_flux_key_from_row_force(row, AllCycle)

            target_structure = tuple(int(x) for x in row["structure"])
            target_code = structure_to_code_force(target_structure)
            target_name = structure_to_name_force(target_structure)

            if flux_key is None:
                pieces = []
                missing_edges = tuple(
                    (target_structure[i], target_structure[i + 1])
                    for i in range(len(target_structure) - 1)
                )
                coverage_fraction = 0.0
            else:
                if flux_key not in catalog_cache:
                    flux_loops = AllCycle.get(flux_key, [])
                    flux_strengths = get_strength_list_for_hypnogram(
                        hyp_index=hyp_index,
                        subject=subject,
                        NocAllStrength=AllStrength,
                    )

                    catalog_cache[flux_key] = build_flux_loop_segment_catalog_with_strength(
                        flux_loops=flux_loops,
                        flux_strengths=flux_strengths,
                    )

                catalog = catalog_cache[flux_key]

                pieces, missing_edges, coverage_fraction = (
                    decompose_cycle_by_flux_loop_segments_force(
                        target_structure=target_structure,
                        catalog=catalog,
                        min_segment_edges=min_segment_edges,
                        allow_loop_reuse=allow_loop_reuse,
                    )
                )

            # ============================================================
            # Loop flux contribution
            # ============================================================

            if count_each_loop_once_per_cycle:
                used_loop_indices = sorted(set(int(p["loop_index"]) for p in pieces))

                loop_strength_sum = 0.0
                loop_strength_edge_weighted_sum = 0.0

                loop_codes = []
                loop_names = []
                loop_strengths_used = []
                loop_n_edges_used = []
                loop_contributions_used = []

                for loop_idx in used_loop_indices:
                    selected = [p for p in pieces if int(p["loop_index"]) == loop_idx]
                    if len(selected) == 0:
                        continue

                    p0 = selected[0]

                    strength = float(p0["loop_strength"])
                    loop_n_edges = int(p0["loop_n_edges"])

                    if multiply_loop_flux_by_n_edges:
                        contribution = strength * loop_n_edges
                    else:
                        contribution = strength

                    loop_strength_sum += contribution
                    loop_strength_edge_weighted_sum += strength * loop_n_edges

                    loop_codes.append(p0["loop_code"])
                    loop_names.append(p0["loop_name"])
                    loop_strengths_used.append(strength)
                    loop_n_edges_used.append(loop_n_edges)
                    loop_contributions_used.append(contribution)

            else:
                loop_strength_sum = 0.0
                loop_strength_edge_weighted_sum = 0.0

                loop_codes = []
                loop_names = []
                loop_strengths_used = []
                loop_n_edges_used = []
                loop_contributions_used = []

                for p in pieces:
                    strength = float(p["loop_strength"])
                    loop_n_edges = int(p["loop_n_edges"])

                    if multiply_loop_flux_by_n_edges:
                        contribution = strength * loop_n_edges
                    else:
                        contribution = strength

                    loop_strength_sum += contribution
                    loop_strength_edge_weighted_sum += strength * loop_n_edges

                    loop_codes.append(p["loop_code"])
                    loop_names.append(p["loop_name"])
                    loop_strengths_used.append(strength)
                    loop_n_edges_used.append(loop_n_edges)
                    loop_contributions_used.append(contribution)

            # ============================================================
            # Symmetric flux contribution for uncovered edges
            # ============================================================

            sym_matrix = get_sym_matrix_for_hypnogram(
                hyp_index=hyp_index,
                subject=subject,
                NocFlux_sym_list=Flux_sym_list,
            )

            if sum_unique_missing_edges:
                missing_edges_for_sum = tuple(sorted(set(missing_edges)))
            else:
                missing_edges_for_sum = tuple(missing_edges)

            sym_uncovered_edge_sum = 0.0

            sym_edge_codes = []
            sym_edge_names = []
            sym_edge_values = []

            for edge in missing_edges_for_sum:
                a, b = int(edge[0]), int(edge[1])

                if (
                    sym_matrix.ndim == 2
                    and a < sym_matrix.shape[0]
                    and b < sym_matrix.shape[1]
                ):
                    val = float(sym_matrix[a, b])
                else:
                    val = 0.0

                if np.isnan(val):
                    val = 0.0

                sym_uncovered_edge_sum += val

                edge_code = f"{a}{b}"
                edge_name = (
                    f"{STATE_NAME_MAP.get(a, str(a))}"
                    f"→{STATE_NAME_MAP.get(b, str(b))}"
                )

                sym_edge_codes.append(edge_code)
                sym_edge_names.append(edge_name)
                sym_edge_values.append(val)

                sym_edge_rows.append({
                    "hyp_index": hyp_index,
                    "subject": subject,
                    "flux_key": flux_key,
                    "cycle_index": cycle_index,

                    "structure_code": target_code,
                    "structure_name": target_name,

                    "uncovered_edge": edge,
                    "uncovered_edge_code": edge_code,
                    "uncovered_edge_name": edge_name,
                    "sym_flux_value": val,
                })

            total_force = loop_strength_sum + sym_uncovered_edge_sum

            # ============================================================
            # Piece-level details
            # ============================================================

            for p in pieces:
                strength = float(p["loop_strength"])
                loop_n_edges = int(p["loop_n_edges"])

                if multiply_loop_flux_by_n_edges:
                    contribution = strength * loop_n_edges
                else:
                    contribution = strength

                loop_piece_rows.append({
                    "hyp_index": hyp_index,
                    "subject": subject,
                    "flux_key": flux_key,
                    "cycle_index": cycle_index,

                    "structure_code": target_code,
                    "structure_name": target_name,

                    "loop_index": int(p["loop_index"]),
                    "loop_code": p["loop_code"],
                    "loop_name": p["loop_name"],
                    "loop_strength": strength,
                    "loop_n_edges": loop_n_edges,
                    "loop_strength_times_n_edges": strength * loop_n_edges,
                    "loop_contribution_used": contribution,

                    "segment_code": p["segment_code"],
                    "segment_name": p["segment_name"],
                    "used_edges": int(p["used_edges"]),
                })

            occurrence_rows.append({
                **row_dict,

                "flux_key": flux_key,
                "target_structure_code": target_code,
                "target_structure_name": target_name,

                "loop_segment_coverage_fraction": float(coverage_fraction),
                "missing_edges_code": ";".join(f"{a}{b}" for a, b in missing_edges),

                "n_loop_pieces_used": int(len(pieces)),
                "n_unique_loops_used": int(len(set(int(p["loop_index"]) for p in pieces))),

                # 这里的 loop_strength_sum 会根据 multiply_loop_flux_by_n_edges 改变
                "loop_strength_sum": float(loop_strength_sum),

                # 额外保留一个永远等于 strength × n_edges 的版本，方便检查
                "loop_strength_times_n_edges_sum": float(loop_strength_edge_weighted_sum),

                "sym_uncovered_edge_sum": float(sym_uncovered_edge_sum),
                "loop_plus_sym_force": float(total_force),

                "used_loop_codes": " + ".join(loop_codes),
                "used_loop_names": " + ".join(loop_names),
                "used_loop_strengths": " + ".join(f"{x:.8g}" for x in loop_strengths_used),
                "used_loop_n_edges": " + ".join(str(x) for x in loop_n_edges_used),
                "used_loop_contributions": " + ".join(f"{x:.8g}" for x in loop_contributions_used),

                "sym_uncovered_edge_codes": ";".join(sym_edge_codes),
                "sym_uncovered_edge_names": ";".join(sym_edge_names),
                "sym_uncovered_edge_values": ";".join(f"{x:.8g}" for x in sym_edge_values),

                "decomposition_code": " + ".join(p["segment_code"] for p in pieces),
            })

        df_occ_force = pd.DataFrame(occurrence_rows)
        df_loop_piece_force = pd.DataFrame(loop_piece_rows)
        df_sym_edge_force = pd.DataFrame(sym_edge_rows)

        numeric_cols = [
            "loop_segment_coverage_fraction",
            "n_loop_pieces_used",
            "n_unique_loops_used",
            "loop_strength_sum",
            "loop_strength_times_n_edges_sum",
            "sym_uncovered_edge_sum",
            "loop_plus_sym_force",
        ]

        for col in numeric_cols:
            if col in df_occ_force.columns:
                df_occ_force[col] = df_occ_force[col].fillna(0.0)

        # ============================================================
        # Hypnogram-level summary
        # ============================================================

        hyp_rows = []

        for hyp_index, subject, _ in iter_hypnograms_force(data):
            df_i = df_occ_force[
                df_occ_force["hyp_index"].astype(int) == int(hyp_index)
            ].copy()

            if len(df_i) == 0:
                hyp_rows.append({
                    "hyp_index": int(hyp_index),
                    "subject": subject,
                    "n_nremrem_cycles": 0,

                    "mean_loop_strength_sum": 0.0,
                    "mean_loop_strength_times_n_edges_sum": 0.0,
                    "mean_sym_uncovered_edge_sum": 0.0,
                    "mean_loop_plus_sym_force": 0.0,

                    "sum_loop_strength_sum": 0.0,
                    "sum_loop_strength_times_n_edges_sum": 0.0,
                    "sum_sym_uncovered_edge_sum": 0.0,
                    "sum_loop_plus_sym_force": 0.0,

                    "mean_loop_segment_coverage_fraction": 0.0,
                    "fully_covered_cycle_fraction": 0.0,
                })
                continue

            hyp_rows.append({
                "hyp_index": int(hyp_index),
                "subject": subject,
                "n_nremrem_cycles": int(len(df_i)),

                "mean_loop_strength_sum": float(df_i["loop_strength_sum"].mean()),
                "mean_loop_strength_times_n_edges_sum": float(
                    df_i["loop_strength_times_n_edges_sum"].mean()
                ),
                "mean_sym_uncovered_edge_sum": float(df_i["sym_uncovered_edge_sum"].mean()),
                "mean_loop_plus_sym_force": float(df_i["loop_plus_sym_force"].mean()),

                "sum_loop_strength_sum": float(df_i["loop_strength_sum"].sum()),
                "sum_loop_strength_times_n_edges_sum": float(
                    df_i["loop_strength_times_n_edges_sum"].sum()
                ),
                "sum_sym_uncovered_edge_sum": float(df_i["sym_uncovered_edge_sum"].sum()),
                "sum_loop_plus_sym_force": float(df_i["loop_plus_sym_force"].sum()),

                "mean_loop_segment_coverage_fraction": float(
                    df_i["loop_segment_coverage_fraction"].mean()
                ),
                "fully_covered_cycle_fraction": float(
                    (df_i["loop_segment_coverage_fraction"] == 1.0).mean()
                ),
            })

        df_hyp_force = pd.DataFrame(hyp_rows)

        for col in df_hyp_force.columns:
            if pd.api.types.is_numeric_dtype(df_hyp_force[col]):
                df_hyp_force[col] = df_hyp_force[col].fillna(0.0)

        force_mean_by_hypnogram = df_hyp_force["mean_loop_plus_sym_force"].to_numpy()
        force_sum_by_hypnogram = df_hyp_force["sum_loop_plus_sym_force"].to_numpy()

        loop_flux_mean_by_hypnogram = df_hyp_force["mean_loop_strength_sum"].to_numpy()
        loop_flux_sum_by_hypnogram = df_hyp_force["sum_loop_strength_sum"].to_numpy()

        loop_flux_times_n_edges_mean_by_hypnogram = (
            df_hyp_force["mean_loop_strength_times_n_edges_sum"].to_numpy()
        )
        loop_flux_times_n_edges_sum_by_hypnogram = (
            df_hyp_force["sum_loop_strength_times_n_edges_sum"].to_numpy()
        )

        sym_flux_mean_by_hypnogram = df_hyp_force["mean_sym_uncovered_edge_sum"].to_numpy()
        sym_flux_sum_by_hypnogram = df_hyp_force["sum_sym_uncovered_edge_sum"].to_numpy()

        assert np.isnan(force_mean_by_hypnogram).sum() == 0
        assert np.isnan(force_sum_by_hypnogram).sum() == 0

        return {
            "df_raw_nremrem_occurrence": df_raw_occ,
            "df_occurrence_force": df_occ_force,
            "df_loop_piece_force": df_loop_piece_force,
            "df_sym_edge_force": df_sym_edge_force,
            "df_hypnogram_force": df_hyp_force,

            # loop + sym
            "force_mean_by_hypnogram": force_mean_by_hypnogram,
            "force_sum_by_hypnogram": force_sum_by_hypnogram,

            # loop only, according to multiply_loop_flux_by_n_edges setting
            "loop_flux_mean_by_hypnogram": loop_flux_mean_by_hypnogram,
            "loop_flux_sum_by_hypnogram": loop_flux_sum_by_hypnogram,

            # loop only, always strength × loop_n_edges
            "loop_flux_times_n_edges_mean_by_hypnogram": (
                loop_flux_times_n_edges_mean_by_hypnogram
            ),
            "loop_flux_times_n_edges_sum_by_hypnogram": (
                loop_flux_times_n_edges_sum_by_hypnogram
            ),

            # symmetric only
            "sym_flux_mean_by_hypnogram": sym_flux_mean_by_hypnogram,
            "sym_flux_sum_by_hypnogram": sym_flux_sum_by_hypnogram,
        }

    return {'analyze_nremrem_loop_flux_plus_sym_force_by_hypnogram': analyze_nremrem_loop_flux_plus_sym_force_by_hypnogram}

_drug_force_helpers = _build_drug_force_helpers()
analyze_nremrem_loop_flux_plus_sym_force_by_hypnogram = _drug_force_helpers['analyze_nremrem_loop_flux_plus_sym_force_by_hypnogram']


def _build_drug_scan_helpers():
    # Telemetry original cell 57, line 1.
    import os

    # Telemetry original cell 57, line 2.
    import warnings

    # Telemetry original cell 57, line 5.
    import matplotlib.pyplot as plt

    # Telemetry original cell 57, line 7.
    from scipy.stats import wilcoxon

    # Telemetry original cell 59, line 1.
    import numpy as np

    # Telemetry original cell 59, line 2.
    import pandas as pd

    # Telemetry original cell 59, line 6.
    STATE_NAME_MAP = {
        0: "Wake",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM",
    }

    # Telemetry original cell 57, line 13.
    def _iter_hypnograms_drug_scan(data):
        """
        Returns:
            [(hyp_index, subject, hypnogram), ...]

        For dict:
            insertion order is preserved.

        For list/tuple:
            subject = hyp_index.
        """

        if isinstance(data, dict):
            out = []
            for i, (subject, hyp) in enumerate(data.items()):
                h = np.asarray(hyp).squeeze()
                out.append((int(i), subject, h))
            return out

        arr = np.asarray(data, dtype=object)

        # single hypnogram
        if (
            arr.ndim == 1
            and len(arr) > 0
            and all(np.isscalar(x) for x in arr)
        ):
            return [(0, 0, np.asarray(data).squeeze())]

        # list of hypnograms
        out = []
        for i, hyp in enumerate(data):
            out.append(
                (int(i), int(i), np.asarray(hyp).squeeze())
            )

        return out

    # Telemetry original cell 57, line 55.
    def _compute_tst_hours_drug_scan(
        data,
        epoch_sec=30,
        rem_state=4,
        nrem_states=(1, 2, 3),
    ):
        """
        TST = N1 + N2 + N3 + REM
        """

        sleep_states = set(nrem_states) | {rem_state}

        items = _iter_hypnograms_drug_scan(data)

        tst = np.full(len(items), np.nan, dtype=float)

        for hyp_index, subject, hyp in items:

            h = np.asarray(hyp).squeeze()

            if h.ndim != 1:
                raise ValueError(
                    f"Hypnogram {hyp_index} is not 1D: shape={h.shape}"
                )

            n_sleep_epochs = np.sum(
                np.isin(h, list(sleep_states))
            )

            tst[hyp_index] = (
                n_sleep_epochs
                * float(epoch_sec)
                / 3600.0
            )

        return tst

    # Telemetry original cell 57, line 96.
    def _get_cycle_duration_min_drug_scan(
        df_intervals,
        epoch_sec=30,
    ):
        if df_intervals is None or len(df_intervals) == 0:
            return np.array([], dtype=float)

        # Prefer already-computed duration
        for col in [
            "duration_min",
            "cycle_duration_min",
            "interval_duration_min",
        ]:
            if col in df_intervals.columns:
                return pd.to_numeric(
                    df_intervals[col],
                    errors="coerce",
                ).to_numpy(dtype=float)

        # Infer from epoch columns
        epoch_pairs = [
            ("cycle_start_epoch", "cycle_end_epoch"),
            ("start_epoch", "end_epoch"),
            ("interval_start_epoch", "interval_end_epoch"),
        ]

        for start_col, end_col in epoch_pairs:

            if (
                start_col in df_intervals.columns
                and end_col in df_intervals.columns
            ):
                st = pd.to_numeric(
                    df_intervals[start_col],
                    errors="coerce",
                ).to_numpy(dtype=float)

                ed = pd.to_numeric(
                    df_intervals[end_col],
                    errors="coerce",
                ).to_numpy(dtype=float)

                # inclusive interval
                return (
                    (ed - st + 1)
                    * float(epoch_sec)
                    / 60.0
                )

        raise KeyError(
            "Cannot determine NREM-REM cycle duration."
        )

    # Telemetry original cell 57, line 153.
    def _force_vector_from_df(
        df_force,
        values,
        n_hypnograms,
    ):
        """
        Align force values to hyp_index.

        values must correspond row-by-row to df_force.
        """

        values = np.asarray(values, dtype=float)

        if len(values) != len(df_force):
            raise ValueError(
                "values and df_force have different lengths."
            )

        out = np.full(
            n_hypnograms,
            np.nan,
            dtype=float,
        )

        if "hyp_index" in df_force.columns:

            inds = pd.to_numeric(
                df_force["hyp_index"],
                errors="coerce",
            ).to_numpy()

            for row_i, hyp_i in enumerate(inds):

                if not np.isfinite(hyp_i):
                    continue

                hyp_i = int(hyp_i)

                if 0 <= hyp_i < n_hypnograms:
                    out[hyp_i] = values[row_i]

        else:
            if len(values) != n_hypnograms:
                raise ValueError(
                    "df_force has no hyp_index and its length "
                    "does not match number of hypnograms."
                )

            out[:] = values

        return out

    # Telemetry original cell 57, line 210.
    def _compute_condition_metrics_at_k(
        data,
        AllCycle,
        AllStrength,
        Flux_sym_list,
        temporal_cycle_occurrence_df,

        k,
        analyze_func,

        epoch_sec=30,
        wake_state=0,
        rem_state=4,
        nrem_states=(1, 2, 3),

        include_post_rem_state=True,
        include_wake_in_structure=True,

        min_segment_edges=1,
        allow_loop_reuse=True,
        count_each_loop_once_per_cycle=True,
        sum_unique_missing_edges=True,
        multiply_loop_flux_by_n_edges=True,
    ):
        items = _iter_hypnograms_drug_scan(data)

        n_hyp = len(items)

        subjects = [
            x[1] for x in items
        ]

        # --------------------------------------------------------
        # k-dependent definition
        # --------------------------------------------------------
        min_nrem_min = 3.0 * float(k)
        min_rem_min = 1.0 * float(k)

        # --------------------------------------------------------
        # TST
        # --------------------------------------------------------
        tst_hours = _compute_tst_hours_drug_scan(
            data=data,
            epoch_sec=epoch_sec,
            rem_state=rem_state,
            nrem_states=nrem_states,
        )

        # --------------------------------------------------------
        # Extract NREM-REM cycles at this k
        # --------------------------------------------------------
        df_intervals = (
            extract_nremrem_intervals_all_hypnograms(
                data=data,

                epoch_sec=epoch_sec,
                wake_state=wake_state,
                rem_state=rem_state,
                nrem_states=nrem_states,

                min_nrem_min=min_nrem_min,
                min_rem_min=min_rem_min,

                include_post_rem_state=include_post_rem_state,
                include_wake_in_structure=include_wake_in_structure,
            )
        )

        # --------------------------------------------------------
        # frequency:
        #     0 cycles -> frequency = 0
        #
        # duration:
        #     0 cycles -> NaN
        # --------------------------------------------------------
        cycle_count = np.zeros(
            n_hyp,
            dtype=float,
        )

        mean_duration = np.full(
            n_hyp,
            np.nan,
            dtype=float,
        )

        if (
            df_intervals is not None
            and len(df_intervals) > 0
        ):
            dfi = df_intervals.copy().reset_index(drop=True)

            # Determine hyp_index
            if "hyp_index" in dfi.columns:

                dfi["_hyp_index_scan"] = pd.to_numeric(
                    dfi["hyp_index"],
                    errors="coerce",
                )

            elif "subject" in dfi.columns:

                subject_to_index = {
                    subject: i
                    for i, subject in enumerate(subjects)
                }

                dfi["_hyp_index_scan"] = (
                    dfi["subject"].map(
                        subject_to_index
                    )
                )

            else:
                raise KeyError(
                    "Cycle dataframe must contain "
                    "'hyp_index' or 'subject'."
                )

            dfi = dfi[
                dfi["_hyp_index_scan"].notna()
            ].copy()

            dfi["_hyp_index_scan"] = (
                dfi["_hyp_index_scan"].astype(int)
            )

            # cycle duration
            dfi["_duration_min_scan"] = (
                _get_cycle_duration_min_drug_scan(
                    dfi,
                    epoch_sec=epoch_sec,
                )
            )

            # count
            g_count = (
                dfi
                .groupby("_hyp_index_scan")
                .size()
            )

            for i, n in g_count.items():
                if 0 <= int(i) < n_hyp:
                    cycle_count[int(i)] = float(n)

            # mean duration
            g_duration = (
                dfi
                .groupby("_hyp_index_scan")[
                    "_duration_min_scan"
                ]
                .mean()
            )

            for i, x in g_duration.items():

                if (
                    0 <= int(i) < n_hyp
                    and np.isfinite(x)
                ):
                    mean_duration[int(i)] = float(x)

        # --------------------------------------------------------
        # frequency = count / TST
        # --------------------------------------------------------
        frequency = np.full(
            n_hyp,
            np.nan,
            dtype=float,
        )

        valid_tst = (
            np.isfinite(tst_hours)
            & (tst_hours > 0)
        )

        frequency[valid_tst] = (
            cycle_count[valid_tst]
            / tst_hours[valid_tst]
        )

        # --------------------------------------------------------
        # Forces at same k
        # --------------------------------------------------------
        out_force = analyze_func(
            data=data,

            AllCycle=AllCycle,
            AllStrength=AllStrength,
            Flux_sym_list=Flux_sym_list,

            temporal_cycle_occurrence_df=(
                temporal_cycle_occurrence_df
            ),

            epoch_sec=epoch_sec,
            wake_state=wake_state,
            rem_state=rem_state,
            nrem_states=nrem_states,

            min_nrem_min=min_nrem_min,
            min_rem_min=min_rem_min,

            include_post_rem_state=include_post_rem_state,
            include_wake_in_structure=include_wake_in_structure,

            min_segment_edges=min_segment_edges,
            allow_loop_reuse=allow_loop_reuse,

            count_each_loop_once_per_cycle=(
                count_each_loop_once_per_cycle
            ),

            sum_unique_missing_edges=(
                sum_unique_missing_edges
            ),

            multiply_loop_flux_by_n_edges=(
                multiply_loop_flux_by_n_edges
            ),
        )

        df_force = (
            out_force["df_hypnogram_force"]
            .copy()
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # EQ force
        # --------------------------------------------------------
        force_eq = _force_vector_from_df(
            df_force=df_force,

            values=df_force[
                "sum_sym_uncovered_edge_sum"
            ].to_numpy(dtype=float),

            n_hypnograms=n_hyp,
        )

        # --------------------------------------------------------
        # LF force
        # --------------------------------------------------------
        force_lf = _force_vector_from_df(
            df_force=df_force,

            values=df_force[
                "sum_loop_strength_sum"
            ].to_numpy(dtype=float),

            n_hypnograms=n_hyp,
        )

        # --------------------------------------------------------
        # Total force
        # --------------------------------------------------------
        total_values = np.asarray(
            out_force["force_sum_by_hypnogram"],
            dtype=float,
        )

        force_total = _force_vector_from_df(
            df_force=df_force,
            values=total_values,
            n_hypnograms=n_hyp,
        )

        return {
            "subjects": subjects,

            "TST_hours": tst_hours,

            "cycle_count": cycle_count,

            "frequency": frequency,

            "mean_duration": mean_duration,

            "force_eq": force_eq,
            "force_lf": force_lf,
            "force_total": force_total,

            "df_intervals": df_intervals,
            "out_force": out_force,
        }

    # Telemetry original cell 57, line 502.
    def _paired_wilcoxon_drug(
        tema,
        placebo,
        min_n=3,
    ):
        tema = np.asarray(
            tema,
            dtype=float,
        )

        placebo = np.asarray(
            placebo,
            dtype=float,
        )

        if len(tema) != len(placebo):
            raise ValueError(
                "Temazepam and Placebo arrays "
                "must have same length."
            )

        mask = (
            np.isfinite(tema)
            & np.isfinite(placebo)
        )

        x = tema[mask]
        y = placebo[mask]

        n = len(x)

        if n < min_n:
            return {
                "n_pairs": n,
                "statistic": np.nan,
                "p": np.nan,

                "tema_mean": (
                    np.nanmean(x)
                    if n > 0 else np.nan
                ),

                "placebo_mean": (
                    np.nanmean(y)
                    if n > 0 else np.nan
                ),

                "tema_median": (
                    np.nanmedian(x)
                    if n > 0 else np.nan
                ),

                "placebo_median": (
                    np.nanmedian(y)
                    if n > 0 else np.nan
                ),

                "mean_diff": (
                    np.nanmean(x - y)
                    if n > 0 else np.nan
                ),

                "median_diff": (
                    np.nanmedian(x - y)
                    if n > 0 else np.nan
                ),
            }

        diff = x - y

        # exactly identical
        if np.allclose(diff, 0):
            stat = 0.0
            p = 1.0

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                try:
                    result = wilcoxon(
                        x,
                        y,
                        alternative="two-sided",
                        zero_method="wilcox",
                    )

                    stat = float(
                        result.statistic
                    )

                    p = float(
                        result.pvalue
                    )

                except Exception:
                    stat = np.nan
                    p = np.nan

        return {
            "n_pairs": int(n),

            "statistic": stat,
            "p": p,

            "tema_mean": float(
                np.mean(x)
            ),

            "placebo_mean": float(
                np.mean(y)
            ),

            "tema_median": float(
                np.median(x)
            ),

            "placebo_median": float(
                np.median(y)
            ),

            "mean_diff": float(
                np.mean(diff)
            ),

            "median_diff": float(
                np.median(diff)
            ),
        }

    # Telemetry original cell 57, line 636.
    def scan_k_temazepam_placebo_differences(
        tema_data,
        placebo_data,

        tema_AllCycle,
        placebo_AllCycle,

        tema_AllStrength,
        placebo_AllStrength,

        tema_Flux_sym_list,
        placebo_Flux_sym_list,

        tema_temporal_cycle_occurrence_df,
        placebo_temporal_cycle_occurrence_df,

        k_values,

        analyze_func,

        epoch_sec=30,
        wake_state=0,
        rem_state=4,
        nrem_states=(1, 2, 3),

        include_post_rem_state=True,
        include_wake_in_structure=True,

        min_segment_edges=1,
        allow_loop_reuse=True,
        count_each_loop_once_per_cycle=True,
        sum_unique_missing_edges=True,
        multiply_loop_flux_by_n_edges=True,

        min_n_pairs=3,

        figsize=(10, 7),

        save_path=None,

        verbose=True,
    ):
        """
        Scan k and compare Temazepam vs Placebo for:

            1. NREM-REM cycle frequency
            2. mean NREM-REM cycle duration
            3. DF_EQ_Uncovered
            4. DF_LF_Selected
            5. DF_Total

        Statistical test:
            two-sided paired Wilcoxon signed-rank test.

        IMPORTANT:
            Temazepam and Placebo data must be in the SAME
            participant order.

        Direction shown in output:
            Tema > Placebo
            Tema < Placebo
        """

        k_values = np.asarray(
            k_values,
            dtype=float,
        )

        n_tema = len(
            _iter_hypnograms_drug_scan(
                tema_data
            )
        )

        n_placebo = len(
            _iter_hypnograms_drug_scan(
                placebo_data
            )
        )

        if n_tema != n_placebo:
            raise ValueError(
                f"Temazepam has {n_tema} hypnograms, "
                f"Placebo has {n_placebo}. "
                "Paired analysis requires the same number."
            )

        # --------------------------------------------------------
        # Metric names
        # --------------------------------------------------------
        metric_info = {
            "frequency": {
                "label": "NREM-REM frequency",
            },

            "mean_duration": {
                "label": "NREM-REM mean duration",
            },

            "force_eq": {
                "label": r"$\mathrm{DF}^{Uncovered}_{EQ}$",
            },

            "force_lf": {
                "label": r"$\mathrm{DF}^{Selected}_{LF}$",
            },

            "force_total": {
                "label": r"$\mathrm{DF}_{Total}$",
            },
        }

        stats_rows = []
        subject_rows = []

        raw_by_k = {}

        # ========================================================
        # Scan k
        # ========================================================
        for k in k_values:

            if verbose:
                print(
                    "\n"
                    + "=" * 85
                )

                print(
                    f"k = {k:.2f} "
                    f"(NREM >= {3*k:.2f} min, "
                    f"REM >= {k:.2f} min)"
                )

            # ----------------------------------------------------
            # Temazepam
            # ----------------------------------------------------
            tema = _compute_condition_metrics_at_k(
                data=tema_data,

                AllCycle=tema_AllCycle,
                AllStrength=tema_AllStrength,
                Flux_sym_list=tema_Flux_sym_list,

                temporal_cycle_occurrence_df=(
                    tema_temporal_cycle_occurrence_df
                ),

                k=k,

                analyze_func=analyze_func,

                epoch_sec=epoch_sec,
                wake_state=wake_state,
                rem_state=rem_state,
                nrem_states=nrem_states,

                include_post_rem_state=(
                    include_post_rem_state
                ),

                include_wake_in_structure=(
                    include_wake_in_structure
                ),

                min_segment_edges=min_segment_edges,
                allow_loop_reuse=allow_loop_reuse,

                count_each_loop_once_per_cycle=(
                    count_each_loop_once_per_cycle
                ),

                sum_unique_missing_edges=(
                    sum_unique_missing_edges
                ),

                multiply_loop_flux_by_n_edges=(
                    multiply_loop_flux_by_n_edges
                ),
            )

            # ----------------------------------------------------
            # Placebo
            # ----------------------------------------------------
            placebo = _compute_condition_metrics_at_k(
                data=placebo_data,

                AllCycle=placebo_AllCycle,
                AllStrength=placebo_AllStrength,
                Flux_sym_list=placebo_Flux_sym_list,

                temporal_cycle_occurrence_df=(
                    placebo_temporal_cycle_occurrence_df
                ),

                k=k,

                analyze_func=analyze_func,

                epoch_sec=epoch_sec,
                wake_state=wake_state,
                rem_state=rem_state,
                nrem_states=nrem_states,

                include_post_rem_state=(
                    include_post_rem_state
                ),

                include_wake_in_structure=(
                    include_wake_in_structure
                ),

                min_segment_edges=min_segment_edges,
                allow_loop_reuse=allow_loop_reuse,

                count_each_loop_once_per_cycle=(
                    count_each_loop_once_per_cycle
                ),

                sum_unique_missing_edges=(
                    sum_unique_missing_edges
                ),

                multiply_loop_flux_by_n_edges=(
                    multiply_loop_flux_by_n_edges
                ),
            )

            raw_by_k[float(k)] = {
                "Temazepam": tema,
                "Placebo": placebo,
            }

            # ----------------------------------------------------
            # Save subject-level values
            # ----------------------------------------------------
            for i in range(n_tema):

                subject_rows.append({
                    "k": float(k),
                    "hyp_index": int(i),

                    "tema_frequency":
                        tema["frequency"][i],

                    "placebo_frequency":
                        placebo["frequency"][i],

                    "tema_mean_duration":
                        tema["mean_duration"][i],

                    "placebo_mean_duration":
                        placebo["mean_duration"][i],

                    "tema_force_eq":
                        tema["force_eq"][i],

                    "placebo_force_eq":
                        placebo["force_eq"][i],

                    "tema_force_lf":
                        tema["force_lf"][i],

                    "placebo_force_lf":
                        placebo["force_lf"][i],

                    "tema_force_total":
                        tema["force_total"][i],

                    "placebo_force_total":
                        placebo["force_total"][i],

                    "tema_cycle_count":
                        tema["cycle_count"][i],

                    "placebo_cycle_count":
                        placebo["cycle_count"][i],
                })

            # ----------------------------------------------------
            # 5 paired tests
            # ----------------------------------------------------
            for metric_key, info in metric_info.items():

                result = _paired_wilcoxon_drug(
                    tema=tema[metric_key],
                    placebo=placebo[metric_key],
                    min_n=min_n_pairs,
                )

                mean_diff = result[
                    "mean_diff"
                ]

                if not np.isfinite(mean_diff):
                    direction = "NA"

                elif mean_diff > 0:
                    direction = (
                        "Temazepam > Placebo"
                    )

                elif mean_diff < 0:
                    direction = (
                        "Temazepam < Placebo"
                    )

                else:
                    direction = (
                        "Temazepam = Placebo"
                    )

                stats_rows.append({
                    "k": float(k),

                    "min_nrem_min": float(
                        3 * k
                    ),

                    "min_rem_min": float(k),

                    "metric": metric_key,

                    "metric_label": info[
                        "label"
                    ],

                    "n_pairs": result[
                        "n_pairs"
                    ],

                    "wilcoxon_statistic":
                        result["statistic"],

                    "p_value":
                        result["p"],

                    "tema_mean":
                        result["tema_mean"],

                    "placebo_mean":
                        result[
                            "placebo_mean"
                        ],

                    "tema_median":
                        result[
                            "tema_median"
                        ],

                    "placebo_median":
                        result[
                            "placebo_median"
                        ],

                    "mean_diff_T_minus_P":
                        result[
                            "mean_diff"
                        ],

                    "median_diff_T_minus_P":
                        result[
                            "median_diff"
                        ],

                    "direction":
                        direction,
                })

                if verbose:

                    print(
                        f"{info['label']:30s}  "
                        f"Tema={result['tema_mean']:.5g}, "
                        f"Placebo={result['placebo_mean']:.5g}, "
                        f"p={result['p']:.3g}, "
                        f"N={result['n_pairs']}, "
                        f"{direction}"
                    )

        # ========================================================
        # DataFrames
        # ========================================================
        df_stats = pd.DataFrame(
            stats_rows
        )

        df_subject_values = pd.DataFrame(
            subject_rows
        )

        # --------------------------------------------------------
        # significance flag
        # --------------------------------------------------------
        df_stats["significant_0p05"] = (
            df_stats["p_value"] < 0.05
        )

        # --------------------------------------------------------
        # Convenient wide p-value table
        # --------------------------------------------------------
        df_p_wide = (
            df_stats
            .pivot(
                index="k",
                columns="metric",
                values="p_value",
            )
            .reset_index()
        )

        # ========================================================
        # Plot
        # ========================================================
        fig, ax = plt.subplots(
            figsize=figsize
        )

        # Plot each of five p-value curves
        for metric_key, info in metric_info.items():

            sub = (
                df_stats[
                    df_stats["metric"]
                    == metric_key
                ]
                .sort_values("k")
                .copy()
            )

            k_arr = sub[
                "k"
            ].to_numpy(dtype=float)

            p_arr = sub[
                "p_value"
            ].to_numpy(dtype=float)

            # Avoid exactly zero on log scale
            p_plot = p_arr.copy()

            finite = np.isfinite(
                p_plot
            )

            p_plot[
                finite & (p_plot <= 0)
            ] = np.finfo(float).tiny

            # line first; let matplotlib choose color
            line, = ax.plot(
                k_arr,
                p_plot,
                linewidth=2,
                label=info["label"],
            )

            line_color = (
                line.get_color()
            )

            mean_diff = sub[
                "mean_diff_T_minus_P"
            ].to_numpy(dtype=float)

            tema_larger = (
                np.isfinite(p_plot)
                & np.isfinite(mean_diff)
                & (mean_diff > 0)
            )

            placebo_larger = (
                np.isfinite(p_plot)
                & np.isfinite(mean_diff)
                & (mean_diff < 0)
            )

            equal_mask = (
                np.isfinite(p_plot)
                & np.isfinite(mean_diff)
                & np.isclose(mean_diff, 0)
            )

            # Filled:
            # Temazepam > Placebo
            ax.scatter(
                k_arr[tema_larger],
                p_plot[tema_larger],
                marker="o",
                s=42,
                facecolors=line_color,
                edgecolors=line_color,
                zorder=3,
            )

            # Open:
            # Temazepam < Placebo
            ax.scatter(
                k_arr[placebo_larger],
                p_plot[placebo_larger],
                marker="o",
                s=42,
                facecolors="none",
                edgecolors=line_color,
                linewidths=1.5,
                zorder=3,
            )

            # x:
            # means identical
            if np.any(equal_mask):
                ax.scatter(
                    k_arr[equal_mask],
                    p_plot[equal_mask],
                    marker="x",
                    s=42,
                    zorder=3,
                )

        # --------------------------------------------------------
        # p threshold
        # --------------------------------------------------------
        ax.axhline(
            0.05,
            linestyle="--",
            linewidth=1.3,
            label="p = 0.05",
        )

        ax.set_yscale(
            "log"
        )

        ax.set_xlabel(
            "k"
        )

        ax.set_ylabel(
            "Paired Wilcoxon p value"
        )

        ax.set_title(
            "Temazepam vs Placebo\n"
            "Filled marker: Temazepam > Placebo; "
            "open marker: Temazepam < Placebo"
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend(
            fontsize=9,
            frameon=True,
        )

        plt.tight_layout()

        # --------------------------------------------------------
        # Save
        # --------------------------------------------------------
        if save_path is not None:

            save_dir = os.path.dirname(
                save_path
            )

            if save_dir != "":
                os.makedirs(
                    save_dir,
                    exist_ok=True,
                )

            fig.savefig(
                save_path,
                bbox_inches="tight",
            )

        return {
            "df_stats": df_stats,

            "df_p_wide": df_p_wide,

            "df_subject_values":
                df_subject_values,

            "raw_by_k":
                raw_by_k,

            "fig": fig,
            "ax": ax,
        }

    # Telemetry original cell 59, line 19.
    def to_1d_hypnogram(x):
        h0 = np.asarray(x)
        h = np.squeeze(h0)

        if h.ndim == 1:
            return h.astype(int)

        if h.ndim == 2 and 1 in h.shape:
            return h.reshape(-1).astype(int)

        raise ValueError(
            f"Invalid hypnogram shape: original={h0.shape}, squeezed={h.shape}"
        )

    # Telemetry original cell 59, line 34.
    def iter_hypnograms(data):
        if isinstance(data, dict):
            items = sorted(data.items(), key=lambda kv: kv[0])
            return [(i, k, v) for i, (k, v) in enumerate(items)]

        arr = np.asarray(data)

        if arr.ndim == 1:
            return [(0, 0, data)]

        if arr.ndim == 2 and arr.dtype != object:
            return [(i, i, arr[i]) for i in range(arr.shape[0])]

        return [(i, i, hyp) for i, hyp in enumerate(data)]

    # Telemetry original cell 59, line 50.
    def find_runs(x):
        x = np.asarray(x)

        if len(x) == 0:
            return (
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )

        change = np.where(np.diff(x) != 0)[0] + 1
        starts = np.r_[0, change]
        ends = np.r_[change - 1, len(x) - 1]
        states = x[starts]

        return states.astype(int), starts.astype(int), ends.astype(int)

    # Telemetry original cell 59, line 68.
    def compress_consecutive(seq):
        seq = list(seq)

        if len(seq) == 0:
            return []

        out = [int(seq[0])]

        for x in seq[1:]:
            x = int(x)
            if x != out[-1]:
                out.append(x)

        return out

    # Telemetry original cell 59, line 84.
    def structure_to_code(seq):
        if seq is None:
            return ""

        if isinstance(seq, str):
            return (
                seq.replace(" ", "")
                   .replace(",", "")
                   .replace("-", "")
                   .replace("_", "")
                   .replace("→", "")
            )

        return "".join(str(int(x)) for x in seq)

    # Telemetry original cell 59, line 111.
    def structure_to_name(seq, state_name_map=None, arrow="→"):
        if state_name_map is None:
            state_name_map = STATE_NAME_MAP

        return arrow.join(state_name_map.get(int(x), str(x)) for x in seq)

    # Telemetry original cell 59, line 183.
    def extract_nremrem_intervals_all_hypnograms(
        data,
        epoch_sec=30,
        wake_state=0,
        rem_state=4,
        nrem_states=(1, 2, 3),
        min_nrem_min=0,
        min_rem_min=0,
        include_post_rem_state=True,
        include_wake_in_structure=False,
    ):
        """
        Extract raw NREM-REM cycle intervals.

        For inside/outside classification:
            if inside_uses_structure_end=True later,
            inside interval = cycle_start_epoch -> structure_end_epoch.

        This lets post-REM return state close loops such as N1→N2→REM→N1.
        """

        rows = []

        min_nrem_epochs = int(np.ceil(min_nrem_min * 60 / epoch_sec))
        min_rem_epochs = int(np.ceil(min_rem_min * 60 / epoch_sec))

        for hyp_index, subject, hyp in iter_hypnograms(data):
            h = to_1d_hypnogram(hyp)

            sleep_states = set(nrem_states) | {rem_state}
            sleep_onset_candidates = np.where(np.isin(h, list(sleep_states)))[0]

            if len(sleep_onset_candidates) == 0:
                continue

            sleep_onset = int(sleep_onset_candidates[0])

            run_states, run_starts, run_ends = find_runs(h)

            rem_runs = []
            for s, st, ed in zip(run_states, run_starts, run_ends):
                if int(s) == rem_state:
                    rem_runs.append((int(st), int(ed)))

            cycle_start = sleep_onset
            cycle_index = 0

            for rem_start, rem_end in rem_runs:
                if rem_end < cycle_start:
                    continue

                pre_rem_segment = h[cycle_start:rem_start]

                nrem_count = int(np.sum(np.isin(pre_rem_segment, nrem_states)))
                rem_count = int(rem_end - rem_start + 1)

                if nrem_count >= min_nrem_epochs and rem_count >= min_rem_epochs:
                    segment_start = int(cycle_start)
                    segment_end = int(rem_end)

                    if include_post_rem_state and rem_end + 1 < len(h):
                        structure_end = int(rem_end + 1)
                    else:
                        structure_end = int(rem_end)

                    segment = h[segment_start:structure_end + 1]

                    structure_with_wake = tuple(compress_consecutive(segment))

                    if include_wake_in_structure:
                        structure = structure_with_wake
                    else:
                        segment_no_wake = [
                            x for x in segment if int(x) != wake_state
                        ]
                        structure = tuple(compress_consecutive(segment_no_wake))

                    duration_epochs = int(segment_end - segment_start + 1)
                    duration_min = duration_epochs * epoch_sec / 60.0

                    rows.append({
                        "hyp_index": int(hyp_index),
                        "subject": subject,
                        "cycle_index": int(cycle_index),

                        "cycle_start_epoch": int(segment_start),
                        "cycle_end_epoch": int(segment_end),
                        "structure_end_epoch": int(structure_end),

                        "rem_start_epoch": int(rem_start),
                        "rem_end_epoch": int(rem_end),

                        "nrem_epochs_before_rem": int(nrem_count),
                        "rem_epochs": int(rem_count),

                        "duration_epochs": int(duration_epochs),
                        "duration_min": float(duration_min),

                        "structure": structure,
                        "structure_code": structure_to_code(structure),
                        "structure_name": structure_to_name(structure),

                        "structure_with_wake": structure_with_wake,
                        "structure_with_wake_code": structure_to_code(structure_with_wake),
                        "structure_with_wake_name": structure_to_name(structure_with_wake),
                    })

                    cycle_index += 1
                    cycle_start = rem_end + 1

        return pd.DataFrame(rows)

    return {'scan_k_temazepam_placebo_differences': scan_k_temazepam_placebo_differences}

_drug_scan_helpers = _build_drug_scan_helpers()
scan_k_temazepam_placebo_differences = _drug_scan_helpers['scan_k_temazepam_placebo_differences']


# Final main Fig5 definitions, original Telemetry cell 31.
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats


def _fig5_vector(values, name):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1:
        raise ValueError(f'{name} must be a one-dimensional paired vector.')
    if np.any(np.isfinite(a) & (a < 0)):
        raise ValueError(f'{name} contains negative values.')
    return a.copy()


def fig5_matrix_summary(matrices, name, kind):
    """Compute manuscript Wake EQ from non-Wake entries in column 0; audit original full-column sum."""
    if kind not in ('symmetric_to_wake',):
        raise ValueError('Unknown matrix summary kind.')
    if isinstance(matrices, dict):
        raise TypeError(f'{name}: supply matrices in the aligned subject order, not an unordered mapping.')
    values, rows = [], []
    for i, item in enumerate(matrices):
        a = np.asarray(item, dtype=float)
        if a.shape != (5, 5):
            raise ValueError(f'{name}[{i}] has shape {a.shape}; expected (5, 5).')
        if np.any(np.isfinite(a) & (a < 0)):
            raise ValueError(f'{name}[{i}] has negative flow.')
        # Manuscript: sum(i != Wake) Qsym[i,Wake]. Original full column agrees
        # whenever Qsym[Wake,Wake] == 0. Audit the difference explicitly.
        value = float(a[1:,0].sum())
        values.append(value)
        rows.append(dict(matrix_collection=name, paired_index=i, summary_kind=kind,
                         value=value, original_full_column_sum=float(a[:,0].sum()),
                         excluded_wake_self_flow=float(a[0,0]), diagonal_sum=float(np.trace(a)),
                         all_entries_finite=bool(np.isfinite(a).all())))
    table = pd.DataFrame(rows, columns=['matrix_collection', 'paired_index', 'summary_kind',
                                     'value','original_full_column_sum','excluded_wake_self_flow', 'diagonal_sum', 'all_entries_finite'])
    if len(table) and np.any(np.isfinite(table.excluded_wake_self_flow) & (table.excluded_wake_self_flow != 0)):
        warnings.warn(f'{name}: Wake self-flow is excluded from EQ as specified in the manuscript; inspect matrix_audit.', stacklevel=2)
    return np.asarray(values, dtype=float), table


def fig5_paired_statistics(placebo, temazepam, metric_name, alpha=.05):
    """Same raw-value, TWO-SIDED paired t/Wilcoxon definitions supplied by user."""
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1.')
    p = _fig5_vector(placebo, metric_name + '.Placebo')
    t = _fig5_vector(temazepam, metric_name + '.Temazepam')
    if p.shape != t.shape:
        raise ValueError(f'{metric_name}: conditions have unequal lengths.')
    n_input = len(p)
    keep = np.isfinite(p) & np.isfinite(t)
    p, t = p[keep], t[keep]
    d = t - p
    if not np.isfinite(d).all():
        raise ValueError('Paired differences overflowed; rescale the inputs.')
    n = len(p)
    mp, mt = (float(p.mean()), float(t.mean())) if n else (np.nan, np.nan)
    md = float(d.mean()) if n else np.nan
    pct = float(100 * ((mt - mp) / mp)) if n and mp != 0 else np.nan
    ts = tp = ws = wp = np.nan
    status = 'ok'
    if n < 2:
        status = 'insufficient_pairs'
    elif np.all(d == 0):
        ws, wp = 0., 1.
        status = 'all_zero_differences; t_test_undefined; wilcoxon_p1_convention'
    else:
        if np.all(d == d[0]):
            ts, tp = float(np.copysign(np.inf, d[0])), 0.
            status = 'constant_nonzero_differences; t_test_zero_standard_error'
        else:
            test = stats.ttest_rel(t, p, alternative='two-sided')
            ts, tp = float(test.statistic), float(test.pvalue)
        test = stats.wilcoxon(d, alternative='two-sided', zero_method='wilcox',
                              correction=False, method='auto')
        ws, wp = float(test.statistic), float(test.pvalue)
    return dict(Metric=metric_name, N_input=n_input, N_pairs=n, N_removed=n_input-n,
        N_nonzero_differences=int(np.count_nonzero(d)), Placebo_mean=mp, Temazepam_mean=mt,
        Placebo_sd=float(p.std(ddof=1)) if n >= 2 else np.nan,
        Temazepam_sd=float(t.std(ddof=1)) if n >= 2 else np.nan,
        Mean_difference_Temazepam_minus_Placebo=md, Percent_change_from_Placebo=pct,
        Direction='Undefined' if not n else 'Increase' if md > 0 else 'Decrease' if md < 0 else 'No change',
        Paired_t_stat=ts, Paired_t_df=n-1 if n >= 2 else np.nan, Paired_t_p=tp,
        Wilcoxon_stat=ws, Wilcoxon_p=wp, Alpha=float(alpha),
        Paired_t_significant=bool(tp < alpha) if np.isfinite(tp) else None,
        Wilcoxon_significant=bool(wp < alpha) if np.isfinite(wp) else None,
        Test_status=status)


def _fig5_change_ci(p, t, n_boot, seed):
    """Percentile CI of 100*(mean(T)-mean(P))/mean(P), paired subject bootstrap."""
    if len(p) < 2 or n_boot == 0 or p.mean() == 0:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(p), len(p))
        mp, mt = p[idx].mean(), t[idx].mean()
        if mp != 0:
            values.append(100 * ((mt - mp) / mp))
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan, len(values)
    lo, hi = np.quantile(values, [.025, .975])
    return float(lo), float(hi), len(values)


def _fig5_ptext(p):
    if not np.isfinite(p):
        return r'$p=\mathrm{NA}$'
    if p == 0:
        return r'$p\approx 0$'
    coefficient, exponent = f'{p:.2e}'.split('e')
    return rf'$p={coefficient}\times 10^{{{int(exponent)}}}$'






def _fig5_text_geometry(fig,strict=False):
    """Check complete rendered text artists, treating math scripts as one label."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer();items=[];seen=set()
    def keep(t):
        if id(t) in seen or not t.get_visible() or not t.get_text():return
        seen.add(id(t));items.append((t,t.get_window_extent(renderer)))
    for a in fig.axes:
        for t in a.texts:keep(t)
        if not a.axison:continue
        if a.xaxis.get_visible():
            keep(a.xaxis.label)
            lo,hi=sorted(a.get_xlim())
            for t in a.get_xticklabels():
                if lo-1e-10<=t.get_position()[0]<=hi+1e-10:keep(t)
        if a.yaxis.get_visible():
            keep(a.yaxis.label)
            lo,hi=sorted(a.get_ylim())
            for t in a.get_yticklabels():
                if lo-1e-10<=t.get_position()[1]<=hi+1e-10:keep(t)
    for legend in fig.legends:
        for t in legend.get_texts():keep(t)
    errors=[]
    for i,(t,box) in enumerate(items):
        if box.x0<0 or box.y0<0 or box.x1>fig.bbox.x1 or box.y1>fig.bbox.y1:
            errors.append('clipped: '+t.get_text())
        for other,otherbox in items[i+1:]:
            if box.overlaps(otherbox):errors.append(t.get_text()+' / '+other.get_text())
    if errors and strict:raise RuntimeError('Rendered Fig5 text overlaps: '+'; '.join(errors[:8]))
    return {'result':'NEEDS_REVIEW' if errors else 'PASS','errors':errors,'complete_text_artists':len(items),'boxes_pixels':[
        {'text':t.get_text(),'bounds':list(b.bounds)} for t,b in items]}


def plot_fig5(placebo_awakening, temazepam_awakening, placebo_eq_wake, temazepam_eq_wake,
              placebo_lf_wake, temazepam_lf_wake, *, subject_ids=None,
              font_size=10., fig_width_mm=180., fig_height_mm=85.,
              n_boot=0, random_seed=1, show_violin=True, show_ci=False,
              show_t_test=True, output_dir='Fig5_output', basename='Fig5_Temazepam',
              dpi=600, matrix_audit=None, strict_layout=False, font_family=None, _qa_callback=None):
    """Three paired Wake-related panels, without an EQ/LF response-contrast block.

    Arrays must already follow identical subject order; never sorted or truncated.
    Subject IDs are optional labels, not an automatic realignment instruction.
    Horizontal bars show MEDIANS, as in the supplied actual violin helper.
    """
    for key, value in [('font_size',font_size), ('fig_width_mm',fig_width_mm),
                       ('fig_height_mm',fig_height_mm), ('dpi',dpi)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite.')
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 0:
        raise ValueError('n_boot must be a nonnegative integer.')
    specs = [
        ('Awakening frequency', placebo_awakening, temazepam_awakening, 'Awakening frequency\n(/min)', '/min'),
        ('Wake-related equilibrium', placebo_eq_wake, temazepam_eq_wake,
         'Equilibrium force to Wake\n'+r'$(\mathrm{DF}_{\mathrm{EQ}}^{\mathrm{Wake}})$', 'probability flow'),
        ('Wake-related loop flux', placebo_lf_wake, temazepam_lf_wake,
         'Wake-related loop flux\n'+r'$(\mathrm{DF}_{\mathrm{LF}}^{\mathrm{Wake}})$', 'probability flow'),
    ]
    q = font_size/7.
    resolved_font=mpl.font_manager.FontProperties(fname=mpl.font_manager.findfont(
        mpl.font_manager.FontProperties(family=font_family or ['Arial','Helvetica','DejaVu Sans']))).get_name()
    rc = {'font.weight':'normal','axes.labelweight':'normal','text.usetex':False,
          'mathtext.fontset':'dejavusans','mathtext.default':'it','font.family':resolved_font, 'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
          'font.size':7*q, 'axes.linewidth':.55, 'axes.spines.top':False,
          'axes.spines.right':False, 'axes.labelcolor':'#263844', 'text.color':'#263844',
          'xtick.color':'#263844', 'ytick.color':'#263844', 'svg.fonttype':'none',
          'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.facecolor':'white'}
    colors = ('#65747F', '#00857B')
    sources, rows, axes = [], [], []
    # Validate ALL inputs before creating a figure or writing any output.
    prepared = []
    for name, p, t, ylabel, unit in specs:
        p, t = _fig5_vector(p, name+'.Placebo'), _fig5_vector(t, name+'.Temazepam')
        if len(p) != len(t):
            raise ValueError(name+': unmatched vector lengths; no truncation allowed.')
        ids = np.arange(len(p)).astype(str) if subject_ids is None else np.asarray(subject_ids).astype(str)
        if ids.ndim != 1 or len(ids) != len(p) or len(np.unique(ids)) != len(ids):
            raise ValueError('subject_ids must be unique and match every panel input length.')
        prepared.append((name,p,t,ylabel,unit,ids))
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(fig_width_mm/25.4, fig_height_mm/25.4), dpi=150)
        layout = fig.add_axes([0,0,1,1], label='layout')
        layout.set(xlim=(0,180), ylim=(0,68)); layout.axis('off')
        def txt(x,y,s,size=6.,weight='normal',color='#263844',ha='left'):
            return layout.text(x,y,s,fontsize=size*q,fontweight=weight,color=color,ha=ha,va='center')
        for j,(name,p,t,ylabel,unit,ids) in enumerate(prepared):
            included = np.isfinite(p) & np.isfinite(t)
            source = pd.DataFrame(dict(metric=name, paired_index=np.arange(len(p)), subject_id=ids,
                                      placebo=p, temazepam=t, included=included, unit=unit))
            source['exclusion_reason'] = np.where(included, '', 'non-finite paired value')
            source['difference_T_minus_P'] = t-p
            sources.append(source)
            row = fig5_paired_statistics(p,t,name)
            pv, tv = p[included], t[included]
            lo, hi, nb = _fig5_change_ci(pv,tv,n_boot,random_seed+j)
            row.update(Change_ci_low=lo,Change_ci_high=hi,Bootstrap_valid=nb,
                       Placebo_median=float(np.median(pv)) if len(pv) else np.nan,
                       Temazepam_median=float(np.median(tv)) if len(tv) else np.nan)
            rows.append(row)
            x = 19+60*j
            ax = fig.add_axes([x/180,14/68,32/180,30/68],label=f'panel_{j}')
            axes.append(ax)
            txt(4+60*j,63,'abc'[j],8.,'bold')
            pct = row['Percent_change_from_Placebo']
            txt(x+16,63,f'Change: {pct:+.1f}%' if np.isfinite(pct) else 'Change: undefined',
                6.3,'bold',color=colors[1],ha='center')
            if show_ci:
                txt(x+16,58,f'95% CI [{lo:+.1f}, {hi:+.1f}]%' if np.isfinite(lo) else '95% CI: NA',5.5,ha='center')
            txt(x+16,58 if not show_ci else 53,'W: '+_fig5_ptext(row['Wilcoxon_p']),7.2,ha='center')
            if show_t_test:
                txt(x+16,53 if not show_ci else 48,'t: '+_fig5_ptext(row['Paired_t_p']),7.2,ha='center')
            jitter = np.random.default_rng(random_seed).uniform(-.045,.045,len(p))
            for pos in np.flatnonzero(included):
                line, = ax.plot([jitter[pos],1+jitter[pos]],[p[pos],t[pos]],
                                 color='#AAB5BE',lw=.45,alpha=.48,zorder=1)
                line.set_gid(f'paired-{j}-{pos}')
            for k,y in enumerate((pv,tv)):
                if show_violin and len(y)>1 and np.ptp(y)>0:
                    grid = np.linspace(float(y.min()),float(y.max()),160)
                    density = stats.gaussian_kde(y)(grid)
                    width = .27*density/density.max()
                    inner = -.10 if k == 0 else 1.10
                    outer = inner-width if k == 0 else inner+width
                    cloud = ax.fill_betweenx(grid,inner,outer,color=colors[k],alpha=.22,lw=0,zorder=0)
                    cloud.set_gid(f'half-density-{j}-{k}')
                points = ax.scatter(k+jitter[included],y,s=7*q*q,color=colors[k],alpha=.78,
                                    edgecolors='none',zorder=3)
                points.set_gid(f'points-{j}-{k}')
                if len(y):
                    bar, = ax.plot([k-.12,k+.12],[np.median(y)]*2,color='#263844',lw=1.15,zorder=4)
                    bar.set_gid(f'median-{j}-{k}')
            values = np.r_[pv,tv]
            upper = float(values.max())*1.10 if len(values) and values.max()>0 else 1.
            ax.set(xlim=(-.43,1.43),ylim=(0,upper),xticks=[0,1],xticklabels=['Placebo','Temazepam'])
            ax.set_ylabel(ylabel,fontsize=7.2*q,labelpad=3)
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,pos:f'{v:.3g}'))
            ax.tick_params(length=2,width=.5,pad=2,labelsize=5.5*q)
            for tick,color in zip(ax.get_xticklabels(),colors):
                tick.set_color(color)
        paired_counts=[int(row['N_pairs']) for row in rows]
        if len(set(paired_counts))==1:
            sample_note=f'n = {paired_counts[0]} paired participants'
        else:
            sample_note='n (a/b/c) = '+ '/'.join(map(str,paired_counts))+' paired participants'
        shared_note=txt(90,3.5,sample_note+'; W: Wilcoxon; t: paired t-test.',5.5,ha='center')
        shared_note.set_gid('shared-sample-and-test-note')
        fig.canvas.draw()
        positions = np.array([a.get_position().bounds for a in axes])
        # Portable blocking alignment gate; no external skill dependency in notebook.
        pt = np.array([fig_width_mm,fig_height_mm,fig_width_mm,fig_height_mm])*72/25.4
        boxes = positions*pt
        gaps = boxes[1:,0]-(boxes[:-1,0]+boxes[:-1,2])
        if any(np.ptp(boxes[:,k])>1.5 for k in (1,2,3)) or np.ptp(gaps)>1.5:
            raise RuntimeError('Panel alignment failed (1.5 pt tolerance).')
        alignment = dict(tolerance_pt=1.5,bounds_pt=boxes.tolist(),gaps_pt=gaps.tolist(),result='PASS')
        source_data = pd.concat(sources,ignore_index=True)
        statistics = pd.DataFrame(rows)
        metadata = dict(figure_size_mm=[fig_width_mm,fig_height_mm],font_size=font_size,font_family=resolved_font,
            pairing='Input subject order, independently finite per metric; no sorting or truncation',
            identity='Supplied subject IDs' if subject_ids is not None else 'Positional paired_index, not verified participant identity',
            center='Median (matches supplied draw_violin_with_points helper)',scale='Raw linear values; zeros retained',
            percent_change='100*(mean(Temazepam)-mean(Placebo))/mean(Placebo), same retained pairs',
            ci=('Not calculated: n_boot=0; no confidence intervals requested' if n_boot==0 else '95% percentile paired-participant bootstrap of the same signed ratio-of-means statistic'),
            show_confidence_intervals=bool(show_ci),
            n_boot=n_boot,seed=random_seed,shared_sample_annotation=sample_note,p_values='Two-sided, unadjusted, raw paired observations',
            c_definition='Supplied participant-level PlaceboWakeCycleStrength / TemazepamWakeCycleStrength; full contribution of loops containing Wake',
            b_definition='Sum of symmetric flow from non-Wake states into Wake, excluding Wake self-flow as in manuscript',
            c_weighting='Already aggregated by user; no reweighting or extra edge-count multiplication; exact original decomposition not re-verified',
            simulations='No raw scientific values are fabricated; uses supplied arrays only')
        result = dict(figure=fig,axes=np.asarray(axes),statistics=statistics,source_data=source_data,
                      matrix_audit=matrix_audit,metadata=metadata,alignment=alignment,paths={})
        if _qa_callback is not None:
            _qa_callback(result)
        result['text_geometry']=_fig5_text_geometry(fig,strict=strict_layout)
        if result['text_geometry']['errors']:
            warnings.warn('Fig5 exported with layout review notes; see text_geometry.json: '+ '; '.join(result['text_geometry']['errors'][:4]),stacklevel=2)
        if output_dir is not None:
            out = Path(output_dir);out.mkdir(parents=True,exist_ok=True)
            for fmt in ('pdf','svg','png'):
                path=out/f'{basename}.{fmt}'
                fig.savefig(path,dpi=dpi,facecolor='white')
                result['paths'][fmt]=str(path.resolve())
            statistics.to_csv(out/f'{basename}_statistics.csv',index=False)
            source_data.to_csv(out/f'{basename}_source_data.csv',index=False)
            if matrix_audit is not None:
                matrix_audit.to_csv(out/f'{basename}_matrix_audit.csv',index=False)
            for suffix,obj in [('metadata',metadata),('alignment',alignment),('text_geometry',result['text_geometry'])]:
                (out/f'{basename}_{suffix}.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

# Final supplementary drug figure definitions, original Telemetry cell 33.
# Cell 1: def. Dependencies: numpy, pandas, scipy, matplotlib.
# Reuses the existing k-scan results; no duplicate cycle-coverage engine.
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats


def _sid_vector(x,name):
    a=np.asarray(x,float)
    if a.ndim!=1 or not len(a):raise ValueError(name+': expected a nonempty 1D vector.')
    return a.copy()


def _sid_paired(placebo,temazepam,*,block,metric,stage='',min_pairs=2):
    p=_sid_vector(placebo,metric+'.Placebo');t=_sid_vector(temazepam,metric+'.Temazepam')
    if p.shape!=t.shape:raise ValueError(metric+': paired lengths differ; align subjects before plotting.')
    valid=np.isfinite(p)&np.isfinite(t);pp=p[valid];tt=t[valid];diff=tt-pp;n=len(pp)
    mean_p=float(pp.mean()) if n else np.nan;mean_t=float(tt.mean()) if n else np.nan
    change=100*(mean_t-mean_p)/mean_p if n and mean_p!=0 else np.nan
    wp=ws=tp=ts=np.nan;status='ok'
    if n<min_pairs:status='insufficient pairs'
    elif np.all(diff==0):ws=0.;wp=1.;status='all zero differences; Wilcoxon p=1 convention'
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            result=stats.wilcoxon(tt,pp,alternative='two-sided',zero_method='wilcox',method='auto')
            ws=float(result.statistic);wp=float(result.pvalue)
            result=stats.ttest_rel(tt,pp,alternative='two-sided');ts=float(result.statistic);tp=float(result.pvalue)
    row=dict(block=block,stage=stage,metric=metric,n_input=len(p),n_pairs=n,n_excluded=int((~valid).sum()),
        placebo_mean=mean_p,temazepam_mean=mean_t,mean_difference_T_minus_P=mean_t-mean_p,
        change_percent=change,wilcoxon_statistic=ws,wilcoxon_p=wp,paired_t_statistic=ts,paired_t_p=tp,status=status)
    source=pd.DataFrame(dict(block=block,stage=stage,metric=metric,pair_position=np.arange(len(p)),
        placebo=p,temazepam=t,included=valid,exclusion_reason=np.where(valid,'','non-finite paired value')))
    return row,source


def _sid_flow_summary(collection):
    """Derive only summaries missing from the notebook; no loop reconstruction."""
    m=np.asarray(collection,float)
    if m.ndim!=3 or m.shape[1:]!=(5,5) or len(m)==0:raise ValueError('Expected a nonempty (n,5,5) flow collection.')
    off=m.copy();idx=np.arange(5);off[:,idx,idx]=0.
    return dict(total=off.sum(axis=(1,2)),incoming=off.sum(axis=1),all_entries=m.sum(axis=(1,2)),incoming_all=m.sum(axis=1))


def prepare_si_drug(global_pairs,stage_frequency_data,stage_flux_data,stage_irr_flux_data,scan_result):
    """Prepare one SI figure from original paired vectors and original scan df_stats.
    All pairs must already be in matching subject order. Never sort or unpair vectors.
    a/b: two-sided Wilcoxon; t-test exported but not displayed; no CI calculations.
    c: retain original df_stats p values without re-testing or altering directions.
    Stage incoming irreversible flux is NOT full-edge flux of all loops containing a stage.
    """
    a_stats=[];b_stats=[];sources=[]
    for key in ('hypnocycle_frequency','total_lf','transition_frequency','total_eq'):
        if key not in global_pairs:raise KeyError('Missing global pair: '+key)
        row,source=_sid_paired(*global_pairs[key],block='a',metric=key)
        a_stats.append(row);sources.append(source)
    for stage in ('N1','N2','N3','REM'):
        for metric,table in [('entry_frequency',stage_frequency_data),('incoming_eq',stage_flux_data),('incoming_irr',stage_irr_flux_data)]:
            if stage not in table:raise KeyError(metric+': missing '+stage)
            row,source=_sid_paired(*table[stage],block='b',metric=metric,stage=stage)
            b_stats.append(row);sources.append(source)
    scan=(scan_result['df_stats'] if isinstance(scan_result,dict) else scan_result).copy(deep=True)
    required={'k','metric','p_value','n_pairs','mean_diff_T_minus_P'}
    if required-set(scan):raise KeyError('Existing k-scan df_stats missing: '+str(sorted(required-set(scan))))
    keys=('frequency','mean_duration','force_eq','force_lf','force_total')
    missing=set(keys)-set(scan.metric)
    if missing:raise ValueError('k-scan is missing metrics: '+str(sorted(missing)))
    scan=scan.loc[scan.metric.isin(keys)].copy()
    if scan.duplicated(['k','metric']).any():raise ValueError('Duplicate (k,metric) rows in k-scan.')
    finite=np.isfinite(pd.to_numeric(scan.p_value,errors='coerce'))
    if ((scan.loc[finite,'p_value']<0)|(scan.loc[finite,'p_value']>1)).any():raise ValueError('Scan p values must be in [0,1].')
    subject_values=scan_result.get('df_subject_values') if isinstance(scan_result,dict) else None
    return dict(a_stats=pd.DataFrame(a_stats),b_stats=pd.DataFrame(b_stats),paired_source=pd.concat(sources,ignore_index=True),
                scan=scan,scan_subject_values=None if subject_values is None else subject_values.copy(deep=True),
                metadata={'paired_order':'Supplied subject order, never sorted; caller must align identities.',
                    'change_definition':'100*(mean_T-mean_P)/mean_P on finite retained pairs; undefined for mean_P=0.',
                    'a_b_test':'Two-sided Wilcoxon signed-rank; wilcox zero method, auto method, unadjusted p. t-tests exported only.',
                    'c_test':'Original scan df_stats p values retained. Supplied scan uses two-sided Wilcoxon, min_n_pairs=3.',
                    'stage_frequency_unit':'Input per-hour-sleep values retained for tests and percent changes; summary displays only percent change.',
                    'global_frequency_unit':'Count / TST minutes = /min.',
                    'incoming_irr':'Sum of the supplied irreversible-flow matrix column; not all-edge Wake/stage-related loop flux.',
                    'summary_limit':'b shows paired-mean percentage effects and p values; individual observations remain in paired_source CSV.',
                    'multiple_testing':'All displayed p values unadjusted; k points are related thresholds, not independent replications.'})


def _sid_pmath(p):
    if not np.isfinite(p):return r'$p=\mathrm{NA}$'
    if p==0:return r'$p\approx0$'
    exponent=int(np.floor(np.log10(p)));coefficient=round(p/10.**exponent,2)
    if coefficient>=10:coefficient/=10;exponent+=1
    return rf'$p={coefficient:.2f}\times10^{{{exponent}}}$'


def plot_si_drug(data,*,font_size=10.,panel_label_size=12.,fig_width_mm=180.,fig_height_mm=236.,
                 significance_alpha=.05,significant_color='#F2B880',nonsignificant_color='#EDF0F2',
                 point_size=9.,random_seed=1,p_plot_floor=1e-300,
                 output_dir='SI_Drug_output',basename='SI_Drug_combined',dpi=600,show=True):
    """a four paired plots; b 4-stage x 3-metric effect matrix; c five k-scan curves.
    Filled k markers denote mean_T>mean_P; open markers denote mean_T<mean_P.
    Panel-b colour is based on unadjusted Wilcoxon p only: significant / non-significant; missing p is white.
    No confidence intervals are drawn or calculated.
    """
    for name,v in [('font_size',font_size),('panel_label_size',panel_label_size),('fig_width_mm',fig_width_mm),('fig_height_mm',fig_height_mm),('dpi',dpi)]:
        if not np.isfinite(v) or v<=0:raise ValueError(name+' must be positive and finite.')
    if not 0<p_plot_floor<1:raise ValueError('p_plot_floor must lie in (0,1).')
    if not 0<significance_alpha<1:raise ValueError('significance_alpha must lie in (0,1).')
    rc={'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':font_size,'axes.labelsize':font_size,'xtick.labelsize':font_size,'ytick.labelsize':font_size,
        'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.usetex':False,'legend.frameon':False}
    rng=np.random.default_rng(random_seed);axes={};paths={};placebo_color='#879BA9';tema_color='#009E9A';ink='#263844';gray='#71808A'
    acolors={'frequency':'#332288','mean_duration':'#CC79A7','force_eq':'#0072B2','force_lf':'#D55E00','force_total':'#65747F'}
    labels={'frequency':'NREM–REM frequency','mean_duration':'NREM–REM mean duration',
            'force_eq':'Uncovered EQ','force_lf':'Selected loop flux','force_total':'Total force'}
    with mpl.rc_context(rc):
        fig=plt.figure(figsize=(fig_width_mm/25.4,fig_height_mm/25.4),dpi=150)
        def text(x,y,s,**kw):return fig.text(x/180,y/236,s,fontsize=font_size,va='center',**kw)
        def add(x,y,w,h,key):
            a=fig.add_axes([x/180,y/236,w/180,h/236],label=key);axes[key]=a;return a
        for letter,y in [('a',220),('b',153),('c',80)]:fig.text(3/180,y/236,letter,fontsize=panel_label_size,weight='bold',va='center')
        fig.legend(handles=[Line2D([],[],ls='',marker='o',color=placebo_color,label='Placebo'),
                            Line2D([],[],ls='',marker='o',color=tema_color,label='Temazepam')],
                   loc='center',bbox_to_anchor=(.58,231/236),ncol=2,fontsize=font_size,handletextpad=.3,columnspacing=1.)
        aheaders=['Hypnocycle\nfrequency','Total\nloop flux','Transition\nfrequency','Total\nequilibrium']
        alabels=['/min',r'$\mathrm{DF}_{\mathrm{LF}}$','/min',r'$\mathrm{DF}_{\mathrm{EQ}}$']
        for i,row in enumerate(data['a_stats'].itertuples(index=False)):
            left=22+42*i;a=add(left,175,24,26,'a_'+str(i))
            part=data['paired_source'].query('block=="a" and metric==@row.metric and included').copy()
            p=part.placebo.to_numpy();t=part.temazepam.to_numpy();jitter=rng.uniform(-.045,.045,len(p))
            for j in range(len(p)):a.plot([jitter[j],1+jitter[j]],[p[j],t[j]],color='#A8B5BD',alpha=.4,lw=.5,zorder=1)
            for pos,v,color in [(0,p,placebo_color),(1,t,tema_color)]:
                if len(v):
                    a.scatter(pos+jitter,v,s=point_size,color=color,alpha=.7,linewidths=0,zorder=3)
                    a.plot([pos-.18,pos+.18],[np.median(v),np.median(v)],color=ink,lw=1.2,zorder=4)
            a.set(xlim=(-.35,1.35),xticks=[0,1],xticklabels=['P','T'])
            a.set_ylabel(alabels[i],labelpad=3);a.spines[['top','right']].set_visible(False)
            a.tick_params(length=2,pad=2)
            a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3));a.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.3g}'))
            change=f'{row.change_percent:+.1f}%' if np.isfinite(row.change_percent) else 'undefined'
            text(left+12,220,aheaders[i],ha='center',weight='bold')
            text(left,212,'Change: '+change,color=tema_color)
            text(left,205,_sid_pmath(row.wilcoxon_p))
        na=data['a_stats'].n_pairs.to_numpy();ntext=f'n = {na[0]} paired participants' if np.ptp(na)==0 else 'Paired n (left to right): '+', '.join(map(str,na))
        text(90,162,ntext+'; P: placebo; T: temazepam',ha='center',color=gray)
        stages=('N1','N2','N3','REM');metrics=('entry_frequency','incoming_eq','incoming_irr')
        p_matrix=data['b_stats'].pivot(index='stage',columns='metric',values='wilcoxon_p').reindex(index=stages,columns=metrics).to_numpy()
        cell_colors=np.ones((4,3,4))
        finite_p=np.isfinite(p_matrix)
        cell_colors[finite_p & (p_matrix<significance_alpha)]=mpl.colors.to_rgba(significant_color)
        cell_colors[finite_p & (p_matrix>=significance_alpha)]=mpl.colors.to_rgba(nonsignificant_color)
        b=add(27,96,138,44,'b')
        b.imshow(cell_colors,aspect='auto',interpolation='nearest')
        b.set_yticks(range(4),stages);b.set_xticks(range(3),['Entry\nfrequency','Incoming\nsymmetric flow','Incoming\nirreversible flux'])
        b.xaxis.tick_top();b.tick_params(length=0,pad=4)
        b.spines[:].set_visible(False)
        for i,stage in enumerate(stages):
            for j,metric in enumerate(metrics):
                row=data['b_stats'].query('stage==@stage and metric==@metric').iloc[0]
                change=f'{row.change_percent:+.1f}%' if np.isfinite(row.change_percent) else 'undefined'
                b.text(j,i-.23,change,ha='center',va='center',fontsize=font_size,weight='bold',color=ink)
                b.text(j,i+.23,_sid_pmath(row.wilcoxon_p),ha='center',va='center',fontsize=font_size,color=ink)
        for x in [.5,1.5]:b.axvline(x,color='white',lw=1)
        for y in [.5,1.5,2.5]:b.axhline(y,color='white',lw=1)
        nb=data['b_stats'].n_pairs.to_numpy();bnote=f'n = {nb[0]} pairs per cell' if np.ptp(nb)==0 else f'Paired n per cell: {nb.min()}–{nb.max()} (see CSV)'
        text(96,88,bnote+f'; orange: p < {significance_alpha:g}; gray: p ≥ {significance_alpha:g}',ha='center',color=gray)
        c=add(17,17,148,40,'c');curve_source=data['scan'].copy(deep=True)
        curve_source['plot_neglog10_p']=-np.log10(np.maximum(curve_source.p_value.to_numpy(float),p_plot_floor))
        for key in labels:
            part=curve_source.loc[curve_source.metric==key].sort_values('k');color=acolors[key]
            c.plot(part.k,part.plot_neglog10_p,color=color,lw=1.1)
            positive=part.mean_diff_T_minus_P.to_numpy()>0
            negative=part.mean_diff_T_minus_P.to_numpy()<0
            for mask,face in [(positive,color),(negative,'white')]:
                c.scatter(part.k.to_numpy()[mask],part.plot_neglog10_p.to_numpy()[mask],s=11,facecolors=face,edgecolors=color,linewidths=.7,zorder=3)
        threshold=-np.log10(.05);c.axhline(threshold,color=gray,lw=.8,ls='--')
        text(165,60,'Dashed: '+_sid_pmath(.05),ha='right',color=gray)
        c.set_xlabel('Minimum REM duration, k (min)',labelpad=3);c.set_ylabel(r'$-\log_{10}(p)$',labelpad=4)
        c.spines[['top','right']].set_visible(False);c.tick_params(length=2.5,pad=3)
        kval=curve_source.k.to_numpy();c.set_xlim(float(np.nanmin(kval)),float(np.nanmax(kval)))
        ymax=max(threshold+1.,float(np.nanmax(curve_source.plot_neglog10_p))*1.12)
        c.set_ylim(0,ymax);c.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        fig.legend(handles=[Line2D([],[],color=acolors[k],lw=1.2,label=labels[k]) for k in labels],loc='center',
                   bbox_to_anchor=(.54,70/236),ncol=3,fontsize=font_size,handlelength=1.3,columnspacing=1.)
        text(90,5,'c: filled = mean T > P; open = mean T < P. All tests: two-sided Wilcoxon.',ha='center',color=gray)
        fig.canvas.draw();renderer=fig.canvas.get_renderer();clipped=[]
        for t in fig.findobj(mpl.text.Text):
            if t.get_visible() and t.get_text().strip():
                box=t.get_window_extent(renderer)
                if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.x1+.5 or box.y1>fig.bbox.y1+.5:clipped.append(t.get_text())
        if clipped:raise ValueError('Text outside canvas; enlarge physical dimensions: '+repr(clipped[:6]))
        bounds={k:list(a.get_position().bounds) for k,a in axes.items()}
        for k in ('a_1','a_2','a_3'):
            if not np.allclose(np.array(bounds[k])[1:],np.array(bounds['a_0'])[1:]):raise RuntimeError('Global paired panels are not aligned.')
        alignment={'status':'PASS','bounds_fraction':bounds,'canvas_mm':[fig_width_mm,fig_height_mm],
            'exemptions':{'b':'Effect-summary matrix, intentionally different width from paired axes','c':'Full-width parameter scan'}}
        paths={}
        if output_dir is not None:
            folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
            for ext in ('pdf','svg','png'):
                path=folder/(basename+'.'+ext);fig.savefig(path,format=ext,dpi=dpi,facecolor='white');paths[ext]=str(path.resolve())
            tables={'global_statistics':data['a_stats'],'stage_statistics':data['b_stats'],
                    'paired_source':data['paired_source'],'scan_original_statistics':data['scan'],'scan_plot_source':curve_source}
            if data['scan_subject_values'] is not None:tables['scan_subject_values']=data['scan_subject_values']
            for suffix,table in tables.items():
                path=folder/(basename+'_'+suffix+'.csv');table.to_csv(path,index=False,encoding='utf-8-sig');paths[suffix]=str(path.resolve())
            meta={**data['metadata'],'alignment':alignment,'stage_colour_rule':{'alpha':significance_alpha,'significant':significant_color,'non_significant':nonsignificant_color,'missing_p':'white','p_adjustment':'none'},
                  'a_black_bars':'Medians of paired retained observations','c_plot_transform':'-log10(max(original p,plot floor)); raw p preserved.',
                  'c_direction_markers':'Sign of paired group mean difference, not sign of Wilcoxon statistic.',
                  'plot_p_floor':p_plot_floor}
            (folder/(basename+'_metadata.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        if show:plt.show()
    return dict(figure=fig,axes=axes,data=data,paths=paths,alignment=alignment)

__all__ = ['Circle', 'ConnectionPatch', 'Counter', 'FancyArrowPatch', 'Line2D', 'LinearSegmentedColormap', 'LogNorm', 'Mapping', 'MplPath', 'Patch', 'Path', 'Rectangle', 'TwoSlopeNorm', '_FIG2_ENGINE', '_FIG3_STAGE', '_FIG4_METRICS', '_Fig3LabelSpace', '_SHORT', '_STATES', '_build_drug_force_helpers', '_build_drug_loading_helpers', '_build_drug_metric_helpers', '_build_drug_scan_helpers', '_build_flow_helpers', '_build_inside_outside_helpers', '_build_loading_helpers', '_build_loop_matrix_helpers', '_build_n2n3_helpers', '_build_relaxation_helpers', '_build_simulation_helpers', '_build_temporal_helpers', '_draw_fig1', '_drug_force_helpers', '_drug_loading_helpers', '_drug_metric_helpers', '_drug_scan_helpers', '_early_sleep_forces', '_fig2_bootstrap_difference', '_fig2_check_alignment', '_fig2_family_tuple', '_fig2_force_engine', '_fig2_grouped_correlations', '_fig2_lookup', '_fig2_pearson', '_fig2_ptext', '_fig2_rc', '_fig2_save', '_fig2_size', '_fig2_text_geometry', '_fig2_vector', '_fig3_alignment', '_fig3_draw_hypnogram_example', '_fig3_duration_rank_tests', '_fig3_duration_stats', '_fig3_family', '_fig3_label', '_fig3_n3_boundaries', '_fig3_no_n3', '_fig3_occurrences', '_fig3_pearson', '_fig3_place_stack_labels', '_fig3_prepare_hypnogram_example', '_fig3_rc', '_fig3_separate_tick_corners', '_fig3_text_geometry', '_fig3_vector', '_fig4_alignment', '_fig4_ci', '_fig4_fit_top_panels', '_fig4_flux_removal_schematic', '_fig4_input_table', '_fig4_neglog_p', '_fig4_ptext', '_fig4_rc', '_fig4_relative', '_fig4_resample_indices', '_fig4_static_rows', '_fig4_text_geometry', '_fig4_threshold_notes', '_fig4_top_scale', '_fig4_tst_mapping', '_fig4_wilcoxon', '_fig5_change_ci', '_fig5_ptext', '_fig5_text_geometry', '_fig5_vector', '_flow_helpers', '_inside_outside_helpers', '_loading_helpers', '_loop_matrix_helpers', '_matrix_stack', '_n2_n3_exchange_interval', '_n2n3_helpers', '_nrem_rem_intervals', '_plot_fig3_once', '_prepare_fig1', '_read_hypnogram', '_record_collection_index', '_record_matrix', '_relaxation_helpers', '_runs', '_sid_flow_summary', '_sid_paired', '_sid_pmath', '_sid_vector', '_sif1_matrices', '_sif1_probabilities', '_sif1_probability_values', '_sif1_relative_difference', '_sif2_ptext', '_sif2_regression', '_sif2_vector', '_sif3_loop_label', '_sif4_pss', '_sif4_stack', '_simulation_helpers', '_stage_name', '_temporal_helpers', '_timing_fig1', 'analyze_inside_outside_duration_significance_by_category', 'analyze_nremrem_loop_flux_plus_sym_force_by_hypnogram', 'analyze_simulated_data_over_k', 'ast', 'build_loop_strength_matrix', 'calculate_tmat_list_ignore_invalid', 'compare_steady_time_for_Tnew_and_Tmat', 'compute_N2_N3_transition_frequency', 'compute_sleep_metrics_from_labels', 'configure_external_helpers', 'count_total_loops_all_subjects', 'csv', 'cycles_to_dataframe', 'decompose_flux_lists', 'fig5_matrix_summary', 'fig5_paired_statistics', 'find_all_temporal_cycles_all_subjects', 'first_passage_counts_array', 'generate_group_sleep_indices', 'generate_simulated_hypnogram_data', 'json', 'lights_off_epoch_idx_by_tstart_end', 'load_psg_y_only', 'load_telemetry_hypnograms', 'match_telemetry_records', 'modify_loops_and_reconstruct_Ts', 'mpl', 'norm', 'np', 'occurrences_to_dataframe', 'parse_sc_filename', 'pd', 'pearsonr', 'plot_fig2_main', 'plot_fig3', 'plot_fig4', 'plot_fig5', 'plot_si_drug', 'plot_si_fig1', 'plot_si_fig2', 'plot_si_fig3', 'plot_si_fig4', 'plot_sleep_fig1', 'plt', 'prepare_fig2_associations', 'prepare_fig2_family_counts', 'prepare_fig3_data', 'prepare_fig4', 'prepare_si_drug', 'prepare_si_fig2', 'prepare_si_fig3', 'prepare_si_fig4', 're', 'read_telemetry_subjects', 'scan_fig2_nremrem', 'scan_k_temazepam_placebo_differences', 'stats', 'student_t', 'sum_nrem_rem_flux', 'warnings']
