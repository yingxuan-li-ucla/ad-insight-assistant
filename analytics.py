"""分析层：所有数字都在这里用 pandas 算，大模型只负责理解问题和解读结果。

设计原则：
- 比率指标（CTR / CPC / CVR / CPA / ROAS）一律先汇总再相除（ratio of sums），
  不对日级比率求平均，避免小样本日子把均值带偏。
- 每个函数的输出都能直接转成 JSON，方便作为工具结果回传给大模型。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLS = ["date", "channel", "campaign", "creative",
                 "spend", "impressions", "clicks", "conversions", "revenue"]
DIMENSIONS = ["channel", "campaign", "market", "creative"]
TIME_DIMS = ["date", "week"]
BASE_METRICS = ["spend", "impressions", "clicks", "conversions", "revenue"]
RATIO_METRICS = ["ctr", "cpc", "cpm", "cvr", "cpa", "roas"]
ALL_METRICS = BASE_METRICS + RATIO_METRICS

METRIC_LABELS = {
    "spend": "花费", "impressions": "曝光", "clicks": "点击", "conversions": "转化",
    "revenue": "收入", "ctr": "点击率", "cpc": "CPC", "cpm": "CPM", "cvr": "转化率",
    "cpa": "CPA", "roas": "ROAS",
}


# ---------------------------------------------------------------- 读数据

def load_data(src) -> pd.DataFrame:
    """读取 CSV（路径或上传的文件对象），校验字段并补充 week 列。"""
    df = pd.read_csv(src)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必需字段：{', '.join(missing)}")
    if "market" not in df.columns:
        df["market"] = "All"

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in BASE_METRICS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"] + BASE_METRICS)
    for c in DIMENSIONS:
        df[c] = df[c].astype(str)
    df["week"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    return df.sort_values("date").reset_index(drop=True)


def data_profile(df: pd.DataFrame) -> dict:
    """给大模型的数据说明：时间范围、各维度取值。"""
    return {
        "date_min": df["date"].min().strftime("%Y-%m-%d"),
        "date_max": df["date"].max().strftime("%Y-%m-%d"),
        "rows": int(len(df)),
        **{f"{d}_values": sorted(df[d].unique().tolist()) for d in DIMENSIONS},
    }


# ---------------------------------------------------------------- 基础计算

def _div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


def add_ratios(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()
    agg["ctr"] = _div(agg["clicks"], agg["impressions"])
    agg["cpc"] = _div(agg["spend"], agg["clicks"])
    agg["cpm"] = _div(agg["spend"], agg["impressions"]) * 1000
    agg["cvr"] = _div(agg["conversions"], agg["clicks"])
    agg["cpa"] = _div(agg["spend"], agg["conversions"])
    agg["roas"] = _div(agg["revenue"], agg["spend"])
    return agg


def filter_df(df: pd.DataFrame, start=None, end=None, filters: dict | None = None) -> pd.DataFrame:
    out = df
    if start:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end:
        out = out[out["date"] <= pd.Timestamp(end)]
    for dim, val in (filters or {}).items():
        if dim not in DIMENSIONS or val in (None, "", []):
            continue
        vals = val if isinstance(val, (list, tuple)) else [val]
        out = out[out[dim].isin([str(v) for v in vals])]
    return out


def summarize(df: pd.DataFrame, group_by: Iterable[str] | None = None) -> pd.DataFrame:
    """按维度汇总，返回基础指标 + 比率指标。group_by 为空时返回一行总计。"""
    group_by = [g for g in (group_by or []) if g in DIMENSIONS + TIME_DIMS]
    if df.empty:
        return pd.DataFrame(columns=group_by + ALL_METRICS)
    if group_by:
        agg = df.groupby(group_by, as_index=False)[BASE_METRICS].sum()
    else:
        agg = df[BASE_METRICS].sum().to_frame().T
    return add_ratios(agg)


def compare_periods(df, cur_start, cur_end, prev_start, prev_end,
                    group_by=None, filters=None) -> pd.DataFrame:
    """两个时间段对比，返回当期、上期和变化幅度。"""
    group_by = [g for g in (group_by or []) if g in DIMENSIONS]
    cur = summarize(filter_df(df, cur_start, cur_end, filters), group_by)
    prev = summarize(filter_df(df, prev_start, prev_end, filters), group_by)
    if group_by:
        merged = cur.merge(prev, on=group_by, how="outer", suffixes=("_cur", "_prev"))
    else:
        merged = pd.concat([cur.add_suffix("_cur"), prev.add_suffix("_prev")], axis=1)
    for m in ALL_METRICS:
        merged[f"{m}_chg"] = _div(merged[f"{m}_cur"] - merged[f"{m}_prev"], merged[f"{m}_prev"].abs())
    return merged


# ---------------------------------------------------------------- 异常检测

def _robust_z(x: float, hist: pd.Series) -> tuple[float, float]:
    """用中位数和 MAD 算稳健 z 分数，对历史里的个别极端值不敏感。"""
    med = float(hist.median())
    mad = float((hist - med).abs().median())
    if mad == 0 or np.isnan(mad):
        return 0.0, med
    return 0.6745 * (x - med) / mad, med


def _poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k)，X ~ Poisson(lam)。不引入 scipy，手写一个对数空间的版本。"""
    if lam <= 0:
        return 1.0
    log_p, total = -lam, 0.0
    for i in range(int(k) + 1):
        if i > 0:
            log_p += np.log(lam) - np.log(i)
        total += np.exp(log_p)
    return min(total, 1.0)


def _daily_flags(df: pd.DataFrame, window: int = 14, min_hist: int = 7) -> list[dict]:
    """在 campaign 日级数据上找单日异常：转化率骤降、CPC 骤升。

    转化数是小样本计数（一天几十个），日级转化率本身噪声很大，直接用 z 分数会漏报。
    这里改用泊松检验：按历史中位转化率算出今天"应该"有多少转化，
    实际转化低到概率 < 0.1% 才报警。
    """
    daily = summarize(df, ["channel", "campaign", "date"]).sort_values("date")
    flags = []
    for (channel, campaign), g in daily.groupby(["channel", "campaign"]):
        g = g.reset_index(drop=True)
        for i in range(min_hist, len(g)):
            hist = g.iloc[max(0, i - window):i]
            row = g.iloc[i]
            if row["clicks"] < 50:
                continue
            med = float(hist["cvr"].dropna().median())
            expected = row["clicks"] * med
            if med > 0 and row["cvr"] < 0.5 * med and _poisson_cdf(row["conversions"], expected) < 0.001:
                kind = "转化追踪疑似中断" if row["cvr"] < 0.2 * med else "转化率骤降"
                flags.append(dict(type=kind, metric="cvr", channel=channel, campaign=campaign,
                                  date=row["date"], value=row["cvr"], baseline=med))
            z, med = _robust_z(row["cpc"], hist["cpc"].dropna())
            if z > 3.5 and med > 0 and row["cpc"] > 1.3 * med:
                flags.append(dict(type="CPC 异常上涨", metric="cpc", channel=channel, campaign=campaign,
                                  date=row["date"], value=row["cpc"], baseline=med))
    return flags


def _merge_consecutive(flags: list[dict]) -> list[dict]:
    """把同一 campaign、同一类型、日期连续（允许隔 1 天）的单日异常合并成一个事件。"""
    events = []
    key = lambda f: (f["type"], f["channel"], f["campaign"])
    for f in sorted(flags, key=lambda f: (key(f), f["date"])):
        last = events[-1] if events else None
        if last and key(last) == key(f) and (f["date"] - last["end"]).days <= 2:
            last["end"] = f["date"]
            last["values"].append(f["value"])
            last["baselines"].append(f["baseline"])
        else:
            events.append({**f, "start": f["date"], "end": f["date"],
                           "values": [f["value"]], "baselines": [f["baseline"]]})
    return events


def _fatigue(df: pd.DataFrame, min_weeks: int = 4, drop: float = 0.30) -> list[dict]:
    """素材疲劳：比较素材头两周和最近两周的 CTR。"""
    weekly = summarize(df, ["channel", "campaign", "creative", "week"]).sort_values("week")
    out = []
    for (channel, campaign, creative), g in weekly.groupby(["channel", "campaign", "creative"]):
        if len(g) < min_weeks:
            continue
        first = summarize(df[(df["creative"] == creative) & df["week"].isin(g["week"].head(2))])
        last = summarize(df[(df["creative"] == creative) & df["week"].isin(g["week"].tail(2))])
        base, now = float(first["ctr"].iloc[0]), float(last["ctr"].iloc[0])
        if base > 0 and (now - base) / base <= -drop:
            out.append(dict(type="素材疲劳", metric="ctr", channel=channel, campaign=campaign,
                            creative=creative, start=g["week"].iloc[0],
                            end=g["week"].iloc[-1] + pd.Timedelta(days=6),
                            value=now, baseline=base))
    return out


def _low_roas(df: pd.DataFrame, threshold: float = 1.0, min_share: float = 0.05) -> list[dict]:
    """花费占比不小、但 ROAS 低于 1（投入大于产出）的 campaign。"""
    camp = summarize(df, ["channel", "campaign"])
    total = camp["spend"].sum()
    out = []
    for _, r in camp.iterrows():
        if total > 0 and r["spend"] / total >= min_share and r["roas"] < threshold:
            out.append(dict(type="ROAS 低于 1", metric="roas", channel=r["channel"],
                            campaign=r["campaign"], start=df["date"].min(), end=df["date"].max(),
                            value=r["roas"], baseline=threshold))
    return out


def _describe(e: dict) -> str:
    rng = e["start"].strftime("%m/%d") if e["start"] == e["end"] else \
        f"{e['start'].strftime('%m/%d')}–{e['end'].strftime('%m/%d')}"
    who = f"{e['channel']} / {e['campaign']}" + (f" / {e['creative']}" if e.get("creative") else "")
    v, b = e["value"], e["baseline"]
    if e["metric"] in ("cvr", "ctr"):
        nums = f"{v:.2%}（基线 {b:.2%}）"
    elif e["metric"] == "roas":
        nums = f"{v:.2f}"
    else:
        nums = f"${v:.2f}（基线 ${b:.2f}）"
    hint = {
        "转化追踪疑似中断": "点击正常但转化几乎归零，优先排查像素/归因配置",
        "转化率骤降": "检查落地页、库存或促销是否有变化",
        "CPC 异常上涨": "可能有竞品抬价，考虑调整出价或关键词",
        "素材疲劳": "该素材点击率持续下滑，建议轮换新素材",
        "ROAS 低于 1": "投入大于直接回报，需确认是否有品牌目标，否则考虑缩减预算",
    }[e["type"]]
    return f"{who}：{rng} {METRIC_LABELS[e['metric']]} {nums}。{hint}。"


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """返回所有异常事件，按严重程度和时间排序。"""
    events = []
    for e in _merge_consecutive(_daily_flags(df)):
        # 成本类指标单日波动很常见，至少持续 2 天才算事件；转化归零这种单日也要报
        if e["metric"] == "cpc" and len(e["values"]) < 2:
            continue
        e["value"] = float(np.mean(e.pop("values")))
        e["baseline"] = float(np.mean(e.pop("baselines")))
        events.append(e)
    events += _fatigue(df) + _low_roas(df)

    rows = []
    for e in events:
        chg = (e["value"] - e["baseline"]) / e["baseline"] if e["baseline"] else np.nan
        severity = "高" if e["type"] in ("转化追踪疑似中断", "ROAS 低于 1") or abs(chg) >= 0.5 else "中"
        rows.append({
            "severity": severity, "type": e["type"], "channel": e["channel"],
            "campaign": e["campaign"], "creative": e.get("creative", ""),
            "start": e["start"], "end": e["end"], "metric": e["metric"],
            "value": e["value"], "baseline": e["baseline"], "change_pct": chg,
            "description": _describe(e),
        })
    cols = ["severity", "type", "channel", "campaign", "creative", "start", "end",
            "metric", "value", "baseline", "change_pct", "description"]
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    out["_s"] = out["severity"].map({"高": 0, "中": 1})
    return out.sort_values(["_s", "start"], ascending=[True, False]).drop(columns="_s").reset_index(drop=True)


# ---------------------------------------------------------------- 周报

def full_weeks(df: pd.DataFrame) -> list[pd.Timestamp]:
    """数据里覆盖完整 7 天的周（以周一为起点）。"""
    days = df.groupby("week")["date"].nunique()
    return sorted(days[days == 7].index.tolist())


def report_stats(df: pd.DataFrame, week_start) -> dict:
    """为周报准备的全部数字。大模型写周报时只能用这里的数。"""
    ws = pd.Timestamp(week_start)
    we, ps, pe = ws + pd.Timedelta(days=6), ws - pd.Timedelta(days=7), ws - pd.Timedelta(days=1)

    total = compare_periods(df, ws, we, ps, pe).iloc[0]
    by_channel = compare_periods(df, ws, we, ps, pe, ["channel"]).sort_values("spend_cur", ascending=False)

    cur = filter_df(df, ws, we)
    cr = summarize(cur, ["channel", "creative"])
    cr = cr[cr["spend"] >= 0.02 * cr["spend"].sum()].sort_values("roas", ascending=False)

    anom = detect_anomalies(filter_df(df, None, we))
    anom = anom[(anom["end"] >= ps) & (anom["start"] <= we)]

    def pick(row, metrics, suffix=""):
        return {m: _clean(row[f"{m}{suffix}"]) for m in metrics}

    key = ["spend", "revenue", "conversions", "roas", "cpa", "ctr", "cvr"]
    return {
        "week": f"{ws:%Y-%m-%d} ~ {we:%Y-%m-%d}",
        "previous_week": f"{ps:%Y-%m-%d} ~ {pe:%Y-%m-%d}",
        "total_current": pick(total, key, "_cur"),
        "total_previous": pick(total, key, "_prev"),
        "total_change_pct": pick(total, key, "_chg"),
        "by_channel": [
            {"channel": r["channel"], **pick(r, ["spend", "revenue", "roas", "cpa"], "_cur"),
             "roas_change_pct": _clean(r["roas_chg"]), "spend_change_pct": _clean(r["spend_chg"])}
            for _, r in by_channel.iterrows()
        ],
        "top_creatives": _records(cr.head(3), ["channel", "creative", "spend", "roas", "ctr"]),
        "bottom_creatives": _records(cr.tail(3).iloc[::-1], ["channel", "creative", "spend", "roas", "ctr"]),
        "anomalies": [{"severity": a["severity"], "type": a["type"], "description": a["description"]}
                      for _, a in anom.iterrows()],
    }


def _clean(v):
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 4)
    return v


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    return [{c: _clean(r[c]) for c in cols} for _, r in df.iterrows()]


def to_records(df: pd.DataFrame, limit: int = 50) -> list[dict]:
    """DataFrame -> 可 JSON 序列化的记录列表（给大模型用）。"""
    return _records(df.head(limit), list(df.columns))


def _pct(v):
    return "—" if v is None else f"{v:+.1%}"


def template_report(s: dict) -> str:
    """没有 API key 时的规则版周报，保证 Demo 不依赖大模型也能跑。"""
    c, p, chg = s["total_current"], s["total_previous"], s["total_change_pct"]
    lines = [
        f"# 投放周报（{s['week']}）",
        "",
        "## 核心指标",
        "",
        "| 指标 | 本周 | 上周 | 环比 |",
        "|---|---|---|---|",
        f"| 花费 | ${c['spend']:,.0f} | ${p['spend']:,.0f} | {_pct(chg['spend'])} |",
        f"| 收入 | ${c['revenue']:,.0f} | ${p['revenue']:,.0f} | {_pct(chg['revenue'])} |",
        f"| 转化 | {c['conversions']:,.0f} | {p['conversions']:,.0f} | {_pct(chg['conversions'])} |",
        f"| ROAS | {c['roas']:.2f} | {p['roas']:.2f} | {_pct(chg['roas'])} |",
        f"| CPA | ${c['cpa']:.2f} | ${p['cpa']:.2f} | {_pct(chg['cpa'])} |",
        "",
        "## 分渠道",
        "",
        "| 渠道 | 花费 | ROAS | ROAS 环比 | CPA |",
        "|---|---|---|---|---|",
    ]
    for r in s["by_channel"]:
        lines.append(f"| {r['channel']} | ${r['spend']:,.0f} | {r['roas']:.2f} | "
                     f"{_pct(r['roas_change_pct'])} | ${r['cpa']:.2f} |")
    lines += ["", "## 素材表现", "", "**ROAS 最高：**"]
    lines += [f"- {r['channel']} / {r['creative']}：ROAS {r['roas']:.2f}，花费 ${r['spend']:,.0f}"
              for r in s["top_creatives"]]
    lines += ["", "**ROAS 最低：**"]
    lines += [f"- {r['channel']} / {r['creative']}：ROAS {r['roas']:.2f}，花费 ${r['spend']:,.0f}"
              for r in s["bottom_creatives"]]
    lines += ["", "## 需要关注"]
    lines += [f"- 【{a['severity']}】{a['description']}" for a in s["anomalies"]] or ["- 本周未发现明显异常"]
    lines += ["", "> 本报告由规则模板生成。接入大模型后可生成带分析和建议的版本。"]
    return "\n".join(lines)
