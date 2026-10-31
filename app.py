"""投放洞察助手 Ad Insight Assistant —— Streamlit 入口。

运行：streamlit run app.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as an
import llm

st.set_page_config(page_title="投放洞察助手", page_icon="📊", layout="wide")

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_ad_data.csv"

# 分类色按固定顺序分配给渠道（颜色跟着渠道走，不随排序变化）；深浅模式各一套
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
OTHER_COLOR = "#898781"
STATUS = {"高": ("🔴", "#d03b3b"), "中": ("🟠", "#ec835a")}
EXAMPLES = [
    "上周哪个渠道 ROAS 最差？为什么？",
    "TikTok 各素材最近四周的点击率趋势怎么样？",
    "如果要从 YouTube 挪预算出来，挪到哪里最合适？",
]


# ---------------------------------------------------------------- 工具函数

def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def channel_colors(channels: list[str]) -> dict:
    pal = PALETTE[theme_mode()]
    return {c: (pal[i] if i < len(pal) else OTHER_COLOR) for i, c in enumerate(sorted(channels))}


@st.cache_data
def load_sample() -> pd.DataFrame:
    return an.load_data(SAMPLE_PATH)


@st.cache_data
def load_upload(raw: bytes) -> pd.DataFrame:
    return an.load_data(io.BytesIO(raw))


@st.cache_data
def anomalies_for(df: pd.DataFrame) -> pd.DataFrame:
    return an.detect_anomalies(df)


def secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:  # 本地没有 secrets.toml 时会抛异常
        return default


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    return fig


COLS = {
    "spend": st.column_config.NumberColumn("花费", format="dollar"),
    "revenue": st.column_config.NumberColumn("收入", format="dollar"),
    "impressions": st.column_config.NumberColumn("曝光", format="localized"),
    "clicks": st.column_config.NumberColumn("点击", format="localized"),
    "conversions": st.column_config.NumberColumn("转化", format="localized"),
    "ctr": st.column_config.NumberColumn("CTR", format="%.2f%%"),
    "cvr": st.column_config.NumberColumn("CVR", format="%.2f%%"),
    "cpc": st.column_config.NumberColumn("CPC", format="$%.2f"),
    "cpm": st.column_config.NumberColumn("CPM", format="$%.2f"),
    "cpa": st.column_config.NumberColumn("CPA", format="$%.2f"),
    "roas": st.column_config.NumberColumn("ROAS", format="%.2f"),
    "channel": "渠道", "campaign": "广告计划", "creative": "素材", "market": "市场",
}


def show_table(t: pd.DataFrame) -> None:
    """表格展示：比率列转成百分数，统一列名和格式。"""
    t = t.copy()
    for c in ("ctr", "cvr"):
        if c in t.columns:
            t[c] = t[c] * 100
    st.dataframe(t, column_config=COLS, hide_index=True, width="stretch")


# ---------------------------------------------------------------- 侧边栏

with st.sidebar:
    st.header("数据")
    source = st.radio("数据来源", ["示例数据", "上传 CSV"], horizontal=True, label_visibility="collapsed")
    df = None
    if source == "上传 CSV":
        up = st.file_uploader("上传投放数据", type="csv")
        st.caption("需要字段：date, channel, campaign, creative, spend, impressions, clicks, "
                   "conversions, revenue；market 可选")
        if up is not None:
            try:
                df = load_upload(up.getvalue())
            except Exception as e:
                st.error(f"读取失败：{e}")
    if df is None:
        df = load_sample()
        if source == "上传 CSV":
            st.info("还没上传文件，先用示例数据演示。")

    st.download_button("下载示例数据 / 模板", SAMPLE_PATH.read_bytes(),
                       file_name="sample_ad_data.csv", mime="text/csv", width="stretch")

    st.header("大模型")
    secret_key = secret("LLM_API_KEY")
    providers = list(llm.PROVIDERS)
    default_provider = secret("LLM_PROVIDER", "DeepSeek")
    provider = st.selectbox("服务商", providers,
                            index=providers.index(default_provider) if default_provider in providers else 0)
    cfg = llm.PROVIDERS[provider]
    if secret_key:
        st.success("已配置演示用 API key")
        user_key = st.text_input("也可以换成你自己的 key", type="password")
        api_key = user_key or secret_key
    else:
        api_key = st.text_input("API key", type="password",
                                help="只保存在本次会话里，不会上传或记录")
    model = st.text_input("模型", value=secret("LLM_MODEL", cfg["model"]))
    base_url = cfg["base_url"]
    if provider.startswith("自定义"):
        base_url = st.text_input("接口地址 base_url", value=secret("LLM_BASE_URL", ""))
    llm_ready = bool(api_key and model)
    if not llm_ready:
        st.caption("没有 key 时，概览、异常检测和规则版周报照常可用；AI 问答需要 key。")

client = llm.make_client(api_key, base_url) if llm_ready else None
colors = channel_colors(df["channel"].unique().tolist())

# ---------------------------------------------------------------- 页头

st.title("投放洞察助手")
st.caption(f"Ad Insight Assistant · 数据范围 {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}"
           f" · {df['channel'].nunique()} 个渠道 · {df['creative'].nunique()} 条素材")

tab_overview, tab_anomaly, tab_chat, tab_report = st.tabs(["概览", "异常检测", "AI 问答", "周报"])

# ---------------------------------------------------------------- 概览

with tab_overview:
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    c1, c2 = st.columns([1, 2])
    with c1:
        rng = st.date_input("时间范围", value=(max(dmin, dmax - pd.Timedelta(days=27)), dmax),
                            min_value=dmin, max_value=dmax)
    with c2:
        chans = st.multiselect("渠道", sorted(df["channel"].unique()), placeholder="全部渠道")

    start, end = (rng[0], rng[1]) if isinstance(rng, tuple) and len(rng) == 2 else (dmin, dmax)
    filt = {"channel": chans} if chans else None
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    prev_end = pd.Timestamp(start) - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    cmp = an.compare_periods(df, start, end, prev_start, prev_end, filters=filt).iloc[0]
    has_prev = pd.notna(cmp["spend_prev"]) and cmp["spend_prev"] > 0

    def delta(m):
        v = cmp[f"{m}_chg"]
        return f"{v:+.1%}" if has_prev and pd.notna(v) else None

    k = st.columns(6)
    k[0].metric("花费", f"${cmp['spend_cur']:,.0f}", delta("spend"), delta_color="off")
    k[1].metric("收入", f"${cmp['revenue_cur']:,.0f}", delta("revenue"))
    k[2].metric("ROAS", f"{cmp['roas_cur']:.2f}", delta("roas"))
    k[3].metric("CPA", f"${cmp['cpa_cur']:.2f}", delta("cpa"), delta_color="inverse")
    k[4].metric("CTR", f"{cmp['ctr_cur']:.2%}", delta("ctr"))
    k[5].metric("CVR", f"{cmp['cvr_cur']:.2%}", delta("cvr"))
    st.caption(f"环比对比区间：{prev_start:%m/%d} ~ {prev_end:%m/%d}（等长的前一段时间）" if has_prev
               else "所选区间之前没有数据，不显示环比")

    cur = an.filter_df(df, start, end, filt)
    left, right = st.columns([3, 2])
    with left:
        daily = an.summarize(cur, ["date", "channel"])
        fig = go.Figure()
        for ch in sorted(daily["channel"].unique()):
            d = daily[daily["channel"] == ch]
            fig.add_scatter(x=d["date"], y=d["spend"], name=ch, mode="lines",
                            line=dict(color=colors[ch], width=2),
                            hovertemplate="$%{y:,.0f}")
        st.markdown("**每日花费（按渠道）**")
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(style_fig(fig), width="stretch")
    with right:
        by_ch = an.summarize(cur, ["channel"]).sort_values("roas")
        fig = go.Figure(go.Bar(
            x=by_ch["roas"], y=by_ch["channel"], orientation="h",
            marker=dict(color=[colors[c] for c in by_ch["channel"]], cornerradius=4),
            text=[f"{v:.2f}" for v in by_ch["roas"]], textposition="outside",
            hovertemplate="%{y}：ROAS %{x:.2f}<extra></extra>",
        ))
        fig.update_layout(hovermode="closest", bargap=0.45)
        fig.update_xaxes(range=[0, by_ch["roas"].max() * 1.18])
        st.markdown("**ROAS（按渠道）**")
        st.plotly_chart(style_fig(fig), width="stretch")
        st.caption("ROAS = 收入 ÷ 花费，低于 1 表示投入大于直接回报")

    st.subheader("广告计划明细")
    table = an.summarize(cur, ["channel", "campaign"]).sort_values("spend", ascending=False)
    show_table(table[["channel", "campaign", "spend", "revenue", "roas", "cpa", "ctr", "cvr", "conversions"]])

# ---------------------------------------------------------------- 异常检测

def anomaly_chart(a: pd.Series) -> go.Figure:
    color = STATUS[a["severity"]][1]
    sub = df[df["campaign"] == a["campaign"]]
    if a["creative"]:
        sub = sub[sub["creative"] == a["creative"]]
    weekly = a["type"] in ("素材疲劳", "ROAS 低于 1")
    s = an.summarize(sub, ["week" if weekly else "date"])
    x = s["week" if weekly else "date"]
    m = a["metric"]
    fmt = {"cvr": ".2%", "ctr": ".2%", "cpc": "$.2f", "roas": ".2f"}[m]
    fig = go.Figure(go.Scatter(x=x, y=s[m], mode="lines+markers" if weekly else "lines",
                               name=an.METRIC_LABELS[m],
                               line=dict(color=PALETTE[theme_mode()][0], width=2), marker=dict(size=8),
                               hovertemplate=f"%{{y:{fmt}}}<extra></extra>"))
    if a["type"] == "ROAS 低于 1":
        fig.add_hline(y=1, line_dash="dot", line_color=color, annotation_text="盈亏线 1.0")
    elif not weekly:
        fig.add_vrect(x0=a["start"] - pd.Timedelta(hours=12), x1=a["end"] + pd.Timedelta(hours=12),
                      fillcolor=color, opacity=0.15, line_width=0,
                      annotation_text="异常区间", annotation_position="top left")
    fig.update_yaxes(tickformat={"cvr": ".0%", "ctr": ".1%", "cpc": ".2f", "roas": ".1f"}[m],
                     tickprefix="$" if m == "cpc" else "", rangemode="tozero")
    fig.update_layout(showlegend=False)
    return style_fig(fig, 280)


with tab_anomaly:
    anom = anomalies_for(df)
    if anom.empty:
        st.success("没有检测到异常。")
    else:
        n_high = int((anom["severity"] == "高").sum())
        st.markdown(f"共发现 **{len(anom)}** 个需要关注的问题，其中 **{n_high}** 个高优先级。")
        with st.popover("检测规则说明"):
            st.markdown(
                "- **转化追踪疑似中断 / 转化率骤降**：按过去 14 天中位转化率推算当天预期转化数，"
                "实际转化低于预期一半、且泊松检验 p < 0.1%\n"
                "- **CPC 异常上涨**：CPC 稳健 z 分数 > 3.5 且高于中位数 30%，连续至少 2 天\n"
                "- **素材疲劳**：素材最近两周 CTR 比头两周下降 30% 以上（至少 4 周数据）\n"
                "- **ROAS 低于 1**：花费占比 ≥ 5% 的广告计划整体 ROAS < 1"
            )
        for i, a in anom.iterrows():
            icon = STATUS[a["severity"]][0]
            with st.expander(f"{icon} {a['severity']} · {a['type']} · {a['channel']} / {a['campaign']}"
                             f"{' / ' + a['creative'] if a['creative'] else ''}", expanded=i == 0):
                st.write(a["description"])
                st.markdown(f"**{a['campaign']}{' / ' + a['creative'] if a['creative'] else ''} · "
                            f"{'每周' if a['type'] in ('素材疲劳', 'ROAS 低于 1') else '每日'}"
                            f"{an.METRIC_LABELS[a['metric']]}**")
                st.plotly_chart(anomaly_chart(a), width="stretch", key=f"anom_{i}")

# ---------------------------------------------------------------- AI 问答

with tab_chat:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    if not llm_ready:
        st.info("在左侧填入大模型 API key 后就能用自然语言提问。\n\n"
                "工作方式：模型先理解问题，调用预先写好的分析函数取数（数字全部由代码计算），"
                "再基于结果给出结论和建议。每一步调用都能展开查看。")

    st.caption("试试这些问题：")
    ex_cols = st.columns(len(EXAMPLES))
    clicked = None
    for col, q in zip(ex_cols, EXAMPLES):
        if col.button(q, width="stretch", disabled=not llm_ready):
            clicked = q

    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("steps"):
                with st.expander(f"查看分析过程（调用了 {len(turn['steps'])} 次分析函数）"):
                    for s in turn["steps"]:
                        st.markdown(f"**{s['name']}**")
                        st.code(json.dumps(s["args"], ensure_ascii=False, indent=1), language="json")
                        if s["table"] is not None:
                            show_table(s["table"])

    typed = st.chat_input("问点什么，比如：上周花费最多的素材是哪个？", disabled=not llm_ready)
    question = typed or clicked
    if question and llm_ready:
        history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.chat]
        st.session_state.chat.append({"role": "user", "content": question})
        try:
            with st.spinner("分析中…"):
                res = llm.ask(client, model, df, question, history)
            st.session_state.chat.append({"role": "assistant", "content": res.answer, "steps": res.steps})
        except llm.LLMError as e:
            st.session_state.chat.pop()
            st.error(str(e))
        else:
            st.rerun()

    if st.session_state.chat and st.button("清空对话"):
        st.session_state.chat = []
        st.rerun()

# ---------------------------------------------------------------- 周报

with tab_report:
    weeks = an.full_weeks(df)
    if len(weeks) < 2:
        st.warning("至少需要两个完整周的数据才能生成周报。")
    else:
        labels = {f"{w:%Y-%m-%d} ~ {w + pd.Timedelta(days=6):%m-%d}": w for w in weeks[1:]}
        c1, c2 = st.columns([2, 1])
        week_label = c1.selectbox("选择周", list(labels)[::-1])
        use_ai = c2.toggle("用 AI 撰写分析和建议", value=llm_ready, disabled=not llm_ready,
                           help="关闭时使用规则模板，不消耗 API 额度")
        if st.button("生成周报", type="primary"):
            stats = an.report_stats(df, labels[week_label])
            try:
                with st.spinner("正在撰写…"):
                    md = llm.write_report(client, model, stats) if use_ai else an.template_report(stats)
                st.session_state.report = (week_label, md)
            except llm.LLMError as e:
                st.error(str(e))

        if st.session_state.get("report"):
            wl, md = st.session_state.report
            with st.container(border=True):
                st.markdown(md)
            st.download_button("下载 Markdown", md.encode("utf-8"),
                               file_name=f"weekly_report_{wl[:10]}.md", mime="text/markdown")
