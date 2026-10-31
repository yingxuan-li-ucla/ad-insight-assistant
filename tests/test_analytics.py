"""运行：python -m pytest tests"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import analytics as an  # noqa: E402
import llm  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return an.load_data(ROOT / "data" / "sample_ad_data.csv")


# ---------------------------------------------------------------- 分析层

def test_ratios_are_ratio_of_sums(df):
    s = an.summarize(df).iloc[0]
    assert s["roas"] == pytest.approx(df["revenue"].sum() / df["spend"].sum())
    assert s["ctr"] == pytest.approx(df["clicks"].sum() / df["impressions"].sum())


def test_group_totals_add_up(df):
    by = an.summarize(df, ["channel"])
    assert by["spend"].sum() == pytest.approx(df["spend"].sum())


def test_filter(df):
    out = an.filter_df(df, "2026-10-01", "2026-10-07", {"channel": ["TikTok"]})
    assert set(out["channel"]) == {"TikTok"}
    assert out["date"].min() >= pd.Timestamp("2026-10-01")
    assert out["date"].max() <= pd.Timestamp("2026-10-07")


def test_missing_columns_raise(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("date,channel\n2026-10-01,Meta\n")
    with pytest.raises(ValueError, match="缺少必需字段"):
        an.load_data(p)


def test_planted_anomalies_found(df):
    a = an.detect_anomalies(df)
    found = {(r["type"], r["campaign"], r["creative"]) for _, r in a.iterrows()}
    assert ("转化追踪疑似中断", "Retargeting_US", "") in found
    assert ("CPC 异常上涨", "Generic_Search", "") in found
    assert ("素材疲劳", "Spark_Ads_US", "ugc_unboxing") in found
    assert ("ROAS 低于 1", "Awareness_Global", "") in found
    tracking = a[a["type"] == "转化追踪疑似中断"].iloc[0]
    assert tracking["start"] == pd.Timestamp("2026-10-14")
    assert tracking["end"] == pd.Timestamp("2026-10-15")


def test_no_false_alarm_on_clean_data(df):
    """去掉埋入的异常后，日级检测不应该报警。"""
    clean = df[df["campaign"].isin(["Prospecting_US", "Brand_Search", "Spark_Ads_SEA"])]
    a = an.detect_anomalies(clean)
    assert a[a["type"].isin(["转化追踪疑似中断", "转化率骤降", "CPC 异常上涨"])].empty


def test_report_stats_json_serializable(df):
    week = an.full_weeks(df)[-1]
    s = an.report_stats(df, week)
    json.dumps(s, ensure_ascii=False)
    assert s["week"] == "2026-10-19 ~ 2026-10-25"
    assert "# 投放周报" in an.template_report(s)


# ---------------------------------------------------------------- 大模型层（用假客户端，不花钱、不联网）

def _msg(content=None, calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls))])


def _call(i, name, args):
    return NS(id=f"call_{i}", function=NS(name=name, arguments=json.dumps(args)))


class FakeClient:
    """第一轮返回工具调用，第二轮返回最终回答；记录模型实际收到的消息。"""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.seen.append(kw["messages"])
        return self.script.pop(0)


def test_agent_runs_tools_and_feeds_results_back(df):
    client = FakeClient([
        _msg(calls=[_call(1, "query_metrics", {
            "group_by": ["channel"], "start_date": "2026-10-19", "end_date": "2026-10-25",
            "sort_by": "roas", "ascending": True})]),
        _msg(content="上周 YouTube ROAS 最低。"),
    ])
    res = llm.ask(client, "fake", df, "上周哪个渠道 ROAS 最差？")
    assert res.answer == "上周 YouTube ROAS 最低。"
    assert res.steps[0]["table"].iloc[0]["channel"] == "YouTube"
    tool_msg = client.seen[1][-1]
    assert tool_msg["role"] == "tool" and "YouTube" in tool_msg["content"]


def test_agent_reports_tool_errors_to_model(df):
    client = FakeClient([
        _msg(calls=[_call(1, "compare_periods", {"current_start": "2026-10-19"})]),  # 缺参数
        _msg(content="好的"),
    ])
    res = llm.ask(client, "fake", df, "环比怎么样")
    assert "error" in client.seen[1][-1]["content"]
    assert res.steps[0]["table"] is None


def test_every_tool_runs(df):
    for name, args in [
        ("query_metrics", {"group_by": ["week", "creative"], "filters": {"channel": ["TikTok"]}}),
        ("compare_periods", {"current_start": "2026-10-19", "current_end": "2026-10-25",
                             "previous_start": "2026-10-12", "previous_end": "2026-10-18",
                             "group_by": ["channel"]}),
        ("list_anomalies", {"start_date": "2026-10-01"}),
    ]:
        out = llm.run_tool(df, name, args)
        assert not out.empty
        json.dumps(an.to_records(out), ensure_ascii=False)
