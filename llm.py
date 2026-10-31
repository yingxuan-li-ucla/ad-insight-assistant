"""大模型层：把自然语言问题翻译成对分析函数的调用（function calling），再解读结果。

为什么不让模型直接写代码 / 直接算数：
- 大模型算术不可靠，投放数据里一个 ROAS 算错就可能误导预算决策；
- 让模型执行任意代码有安全风险；
- 预先定义好的分析工具 = 可控、可测试、结果可复现。
模型只做两件事：决定调哪个工具、用什么参数；拿到结果后组织语言给结论。

使用 OpenAI 兼容接口，DeepSeek / OpenAI / 通义千问 等都能接。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

import analytics as an

PROVIDERS = {
    "DeepSeek": {"base_url": "https://api.deepseek.com", "model": "deepseek-flash"},
    "OpenAI": {"base_url": None, "model": "gpt-4.1-mini"},
    "通义千问": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
    "自定义（OpenAI 兼容）": {"base_url": "", "model": ""},
}


class LLMError(Exception):
    pass


def make_client(api_key: str, base_url: str | None):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url or None, timeout=60)


# ---------------------------------------------------------------- 工具定义

_FILTERS = {
    "type": "object",
    "description": "按维度筛选，每个维度给一个取值列表；不需要筛选就省略",
    "properties": {d: {"type": "array", "items": {"type": "string"}} for d in an.DIMENSIONS},
}
_GROUP_BY = {
    "type": "array",
    "items": {"type": "string", "enum": an.DIMENSIONS + an.TIME_DIMS},
    "description": "汇总维度；week 表示按周（周一开始）。为空则返回总计",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_metrics",
            "description": "按维度和时间范围汇总投放指标，返回花费、曝光、点击、转化、收入及 CTR/CPC/CPM/CVR/CPA/ROAS。",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": _GROUP_BY,
                    "start_date": {"type": "string", "description": "YYYY-MM-DD，含当天"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD，含当天"},
                    "filters": _FILTERS,
                    "sort_by": {"type": "string", "enum": an.ALL_METRICS},
                    "ascending": {"type": "boolean"},
                    "limit": {"type": "integer", "description": "最多返回多少行，默认 20"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_periods",
            "description": "对比两个时间段的指标，返回当期(_cur)、上期(_prev)和变化率(_chg，0.1 表示 +10%)。用于环比、前后对比、找变化原因。",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_start": {"type": "string"},
                    "current_end": {"type": "string"},
                    "previous_start": {"type": "string"},
                    "previous_end": {"type": "string"},
                    "group_by": {**_GROUP_BY, "items": {"type": "string", "enum": an.DIMENSIONS}},
                    "filters": _FILTERS,
                },
                "required": ["current_start", "current_end", "previous_start", "previous_end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_anomalies",
            "description": "返回规则引擎检测到的异常事件（追踪中断、CPC 异常、素材疲劳、ROAS<1 等），可按时间段筛选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
            },
        },
    },
]


def run_tool(df: pd.DataFrame, name: str, args: dict) -> pd.DataFrame:
    """执行一个工具调用，返回 DataFrame。参数都做了容错。"""
    if name == "query_metrics":
        data = an.filter_df(df, args.get("start_date"), args.get("end_date"), args.get("filters"))
        out = an.summarize(data, args.get("group_by"))
        sort_by = args.get("sort_by")
        if sort_by in out.columns:
            out = out.sort_values(sort_by, ascending=bool(args.get("ascending", False)))
        return out.head(int(args.get("limit") or 20))

    if name == "compare_periods":
        return an.compare_periods(
            df, args["current_start"], args["current_end"],
            args["previous_start"], args["previous_end"],
            args.get("group_by"), args.get("filters"),
        )

    if name == "list_anomalies":
        a = an.detect_anomalies(df)
        if args.get("start_date"):
            a = a[a["end"] >= pd.Timestamp(args["start_date"])]
        if args.get("end_date"):
            a = a[a["start"] <= pd.Timestamp(args["end_date"])]
        return a.drop(columns=["value", "baseline"])

    raise ValueError(f"未知工具：{name}")


# ---------------------------------------------------------------- 问答

def system_prompt(df: pd.DataFrame) -> str:
    profile = an.data_profile(df)
    weeks = an.full_weeks(df)
    last_week = f"{weeks[-1]:%Y-%m-%d} ~ {weeks[-1] + pd.Timedelta(days=6):%Y-%m-%d}" if weeks else "无"
    return f"""你是一名广告投放数据分析师，帮营销团队解读投放数据。

## 数据
{json.dumps(profile, ensure_ascii=False)}
"今天"按数据最后一天 {profile['date_max']} 算；"上周/最近一周"指最后一个完整周 {last_week}。

## 指标口径
CTR=点击/曝光，CPC=花费/点击，CPM=花费/曝光×1000，CVR=转化/点击，CPA=花费/转化，ROAS=收入/花费。
金额单位为美元。比率是先汇总再相除。

## 规则
1. 任何数字都必须来自工具返回结果，禁止自己估算或心算新数字。需要的数没有就再调一次工具。
2. 回答结构：先一句话结论 → 关键数字 → 可能原因（明确标注"推断"）→ 1~3 条可执行建议。
3. 简洁口语化，用中文。百分比保留一位小数，ROAS 保留两位。
4. 问题和这份投放数据无关时，礼貌说明你只能分析这份数据。"""


@dataclass
class AgentResult:
    answer: str
    steps: list[dict] = field(default_factory=list)  # [{name, args, table}]


def ask(client, model: str, df: pd.DataFrame, question: str,
        history: list[dict] | None = None, max_rounds: int = 6) -> AgentResult:
    messages = [{"role": "system", "content": system_prompt(df)}]
    messages += (history or [])[-6:]
    messages.append({"role": "user", "content": question})
    steps: list[dict] = []

    for _ in range(max_rounds):
        resp = _chat(client, model=model, messages=messages, tools=TOOLS, temperature=0.2)
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            return AgentResult(msg.content or "", steps)

        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}}
                           for c in calls],
        })
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
                table = run_tool(df, c.function.name, args)
                content = json.dumps(an.to_records(table), ensure_ascii=False)
            except Exception as e:  # 工具出错时把错误告诉模型，让它自己修正参数
                args, table = {}, None
                content = json.dumps({"error": str(e)}, ensure_ascii=False)
            steps.append({"name": c.function.name, "args": args, "table": table})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": content})

    return AgentResult("这个问题需要的分析步骤太多了，换个更具体的问法试试？", steps)


# ---------------------------------------------------------------- 周报

REPORT_PROMPT = """下面是本周投放数据的统计结果（JSON，变化率 0.1 表示 +10%）。
请写一份给客户看的中文投放周报，Markdown 格式，包含：
1. 一句话总结
2. 核心指标（表格：本周 / 上周 / 环比）
3. 渠道表现（指出最好和最需要关注的渠道）
4. 素材表现
5. 风险与异常（逐条说明影响和处理建议）
6. 下周优化建议（3 条，具体到渠道或素材，可执行）

要求：所有数字只能取自下面的 JSON，不得编造或另行计算；语气专业但不啰嗦。

{stats}"""


def write_report(client, model: str, stats: dict) -> str:
    resp = _chat(client, model=model, temperature=0.3, messages=[
        {"role": "system", "content": "你是资深的效果广告优化师，负责给品牌客户写投放周报。"},
        {"role": "user", "content": REPORT_PROMPT.format(stats=json.dumps(stats, ensure_ascii=False, indent=1))},
    ])
    return resp.choices[0].message.content or ""


def _chat(client, **kwargs):
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        name = type(e).__name__
        if "Authentication" in name:
            raise LLMError("API key 无效，请检查是否复制完整、是否选对了服务商。") from e
        if "NotFound" in name:
            raise LLMError("找不到这个模型，请检查模型名称。") from e
        if "RateLimit" in name:
            raise LLMError("调用太频繁或余额不足，请稍后再试或检查账户余额。") from e
        if "Connection" in name or "Timeout" in name:
            raise LLMError("连不上大模型服务，请检查网络或接口地址。") from e
        raise LLMError(f"调用大模型出错：{e}") from e
