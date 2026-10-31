"""生成模拟的跨渠道广告投放日级数据（8 周）。

数据是虚构的，但刻意埋了几个真实业务里常见的问题，方便演示异常检测和 AI 问答：
1. TikTok 素材 ugc_unboxing 出现素材疲劳：CTR 随时间持续下滑
2. Meta 再营销计划 10/14-10/15 两天转化几乎为 0，但点击正常（像是像素/追踪断了）
3. Google 泛词搜索 10/05 那一周 CPC 大涨（竞品抬价），CPA 跟着飙升
4. YouTube 品牌曝光计划整体 ROAS 偏低，适合讨论预算重新分配
5. TikTok KOL 合作素材 10/12 上线，效果很好

用法：python scripts/generate_data.py  -> 写出 data/sample_ad_data.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = "2026-08-31"  # 周一
END = "2026-10-25"    # 周日，正好 8 周

# channel, campaign, market, creative, daily_spend, cpm, ctr, cvr, aov, start_date
CREATIVES = [
    ("Meta", "Prospecting_US", "US", "video_A", 900, 11.0, 0.0120, 0.030, 68, None),
    ("Meta", "Prospecting_US", "US", "carousel_B", 600, 10.0, 0.0105, 0.028, 66, None),
    ("Meta", "Retargeting_US", "US", "static_C", 350, 18.0, 0.0180, 0.065, 72, None),
    ("Google Search", "Brand_Search", "US", "rsa_brand", 400, 45.0, 0.0800, 0.090, 70, None),
    ("Google Search", "Generic_Search", "US", "rsa_generic", 700, 30.0, 0.0350, 0.032, 64, None),
    ("TikTok", "Spark_Ads_US", "US", "ugc_unboxing", 800, 7.0, 0.0220, 0.022, 58, None),
    ("TikTok", "Spark_Ads_US", "US", "ugc_review", 500, 7.5, 0.0160, 0.024, 58, None),
    ("TikTok", "Spark_Ads_SEA", "SEA", "kol_collab", 600, 4.0, 0.0240, 0.030, 45, "2026-10-12"),
    ("TikTok", "Spark_Ads_SEA", "SEA", "ugc_local", 300, 3.5, 0.0150, 0.018, 42, None),
    ("YouTube", "Awareness_Global", "Global", "bumper_6s", 450, 6.0, 0.0040, 0.012, 65, None),
    ("YouTube", "Awareness_Global", "Global", "instream_30s", 550, 9.0, 0.0060, 0.015, 65, None),
]

# 周末效应：消费品电商周末转化更好一点
WEEKDAY_CVR = {0: 0.95, 1: 0.95, 2: 0.97, 3: 1.0, 4: 1.03, 5: 1.08, 6: 1.07}


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    rows = []

    for channel, campaign, market, creative, spend0, cpm0, ctr0, cvr0, aov0, start in CREATIVES:
        for i, d in enumerate(dates):
            if start and d < pd.Timestamp(start):
                continue

            spend = spend0 * rng.normal(1.0, 0.08)
            cpm = cpm0 * rng.normal(1.0, 0.05)
            ctr = ctr0 * rng.normal(1.0, 0.06)
            cvr = cvr0 * rng.normal(1.0, 0.08) * WEEKDAY_CVR[d.weekday()]
            aov = aov0 * rng.normal(1.0, 0.04)

            # 异常 1：素材疲劳，CTR 指数衰减到原来约 40%
            if creative == "ugc_unboxing":
                ctr *= 0.4 + 0.6 * np.exp(-i / 22)

            # 异常 2：追踪中断，转化几乎为 0
            if campaign == "Retargeting_US" and d in (pd.Timestamp("2026-10-14"), pd.Timestamp("2026-10-15")):
                cvr *= 0.03

            # 异常 3：竞品抬价，CPM（进而 CPC）上涨约 65%，预算没动所以点击变少
            if campaign == "Generic_Search" and pd.Timestamp("2026-10-05") <= d <= pd.Timestamp("2026-10-11"):
                cpm *= 1.65

            spend = round(max(spend, 0), 2)
            impressions = int(spend / cpm * 1000)
            clicks = int(rng.binomial(impressions, min(max(ctr, 0), 1)))
            conversions = int(rng.binomial(clicks, min(max(cvr, 0), 1)))
            revenue = round(conversions * aov, 2)

            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "channel": channel,
                "campaign": campaign,
                "market": market,
                "creative": creative,
                "spend": spend,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "revenue": revenue,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "sample_ad_data.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")
