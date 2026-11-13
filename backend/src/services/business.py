"""包装节拍对照实验：异方差 Welch t 与均值差区间；全部班次口径下附两组合并占比加权的班次比较。"""
import math
import random
import numpy as np
from scipy import stats

MIN_CELL = 3  # 每个班次 × 每组最少独立片段数；不足即阻断，不剔除班次


def metadata():
    return {"name": "包装线节拍对照实验", "short": "pack-welch", "tagline": "把工艺调整的节拍差，放进可复查的实验区间。", "client": "榆川包装 · 52 人食品外包装车间，工艺组比较两种封装动作", "workflow": "工艺员用秒表登记每段节拍，把两列平均值截图交给生产经理。", "pains": ["忽略两组方差差异", "只报改善秒数没有不确定区间", "班次混杂未单独观察"], "upstream": "随机分配试运行片段 YC_pack_trial.csv", "downstream": "工艺经理决定是否继续扩大试验",
            "entities": ["实验片段", "节拍比较", "工艺评审简报"], "dataset_name": "A 原动作 / B 改动作 · 160 段观测",
            "columns": [{"key": "run", "label": "片段", "type": "string"}, {"key": "arm", "label": "实验组", "type": "string"}, {"key": "shift", "label": "班次", "type": "string"}, {"key": "seconds", "label": "单件节拍/秒", "type": "number"}],
            "parameters": [{"key": "confidence", "label": "区间置信度", "type": "select", "default": "0.95", "options": [{"value": str(v), "label": f"{v*100:g}%"} for v in [.9, .95, .99]]}, {"key": "shift", "label": "实验班次", "type": "select", "default": "全部", "options": [{"value": n, "label": n} for n in ["全部", "早班", "晚班"]]}],
            "workbench_title": "节拍效果核验", "action_label": "执行 Welch 比较", "unit": "秒", "accent": "oklch(58% 0.16 69)", "style": "industrial",
            "seed_story": ["08:20 工艺员导入 YC_pack_trial.csv，A/B 各 80 段。", "08:30 核查片段按班次交错随机分配。", "08:40 分析员选择 95% 置信度和全部班次。", "08:45 执行 Welch 检验，查看 A−B 改善秒数。", "08:50 对比总体与班次加权口径，查看各班贡献。", "09:00 检查置信区间是否跨 0。", "09:20 改为晚班重算，查看异方差影响。", "09:45 导出工艺评审简报，交经理安排扩大试验。"],
            "method": "改善量=均值(A)−均值(B)；SE=√(sA²/nA+sB²/nB)；Welch–Satterthwaite 自由度；双侧 t 检验和 t 分布置信区间；Cohen d 用合并标准差。全部班次口径下另报班次加权改善量：权重 w=两组合并班次占比，Δ=Σ w·(均值A−均值B)，SE=√(Σ w²·SE²)，自由度按 Welch–Satterthwaite 合成；任一班缺组或样本不足即阻断，不剔除班次。",
            "limitations": ["假定观测片段近似独立", "未校正多个班次的探索性比较", "显著节拍差不能证明成品质检合格", "加权口径以两组合并班次占比为固定权重，不代表车间真实班次负荷"]}


def seed_records():
    rng = random.Random(1004)
    rows = []
    for arm, mu, sigma in [("A", 19.5, 1.4), ("B", 18.2, 2.1)]:
        for i in range(80):
            shift = "早班" if i % 2 == 0 else "晚班"
            rows.append({"run": f"YC-{arm}{i+1:03}", "arm": arm, "shift": shift, "seconds": round(max(1, rng.gauss(mu + (.8 if shift == "晚班" else 0), sigma)), 3)})
    return rows


def _welch(a, b):
    """两组 Welch 汇总：均值、样本方差、均值差方差、Welch–Satterthwaite 自由度。两组均无变异时自由度记 0，由调用方按口径阻断。"""
    na, nb = len(a), len(b)
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se2 = va / na + vb / nb
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = se2 * se2 / den if den > 0 else 0.0
    return ma, mb, va, vb, se2, df


def _weighted(obs, confidence):
    """两组合并班次占比加权的改善量与各班贡献。任一班缺组或样本不足即抛出阻断错误，不静默剔除。"""
    strata = sorted({s for s, _, _ in obs})
    total = len(obs)
    rows, weights, raw = [], {}, []
    diff_w = var_w = denom_df = 0.0
    for s in strata:
        a = [x for sh, arm, x in obs if sh == s and arm == "A"]
        b = [x for sh, arm, x in obs if sh == s and arm == "B"]
        if not a or not b:
            raise ValueError(f"{s} 缺少 {'A' if not a else 'B'} 组样本，班次加权比较已阻断；请补齐迟到记录后新建数据批次重算，已完成分析与已分发简报仍保留当时样本、权重和结果")
        if min(len(a), len(b)) < MIN_CELL:
            raise ValueError(f"{s} 样本不足：A 组 {len(a)} 段、B 组 {len(b)} 段，每班每组至少 {MIN_CELL} 段；班次加权比较已阻断，不剔除班次。请补齐迟到记录后新建数据批次重算")
        ma, mb, va, vb, se2, df = _welch(a, b)
        if se2 <= 1e-15:
            raise ValueError(f"{s} 两组节拍均无变异，无法估计该班误差，班次加权比较已阻断；请核查该班记录后新建数据批次重算")
        w = (len(a) + len(b)) / total
        d = ma - mb
        diff_w += w * d
        var_w += w * w * se2
        denom_df += (w * w * se2) ** 2 / df
        weights[s] = round(w, 6)
        raw.append((s, w, d))
        rows.append({"shift": s, "n_a": len(a), "n_b": len(b), "weight": round(w, 4), "diff": round(d, 4), "contribution": round(w * d, 4)})
    se_w = math.sqrt(var_w)
    df_w = var_w * var_w / denom_df
    statistic = diff_w / se_w
    p = float(2 * stats.t.sf(abs(statistic), df_w))
    margin = float(stats.t.ppf((1 + confidence) / 2, df_w)) * se_w
    low, high = diff_w - margin, diff_w + margin
    share = "、".join(f"{s} {w*100:.1f}%" for s, w, _ in raw)
    dist = {}
    for arm in ("A", "B"):
        n = sum(1 for _, g, _ in obs if g == arm)
        dist[arm] = "、".join(f"{s} {sum(1 for sh, g, _ in obs if sh == s and g == arm)/n*100:.1f}%" for s in strata)
    return {"summary": f"按两组合并班次占比加权后，B 相对 A 平均节拍缩短 {diff_w:.3f} 秒；{confidence*100:g}% 区间为 [{low:.3f}, {high:.3f}] 秒。",
            "metrics": [{"label": "加权改善量 A−B", "value": round(diff_w, 5), "unit": "秒"}, {"label": f"{confidence*100:g}% 区间下界", "value": round(low, 5), "unit": "秒"}, {"label": f"{confidence*100:g}% 区间上界", "value": round(high, 5), "unit": "秒"}, {"label": "双侧 p 值（加权）", "value": float(f"{p:.6g}")}, {"label": "加权自由度", "value": round(df_w, 3)}],
            "table": {"columns": [{"key": "shift", "label": "班次"}, {"key": "n_a", "label": "A 片段数"}, {"key": "n_b", "label": "B 片段数"}, {"key": "weight", "label": "合并占比"}, {"key": "diff", "label": "班内改善 A−B/秒"}, {"key": "contribution", "label": "加权贡献/秒"}], "rows": rows},
            "chart": {"type": "bar", "labels": [s for s, _, _ in raw], "series": [{"name": "班内改善量 A−B/秒", "values": [round(d, 4) for _, _, d in raw]}, {"name": "加权贡献/秒", "values": [round(w * d, 4) for _, w, d in raw]}]},
            "findings": [f"班次权重取两组合并占比：{share}；各班加权贡献之和等于加权改善量。", f"A 组班次分布：{dist['A']}；B 组班次分布：{dist['B']}。", f"加权 t={statistic:.4f}，合成自由度={df_w:.3f}；区间下界 {low:.4f} 秒，上界 {high:.4f} 秒。"],
            "weights": weights, "diff": round(diff_w, 6), "se": round(se_w, 6), "df": round(df_w, 4), "low": round(low, 6), "high": round(high, 6),
            "rule": f"权重=两组合并班次占比；每班每组至少 {MIN_CELL} 段独立片段，缺组或不足即阻断，不剔除班次。本快照保留当时样本与权重，迟到记录请新建批次重算。"}


def analyze(records, params):
    try:
        confidence = float(params.get("confidence", .95))
    except (TypeError, ValueError):
        raise ValueError("置信度只接受 0.90/0.95/0.99")
    if confidence not in (.9, .95, .99):
        raise ValueError("置信度只接受 0.90/0.95/0.99")
    shift = str(params.get("shift", "全部"))
    if shift not in ("全部", "早班", "晚班"):
        raise ValueError("未知班次")
    groups = {"A": [], "B": []}
    obs = []
    seen = set()
    for r in records:
        try:
            arm, run = str(r["arm"]), str(r["run"])
            seconds = float(r["seconds"])
            if arm not in groups or not math.isfinite(seconds) or seconds <= 0 or run in seen:
                raise ValueError()
            seen.add(run)
            obs.append((str(r["shift"]), arm, seconds))
            if shift == "全部" or r["shift"] == shift:
                groups[arm].append(seconds)
        except (KeyError, TypeError, ValueError):
            raise ValueError("片段号须唯一，组别为 A/B，节拍必须为正有限数")
    if min(map(len, groups.values())) < 3:
        raise ValueError("每组至少需要 3 个独立实验片段")
    a, b = groups["A"], groups["B"]
    na, nb = len(a), len(b)
    ma, mb, va, vb, se2, df = _welch(a, b)
    if se2 <= 1e-15:
        raise ValueError("两组均无变异，Welch 检验无法估计误差")
    diff, se = ma - mb, math.sqrt(se2)
    statistic = diff / se
    p = float(2 * stats.t.sf(abs(statistic), df))
    margin = float(stats.t.ppf((1 + confidence) / 2, df)) * se
    low, high = diff - margin, diff + margin
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    effect = diff / pooled
    rows = [{"arm": k, "n": len(groups[k]), "mean": round(float(np.mean(groups[k])), 4), "sd": round(float(np.std(groups[k], ddof=1)), 4)} for k in ("A", "B")]
    result = {"summary": f"B 相对 A 平均节拍缩短 {diff:.3f} 秒；{confidence*100:g}% 区间为 [{low:.3f}, {high:.3f}] 秒。",
            "metrics": [{"label": "改善量 A−B", "value": round(diff, 5), "unit": "秒"}, {"label": "双侧 p 值", "value": float(f"{p:.6g}")}, {"label": "Cohen d", "value": round(effect, 5)}, {"label": "Welch 自由度", "value": round(df, 3)}],
            "table": {"columns": [{"key": "arm", "label": "实验组"}, {"key": "n", "label": "片段数"}, {"key": "mean", "label": "均值/秒"}, {"key": "sd", "label": "样本标准差/秒"}], "rows": rows},
            "chart": {"type": "bar", "labels": ["A 原动作", "B 改动作"], "series": [{"name": "平均单件节拍/秒", "values": [round(ma, 4), round(mb, 4)]}]},
            "findings": [f"区间下界 {low:.4f} 秒，上界 {high:.4f} 秒；{'区间未跨 0' if low > 0 or high < 0 else '区间跨 0'}。", f"t={statistic:.4f}，自由度={df:.3f}；分析班次：{shift}。", "仅在分组随机、片段独立的前提下支持实验推断；扩大试验仍须核查质检指标。"], "method": metadata()["method"], "warnings": ["班次子组比较为探索性分析，未调整多重检验。"]}
    if shift == "全部":
        w = _weighted(obs, confidence)
        result["weighted"] = w
        result["findings"].append(f"班次加权改善量 {w['diff']:.4f} 秒，{confidence*100:g}% 区间 [{w['low']:.4f}, {w['high']:.4f}] 秒；各班贡献见班次加权比较。")
        if (diff > 0) != (w["diff"] > 0) and abs(diff) > 1e-12 and abs(w["diff"]) > 1e-12:
            result["warnings"].append("班次占比不均已使总体改善量与加权改善量方向相反（辛普森悖论），请以加权口径为准。")
            w["findings"].append("总体改善量与加权改善量方向相反：两班占比不均使总体比较反向，请以加权口径为准。")
        else:
            w["findings"].append("总体与加权口径方向一致。")
    return result
