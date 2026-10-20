# -*- coding: utf-8 -*-
"""train.jsonl 探索性数据分析：高频词 / 标签分布 / 句子长度分布
统计口径：
  1) 句子长度 = 去除首尾空白后的字符数（含标点与英文，不做分词）
  2) 分词 = jieba 精确模式；去除停用词、纯数字/英文/符号、单字词
"""
import json
import os
import re
import statistics
from collections import Counter

import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "train.jsonl")
OUT = os.path.join(BASE, "eda_output")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#E8E8E8"
plt.rcParams["grid.linewidth"] = 0.8
plt.rcParams["axes.axisbelow"] = True

C_MAIN = "#3B6EA5"
C_ACC = "#E07B54"
PALETTE = ["#3B6EA5", "#E07B54", "#5FA87C", "#B07AA1", "#E2B33C"]

# ---------------------------------------------------------------- 1. 读数据
records = []
with open(SRC, encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[跳过] 第 {line_no} 行 JSON 解析失败")

for r in records:
    r["text"] = (r.get("text") or "").strip()
    r["length"] = len(r["text"])

n = len(records)
empty_cnt = sum(1 for r in records if r["length"] == 0)
print(f"有效样本数: {n}；空文本: {empty_cnt}")

# ---------------------------------------------------------------- 2. 高频词
STOPWORDS = set("""
的 了 是 我 你 他 她 它 我们 你们 他们 她们 它们 自己 这 那 这个 那个 这些 那些 这样 那样 这种 那种
就 都 也 还 又 再 才 只 很 太 更 最 挺 非常 特别 十分 有点 一些 一点 一个 一种 一样 一般 一直 一定
不 没 没有 别 不要 不是 不能 不会 不用 不要 不过 不管 不仅 不但 而且 并且 但是 但 可是 然而 因为 所以 因此
如果 要是 虽然 即使 只要 只有 无论 除了 关于 对于 由于 至于 以及 及其 还有 或者 还是 而是 然后 接着 于是
在 有 和 与 跟 同 及 或 把 被 让 给 对 从 到 向 往 于 为 以 按 依 据 通过 根据 针对
上 下 里 外 中 内 前 后 左 右 边 面 之间 之中 之后 之前 以前 以后 现在 目前 时候 时间 今天 明天 昨天
个 们 些 种 样 点 次 回 条 件 位 名 岁 年 月 日 天 分 秒
吗 呢 吧 啊 呀 哦 嗯 哈 哈哈 呢 么 嘛 啦 喽 哟 哇 哎 唉 呵 呵呵 嘿 喂 呗
什么 怎么 怎样 为什么 哪 哪个 哪里 哪些 谁 多少 几 如何 是否 可以 可能 应该 需要 觉得 认为 知道 说 讲 想 要 会 能 去 来 做 用 看 走 站 帮
自己 人家 大家 别人 有人 没人 东西 事情 问题 方法 方面 情况 地方 样子 感觉 觉得
真是 真的 确实 其实 反正 到底 究竟 居然 竟然 简直 几乎 大概 也许 好像 似乎 估计 差不多
一 二 三 四 五 六 七 八 九 十 百 千 万 亿 两 第一 第二
""".split())

TOKEN_RE = re.compile(r"[\u4e00-\u9fa5]+")


def tokenize(text):
    for w in jieba.cut(text):
        w = w.strip()
        if len(w) < 2:
            continue
        if not TOKEN_RE.fullmatch(w):
            continue
        if w in STOPWORDS:
            continue
        yield w


word_counter = Counter()
label_word_counter = {k: Counter() for k in ("hate", "offensive", "normal")}
for r in records:
    toks = list(tokenize(r["text"]))
    word_counter.update(toks)
    if r["label"] in label_word_counter:
        label_word_counter[r["label"]].update(toks)

total_tokens = sum(word_counter.values())
uniq_tokens = len(word_counter)
top30 = word_counter.most_common(30)

print(f"有效词总数: {total_tokens}；词表规模: {uniq_tokens}")
print("Top30 高频词:", top30)

# 图1：高频词条形图
fig, ax = plt.subplots(figsize=(10, 9))
words = [w for w, _ in top30][::-1]
counts = [c for _, c in top30][::-1]
colors = [C_MAIN if i < 10 else C_MAIN + "" for i in range(len(words))]
bars = ax.barh(words, counts, color=C_MAIN, alpha=0.9, height=0.72, edgecolor="none")
for i, b in enumerate(bars):
    if i >= len(words) - 10:
        b.set_color(C_ACC)
    ax.text(b.get_width() + max(counts) * 0.008, b.get_y() + b.get_height() / 2,
            f"{int(b.get_width())}", va="center", fontsize=9, color="#444444")
ax.set_xlabel("出现次数", fontsize=11)
ax.set_ylabel("词语", fontsize=11)
ax.set_title(f"图1  训练集高频词 Top 30\n（jieba 分词，去停用词与单字；有效词总数 {total_tokens:,}）",
             fontsize=14, pad=14, fontweight="bold")
ax.set_xlim(0, max(counts) * 1.12)
ax.grid(axis="y", visible=False)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[Patch(color=C_ACC, label="Top 10 高频词"), Patch(color=C_MAIN, label="其余 Top 30")],
          loc="lower right", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "01_高频词Top30.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 3. 标签分布
label_order = ["hate", "offensive", "normal"]
label_cn = {"hate": "仇恨 (hate)", "offensive": "冒犯 (offensive)", "normal": "正常 (normal)"}
label_cnt = Counter(r["label"] for r in records)
vals = [label_cnt.get(k, 0) for k in label_order]
pcts = [v / n * 100 for v in vals]
print("标签分布:", {k: label_cnt.get(k, 0) for k in label_order}, [f"{p:.2f}%" for p in pcts])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
bars = ax.bar([label_cn[k] for k in label_order], vals, color=PALETTE[:3], width=0.58, edgecolor="none")
for b, v, p in zip(bars, vals, pcts):
    ax.text(b.get_x() + b.get_width() / 2, v + n * 0.008, f"{v:,}\n({p:.2f}%)",
            ha="center", va="bottom", fontsize=10.5, color="#333333")
ax.set_ylabel("样本数", fontsize=11)
ax.set_ylim(0, max(vals) * 1.25)
ax.set_title("各类别样本数与占比", fontsize=12, fontweight="bold")
ax.grid(axis="x", visible=False)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
wedges, _, autotexts = ax.pie(
    vals, labels=None, colors=PALETTE[:3], autopct=lambda p: f"{p:.2f}%",
    startangle=90, counterclock=False, pctdistance=0.75,
    wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 11, "color": "white", "fontweight": "bold"})
ax.legend(wedges, [f"{label_cn[k]}  {label_cnt.get(k, 0):,}" for k in label_order],
          loc="center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False, fontsize=10)
ax.set_title("类别构成环形图", fontsize=12, fontweight="bold")

fig.suptitle(f"图2  标签分布（n = {n:,}）", fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02_标签分布.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 4. 元数据标签分布（平台 / 主题）
platform_cnt = Counter((r.get("metadata") or {}).get("platform", "未知") for r in records)
topic_cnt = Counter((r.get("metadata") or {}).get("topic", "未知") for r in records)
print("平台分布:", dict(platform_cnt))
print("主题分布:", dict(topic_cnt))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

ax = axes[0]
plats = [p for p, _ in platform_cnt.most_common()]
bottom = np.zeros(len(plats))
for i, k in enumerate(label_order):
    heights = np.array([
        sum(1 for r in records
            if (r.get("metadata") or {}).get("platform") == p and r["label"] == k)
        for p in plats
    ])
    ax.bar(plats, heights, bottom=bottom, color=PALETTE[i], label=label_cn[k], width=0.5)
    for x, h, b in zip(range(len(plats)), heights, bottom):
        if h > n * 0.01:
            ax.text(x, b + h / 2, f"{h:,}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    bottom += heights
ax.set_ylabel("样本数", fontsize=11)
ax.set_title("平台 × 标签 堆叠分布", fontsize=12, fontweight="bold")
ax.grid(axis="x", visible=False)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
tops = [t for t, _ in topic_cnt.most_common()]
bottom = np.zeros(len(tops))
for i, k in enumerate(label_order):
    heights = np.array([
        sum(1 for r in records
            if (r.get("metadata") or {}).get("topic") == t and r["label"] == k)
        for t in tops
    ])
    ax.bar(tops, heights, bottom=bottom, color=PALETTE[i], label=label_cn[k], width=0.5)
    for x, h, b in zip(range(len(tops)), heights, bottom):
        if h > n * 0.01:
            ax.text(x, b + h / 2, f"{h:,}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    bottom += heights
ax.set_ylabel("样本数", fontsize=11)
ax.set_title("主题 × 标签 堆叠分布", fontsize=12, fontweight="bold")
ax.grid(axis="x", visible=False)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9)

fig.suptitle("图3  元数据维度分布（platform / topic）", fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "03_平台主题分布.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 5. 句子长度
lengths = np.array([r["length"] for r in records], dtype=int)
mean_len = float(np.mean(lengths))
median_len = float(np.median(lengths))
std_len = float(np.std(lengths, ddof=1))
p90, p95, p99 = np.percentile(lengths, [90, 95, 99])
print(f"句子长度: mean={mean_len:.2f} median={median_len:.1f} std={std_len:.2f} "
      f"min={lengths.min()} max={lengths.max()} P90={p90:.0f} P95={p95:.0f} P99={p99:.0f}")

BINS = [0, 10, 20, 30, 40, 50, 60, 80, 100, float("inf")]
BIN_LABELS = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80-100", "100+"]
seg_idx = np.digitize(lengths, BINS[1:-1], right=False)
seg_cnt = [int((seg_idx == i).sum()) for i in range(len(BIN_LABELS))]
seg_pct = [c / n * 100 for c in seg_cnt]
for lb, c, p in zip(BIN_LABELS, seg_cnt, seg_pct):
    print(f"  {lb:>7} 字: {c:>5} 条 ({p:5.2f}%)")

# 图4：分段计数
fig = plt.figure(figsize=(13.5, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1], wspace=0.18)
ax = fig.add_subplot(gs[0, 0])
bars = ax.bar(BIN_LABELS, seg_cnt, color=C_MAIN, width=0.62, edgecolor="none")
bars[max(range(len(seg_cnt)), key=lambda i: seg_cnt[i])].set_color(C_ACC)
for b, c, p in zip(bars, seg_cnt, seg_pct):
    ax.text(b.get_x() + b.get_width() / 2, c + max(seg_cnt) * 0.015, f"{c:,}\n{p:.1f}%",
            ha="center", va="bottom", fontsize=9, color="#333333")
ax.set_xlabel("句子长度区间（字符数）", fontsize=11)
ax.set_ylabel("样本数", fontsize=11)
ax.set_ylim(0, max(seg_cnt) * 1.25)
ax.set_title("句子长度分段计数", fontsize=12, fontweight="bold")
ax.grid(axis="x", visible=False)
ax.spines[["top", "right"]].set_visible(False)

ax2 = fig.add_subplot(gs[0, 1])
ax2.axis("off")
ax2.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax2.transAxes,
                            facecolor="#F6F8FB", edgecolor="#D8E0EA", lw=1.2, zorder=0))
rows = [
    ("样本总数", f"{n:,}"),
    ("平均长度", f"{mean_len:.2f} 字"),
    ("中位长度", f"{median_len:.1f} 字"),
    ("标准差", f"{std_len:.2f}"),
    ("最短 / 最长", f"{lengths.min()} / {lengths.max()} 字"),
    ("P90/95/99", f"{p90:.0f} / {p95:.0f} / {p99:.0f}"),
    ("均值 vs 中位数", "右偏（长尾）" if mean_len > median_len else "左偏"),
]
ax2.text(0.5, 0.93, "核心统计指标", ha="center", va="center",
         fontsize=13, fontweight="bold", color="#2F4A6B", transform=ax2.transAxes)
for i, (k, v) in enumerate(rows):
    y = 0.81 - i * 0.108
    ax2.text(0.07, y, k, fontsize=10.5, color="#5A6B80", transform=ax2.transAxes)
    ax2.text(0.93, y, v, fontsize=10.5, color="#1F3B57", ha="right",
             fontweight="bold", transform=ax2.transAxes)

fig.suptitle(f"图4  句子长度分段分布与统计指标（按字符数计，n = {n:,}）",
             fontsize=14, fontweight="bold", y=0.975)
fig.subplots_adjust(left=0.065, right=0.975, top=0.87, bottom=0.13, wspace=0.16)
fig.savefig(os.path.join(OUT, "04_句子长度分段分布.png"), dpi=160)
plt.close(fig)

# 图5：直方图 + 各标签箱线图
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
ax.hist(lengths, bins=60, color=C_MAIN, alpha=0.88, edgecolor="white", linewidth=0.4)
ax.axvline(mean_len, color=C_ACC, lw=2, label=f"平均数 {mean_len:.2f}")
ax.axvline(median_len, color="#5FA87C", lw=2, ls="--", label=f"中位数 {median_len:.1f}")
ax.set_xlabel("句子长度（字符数）", fontsize=11)
ax.set_ylabel("频数", fontsize=11)
ax.set_title("句子长度直方图（bins=60）", fontsize=12, fontweight="bold")
ax.legend(frameon=False, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
data_by_label = [[r["length"] for r in records if r["label"] == k] for k in label_order]
bp = ax.boxplot(data_by_label, labels=[f"{k}\nn={len(d)}" for k, d in zip(label_order, data_by_label)],
                patch_artist=True, widths=0.55, showmeans=True, meanline=True,
                medianprops={"color": "#1F3B57", "lw": 1.8},
                meanprops={"color": C_ACC, "ls": "--", "lw": 1.4},
                flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.35,
                            "markerfacecolor": "#888888", "markeredgecolor": "none"})
for patch, c in zip(bp["boxes"], PALETTE[:3]):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
    patch.set_edgecolor("none")
ax.set_ylim(0, max(max(d) for d in data_by_label) * 1.18)
for i, d in enumerate(data_by_label, 1):
    ax.text(i, max(max(dd) for dd in data_by_label) * 1.08,
            f"均值 {np.mean(d):.1f}\n中位数 {np.median(d):.0f}",
            ha="center", va="top", fontsize=8.5, color="#333333",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "#DDDDDD", "alpha": 0.9})
ax.set_ylabel("句子长度（字符数）", fontsize=11)
ax.set_title("各标签句子长度箱线图（实线=中位数，虚线=平均数，圆点=离群值）",
             fontsize=11, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("图5  句子长度分布形态与分标签对比", fontsize=14, fontweight="bold", y=1.0)
fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.1, wspace=0.18)
fig.savefig(os.path.join(OUT, "05_句子长度分布形态.png"), dpi=160)
plt.close(fig)

# 图6：各标签 Top10 高频词对比
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
for ax, k, c in zip(axes, label_order, PALETTE[:3]):
    top10 = label_word_counter[k].most_common(10)[::-1]
    ws = [w for w, _ in top10]
    cs = [x for _, x in top10]
    ax.barh(ws, cs, color=c, alpha=0.88, height=0.65, edgecolor="none")
    for y, x in enumerate(cs):
        ax.text(x + max(cs) * 0.02, y, f"{x}", va="center", fontsize=8.5, color="#444444")
    ax.set_xlim(0, max(cs) * 1.18)
    ax.set_title(f"{label_cn[k]}\n(n={label_cnt.get(k, 0):,})", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("出现次数", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("图6  各标签 Top 10 高频词对比（同一分词与停用词口径）",
             fontsize=14, fontweight="bold", y=0.99)
fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.12, wspace=0.42)
fig.savefig(os.path.join(OUT, "06_各标签高频词.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 6. 各标签高频词（供解读）
print("\n各标签 Top 10 高频词：")
for k in label_order:
    print(f"  [{k}]", label_word_counter[k].most_common(10))

# ---------------------------------------------------------------- 7. 文本长度与标签的统计表
print("\n各标签长度统计：")
for k, d in zip(label_order, data_by_label):
    print(f"  {k:>9}: n={len(d):>4} mean={np.mean(d):6.2f} median={np.median(d):5.1f} "
          f"P90={np.percentile(d, 90):5.0f} max={max(d)}")

print("\n输出目录:", OUT)
for f in sorted(os.listdir(OUT)):
    print("  -", f)
