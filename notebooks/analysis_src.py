# %% [code]
!pip3 install pandas
!pip3 install numpy
!pip3 install scipy
!pip3 install statsmodels
!pip3 install scikit-learn
!pip3 install matplotlib
!pip3 install seaborn
!pip3 install rich
!pip3 install python-dotenv

# %% [code]
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from random import choice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from dotenv import load_dotenv
from rich import print as rprint
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %% [code]
%reload_ext autoreload
%autoreload 2
load_dotenv()


@dataclass
class Config:
    seed: int = int(os.getenv("SEED", "42"))
    n_boot: int = int(os.getenv("N_BOOT", "2000"))
    alpha: float = float(os.getenv("ALPHA", "0.05"))
    base_chunk: int = int(os.getenv("BASE_CHUNK", "100"))
    base_k: int = int(os.getenv("BASE_K", "5"))
    base_retriever: str = os.getenv("BASE_RETRIEVER", "hybrid")


CFG = Config()

# %% [code]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analysis")
warnings.filterwarnings("ignore")

# ----------
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_LOGS = PROJECT_ROOT / "data" / "logs"
OUT_FIG = PROJECT_ROOT / "report" / "figures"
OUT_TAB = PROJECT_ROOT / "report" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)
# ----------
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams["figure.dpi"] = 120
RNG = np.random.default_rng(CFG.seed)
DATASETS = ["hotpotqa", "squad"]
RETRIEVERS = ["bm25", "dense", "hybrid"]
DATASET_RU = {"hotpotqa": "HotpotQA", "squad": "SQuAD"}
RETRIEVER_RU = {"bm25": "BM25", "dense": "плотный", "hybrid": "гибрид"}
METRIC_RU = {"recall_at_k": "полнота поиска", "f1": "F1 ответа", "recall": "полнота поиска"}
RESULTS = {}


def save_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Сохранить таблицу для отчета (csv) и вернуть ее."""
    df.to_csv(OUT_TAB / f"{name}.csv")
    return df


def save_fig(fig, name: str):
    """Сохранить рисунок для отчета (pdf + png)."""
    fig.savefig(OUT_FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_FIG / f"{name}.png", bbox_inches="tight", dpi=150)


def boot_ci(x: np.ndarray, n_boot: int = CFG.n_boot) -> tuple[float, float]:
    """
    Перцентильный бутстреп-интервал 95% для среднего.

    :param x: выборка
    :param n_boot: число повторов
    :return: (нижняя, верхняя) граница
    """
    x = np.asarray(x, dtype=float)
    means = RNG.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# %% [markdown]
# # 1. Загрузка логов

# %% [code]
answers = pd.read_parquet(DATA_LOGS / "answer_log.parquet")
search = pd.read_parquet(DATA_LOGS / "search_log.parquet")
iters = pd.read_parquet(DATA_LOGS / "iteration_log.parquet")
questions = {d: pd.read_parquet(DATA_LOGS / f"questions_{d}.parquet") for d in DATASETS}

rprint(answers.shape, search.shape, iters.shape)
rprint(answers.head(3).T)

# %% [code]
quess = answers.sample(1, random_state=CFG.seed).iloc[0]
rprint(quess[["dataset", "retriever", "chunk_size", "top_k", "gold", "prediction", "f1", "recall_at_k"]])

# %% [markdown]
# # 2. Описательная статистика

# %% [code]
# === 1) Выборки и корпуса ===
corpus = (
    answers.groupby(["dataset", "chunk_size"])["n_chunks_corpus"].first().unstack()
    .rename(columns=lambda c: f"chunks_{c}")
)
qstat = pd.concat(
    {
        d: questions[d][["q_len", "answer_len", "lex_overlap"]].describe().T[["mean", "std", "50%", "min", "max"]]
        for d in DATASETS
    }
)
rprint(corpus)
rprint(qstat.round(2))
save_table(corpus, "corpus")
save_table(qstat.round(3), "question_stats")
RESULTS["n_questions"] = {d: int(len(questions[d])) for d in DATASETS}
RESULTS["level_counts"] = questions["hotpotqa"]["level"].value_counts().to_dict()
RESULTS["type_counts"] = questions["hotpotqa"]["q_type"].value_counts().to_dict()

# %% [code]
# === 2) Метрики по конфигурациям ===
METRICS = ["hit_at_k", "recall_at_k", "mrr_at_k", "answer_in_context", "em", "f1", "correct"]
by_cfg = answers.groupby(["dataset", "retriever", "chunk_size", "top_k"])[METRICS].mean()
save_table(by_cfg.round(4), "metrics_by_config")
rprint(by_cfg.round(3).head(24))

base = answers[(answers.chunk_size == CFG.base_chunk) & (answers.top_k == CFG.base_k)]
desc_rows = []
for d in DATASETS:
    for r in RETRIEVERS:
        part = base[(base.dataset == d) & (base.retriever == r)]
        row = {"dataset": d, "retriever": r}
        for m in ["recall_at_k", "f1", "correct"]:
            lo, hi = boot_ci(part[m].values)
            row[m] = part[m].mean()
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
        row["f1_median"] = part["f1"].median()
        row["f1_std"] = part["f1"].std()
        desc_rows.append(row)
desc_base = pd.DataFrame(desc_rows).set_index(["dataset", "retriever"])
save_table(desc_base.round(4), "desc_base")
rprint(desc_base.round(3))

# %% [code]
# === 3) Распределения ===
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
plot_base = base[base.retriever == CFG.base_retriever].assign(Датасет=lambda x: x.dataset.map(DATASET_RU))
plot_q = pd.concat([questions[d].assign(Датасет=DATASET_RU[d]) for d in DATASETS])
sns.histplot(data=plot_base, x="f1", hue="Датасет", bins=20, ax=axes[0], stat="probability", common_norm=False)
axes[0].set_title("F1 ответа (гибрид, чанк 100, top-5)")
axes[0].set_xlabel("F1 ответа")
sns.histplot(data=plot_q, x="q_len", hue="Датасет", bins=30, ax=axes[1], stat="probability", common_norm=False)
axes[1].set_title("Длина вопроса")
axes[1].set_xlabel("слов в вопросе")
sns.histplot(data=plot_q, x="lex_overlap", hue="Датасет", bins=20, ax=axes[2], stat="probability", common_norm=False)
axes[2].set_title("Пересечение вопроса с эталонным документом")
axes[2].set_xlabel("доля слов вопроса в документе")
for ax in axes:
    ax.set_ylabel("доля вопросов")
fig.tight_layout()
save_fig(fig, "distributions")

f1_shape = {
    d: {
        "share_f1_zero": float((base[(base.dataset == d) & (base.retriever == CFG.base_retriever)].f1 == 0).mean()),
        "share_f1_one": float((base[(base.dataset == d) & (base.retriever == CFG.base_retriever)].f1 == 1).mean()),
        "shapiro_p": float(stats.shapiro(base[(base.dataset == d) & (base.retriever == CFG.base_retriever)].f1.values).pvalue),
    }
    for d in DATASETS
}
RESULTS["f1_shape"] = f1_shape
rprint(f1_shape)

# %% [code]
# === 4) Кривые по top-k ===
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for i, d in enumerate(DATASETS):
    part = answers[(answers.dataset == d) & (answers.chunk_size == CFG.base_chunk)]
    for j, m in enumerate(["recall_at_k", "f1"]):
        sns.pointplot(data=part.assign(Ретривер=part.retriever.map(RETRIEVER_RU)), x="top_k", y=m, hue="Ретривер",
                      hue_order=[RETRIEVER_RU[r] for r in RETRIEVERS], ax=axes[j, i], errorbar=("ci", 95), dodge=0.3, markersize=4)
        axes[j, i].set_title(f"{DATASET_RU[d]}: {METRIC_RU[m]} (чанк {CFG.base_chunk})")
        axes[j, i].set_xlabel("top-k, фрагментов")
        axes[j, i].set_ylabel(METRIC_RU[m])
fig.tight_layout()
save_fig(fig, "topk_curves")

# %% [markdown]
# # 3. Проверка гипотез

# %% [code]
# === H1) Ретривер влияет на качество (парные данные: один вопрос — три ретривера) ===
def paired_matrix(df: pd.DataFrame, factor: str, levels: list, metric: str) -> pd.DataFrame:
    """Матрица вопрос × уровень фактора для парных тестов."""
    return df.pivot_table(index="qid", columns=factor, values=metric).reindex(columns=levels).dropna()


def wilcoxon_p(a: pd.Series, b: pd.Series) -> float:
    """
    Знаково-ранговый тест Вилкоксона, нулевые разности отбрасываются (zero_method='wilcox').

    :return: p-значение (1.0, если все разности нулевые)
    """
    diff = (b - a).values
    if not (diff != 0).any():
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)


def paired_boot_ci(diff: np.ndarray, level: float = 0.95, n_boot: int = CFG.n_boot) -> tuple[float, float]:
    """
    Бутстреп-интервал средней парной разности (вопросы ресэмплируются целиком).

    :param diff: разности по вопросам
    :param level: уровень доверия
    :return: (нижняя, верхняя) граница
    """
    diff = np.asarray(diff, dtype=float)
    means = RNG.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def kendall_w(friedman_stat: float, n: int, k: int) -> float:
    """Коэффициент согласованности Кендалла W из статистики Фридмана."""
    return friedman_stat / (n * (k - 1))


def friedman_block(df, factor, levels, metric_cont, metric_bin):
    """
    Фридман по непрерывной метрике + Q Кохрена по бинарной + попарные Вилкоксон/Мак-Немар (Холм).

    :return: словарь со статистиками
    """
    mat = paired_matrix(df, factor, levels, metric_cont)
    fr = stats.friedmanchisquare(*[mat[c].values for c in levels])
    binm = paired_matrix(df, factor, levels, metric_bin)
    q = cochrans_q(binm.values.astype(int))
    pairs = []
    for a, b in combinations(levels, 2):
        diff = mat[b] - mat[a]
        p = wilcoxon_p(mat[a], mat[b])
        lo, hi = paired_boot_ci(diff.values)
        tab = pd.crosstab(binm[a].astype(int), binm[b].astype(int)).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        mc = mcnemar(tab.values, exact=False, correction=True)
        pairs.append({"a": a, "b": b, "mean_a": mat[a].mean(), "mean_b": mat[b].mean(),
                      "diff_b_minus_a": diff.mean(), "diff_lo": lo, "diff_hi": hi,
                      "wilcoxon_p": p, "mcnemar_p": mc.pvalue,
                      "acc_a": binm[a].mean(), "acc_b": binm[b].mean()})
    pairs = pd.DataFrame(pairs)
    pairs["wilcoxon_p_holm"] = multipletests(pairs["wilcoxon_p"], method="holm")[1]
    pairs["mcnemar_p_holm"] = multipletests(pairs["mcnemar_p"], method="holm")[1]
    return {
        "n": len(mat), "friedman_chi2": fr.statistic, "friedman_p": fr.pvalue,
        "kendall_w": kendall_w(fr.statistic, len(mat), len(levels)),
        "cochran_q": q.statistic, "cochran_p": q.pvalue, "pairs": pairs,
    }


h1 = {}
for d in DATASETS:
    part = base[base.dataset == d]
    for metric in ["recall_at_k", "f1"]:
        res = friedman_block(part, "retriever", RETRIEVERS, metric, "correct" if metric == "f1" else "hit_at_k")
        h1[(d, metric)] = res
        rprint(d, metric, {k: v for k, v in res.items() if k != "pairs"})
        rprint(res["pairs"].round(4))

h1_table = pd.DataFrame(
    [{"dataset": d, "metric": m, **{k: v for k, v in r.items() if k != "pairs"}} for (d, m), r in h1.items()]
).set_index(["dataset", "metric"])
h1_pairs = pd.concat({f"{d}|{m}": r["pairs"] for (d, m), r in h1.items()})
save_table(h1_table, "h1_retriever")
save_table(h1_pairs, "h1_retriever_pairs")

# %% [code]
# === H2) Рост top-k: recall монотонно растет, F1 насыщается (тест тренда Пейджа) ===
h2_rows = []
for d in DATASETS:
    part = answers[(answers.dataset == d) & (answers.chunk_size == CFG.base_chunk) & (answers.retriever == CFG.base_retriever)]
    for metric in ["recall_at_k", "f1"]:
        mat = paired_matrix(part, "top_k", [1, 3, 5, 10], metric)
        page = stats.page_trend_test(mat.values, ranked=False)
        fr = stats.friedmanchisquare(*[mat[c].values for c in mat.columns])
        diff = (mat[10] - mat[5]).values
        lo95, hi95 = paired_boot_ci(diff, 0.95)
        lo90, hi90 = paired_boot_ci(diff, 0.90)
        if metric == "recall_at_k" and d == "squad":
            tab = pd.crosstab(mat[5].astype(int), mat[10].astype(int)).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            p_5_10 = mcnemar(tab.values, exact=False, correction=True).pvalue
        else:
            p_5_10 = wilcoxon_p(mat[5], mat[10])
        h2_rows.append({"dataset": d, "metric": metric, "mean_k1": mat[1].mean(), "mean_k3": mat[3].mean(),
                        "mean_k5": mat[5].mean(), "mean_k10": mat[10].mean(),
                        "page_L": page.statistic, "page_p": page.pvalue, "friedman_p": fr.pvalue,
                        "p_5_vs_10": p_5_10, "diff_10_minus_5": diff.mean(), "diff_ci95_lo": lo95, "diff_ci95_hi": hi95,
                        "tost_ci90_lo": lo90, "tost_ci90_hi": hi90,
                        "equivalent_within_0_02": bool(lo90 > -0.02 and hi90 < 0.02)})
h2 = save_table(pd.DataFrame(h2_rows).set_index(["dataset", "metric"]), "h2_topk")
rprint(h2.round(4))

# %% [code]
# === H3) Размер чанка влияет на качество ===
h3_rows = []
for d in DATASETS:
    part = answers[(answers.dataset == d) & (answers.top_k == CFG.base_k) & (answers.retriever == CFG.base_retriever)]
    res = friedman_block(part, "chunk_size", [50, 100, 200], "f1", "correct")
    rec = friedman_block(part, "chunk_size", [50, 100, 200], "recall_at_k", "hit_at_k")
    h3_rows.append({"dataset": d, "f1_50": res["pairs"].iloc[0]["mean_a"], "f1_100": res["pairs"].iloc[0]["mean_b"],
                    "f1_200": res["pairs"].iloc[1]["mean_b"], "friedman_f1_p": res["friedman_p"], "kendall_w_f1": res["kendall_w"],
                    "cochran_correct_p": res["cochran_p"],
                    "recall_50": rec["pairs"].iloc[0]["mean_a"], "recall_100": rec["pairs"].iloc[0]["mean_b"],
                    "recall_200": rec["pairs"].iloc[1]["mean_b"], "friedman_recall_p": rec["friedman_p"]})
h3 = save_table(pd.DataFrame(h3_rows).set_index("dataset"), "h3_chunk")
rprint(h3.round(4))

# %% [code]
# === H4) Тип и сложность вопроса (HotpotQA, межгрупповые сравнения) ===
hp = base[(base.dataset == "hotpotqa") & (base.retriever == CFG.base_retriever)]
mw = stats.mannwhitneyu(hp[hp.q_type == "bridge"].f1, hp[hp.q_type == "comparison"].f1)
n1, n2 = (hp.q_type == "bridge").sum(), (hp.q_type == "comparison").sum()
rbc = 1 - 2 * mw.statistic / (n1 * n2)
kw = stats.kruskal(*[hp[hp.level == lv].f1 for lv in ["easy", "medium", "hard"]])
eps2 = kw.statistic / (len(hp) - 1)
ct_level = pd.crosstab(hp.level, hp.correct).reindex(["easy", "medium", "hard"])
chi_level = stats.chi2_contingency(ct_level)
cramer_v = np.sqrt(chi_level.statistic / (ct_level.values.sum() * (min(ct_level.shape) - 1)))
ct_type = pd.crosstab(hp.q_type, hp.correct)
chi_type = stats.chi2_contingency(ct_type)
level_pairs = []
for a, b in combinations(["easy", "medium", "hard"], 2):
    level_pairs.append({"a": a, "b": b, "p": stats.mannwhitneyu(hp[hp.level == a].f1, hp[hp.level == b].f1).pvalue})
level_pairs = pd.DataFrame(level_pairs)
level_pairs["p_holm"] = multipletests(level_pairs.p, method="holm")[1]
h4 = {
    "f1_bridge": hp[hp.q_type == "bridge"].f1.mean(), "f1_comparison": hp[hp.q_type == "comparison"].f1.mean(),
    "mw_p": mw.pvalue, "rank_biserial": rbc, "n_bridge": int(n1), "n_comparison": int(n2),
    "f1_easy": hp[hp.level == "easy"].f1.mean(), "f1_medium": hp[hp.level == "medium"].f1.mean(), "f1_hard": hp[hp.level == "hard"].f1.mean(),
    "recall_easy": hp[hp.level == "easy"].recall_at_k.mean(), "recall_medium": hp[hp.level == "medium"].recall_at_k.mean(), "recall_hard": hp[hp.level == "hard"].recall_at_k.mean(),
    "kruskal_H": kw.statistic, "kruskal_p": kw.pvalue, "epsilon2": eps2,
    "chi2_level": chi_level.statistic, "chi2_level_p": chi_level.pvalue, "cramer_v_level": cramer_v,
    "chi2_type_p": chi_type.pvalue,
}
type_by_level = pd.crosstab(hp.level, hp.q_type).reindex(["easy", "medium", "hard"])
save_table(type_by_level, "h4_type_by_level")
h4_ols = smf.ols("f1 ~ C(q_type, Treatment('bridge')) + C(level, Treatment('easy'))", data=hp).fit(cov_type="HC3")
h4["type_coef_adj"] = float(h4_ols.params["C(q_type, Treatment('bridge'))[T.comparison]"])
h4["type_p_adj"] = float(h4_ols.pvalues["C(q_type, Treatment('bridge'))[T.comparison]"])
h4["rank_biserial_convention"] = "1 - 2U/(n1*n2), U for bridge; negative means bridge > comparison"
RESULTS["h4"] = h4
save_table(level_pairs, "h4_level_pairs")
rprint(h4)
rprint(level_pairs)

# %% [code]
# === H5) Найденный контекст не гарантирует ответа: доля ошибок ридера ===
ctx = base.groupby(["dataset", "retriever", "answer_in_context"])["correct"].agg(["mean", "size"]).unstack("answer_in_context")
save_table(ctx, "h5_context_vs_correct")
h5 = {}
for d in DATASETS:
    part = base[(base.dataset == d) & (base.retriever == CFG.base_retriever)]
    tab = pd.crosstab(part.answer_in_context, part.correct)
    odds, p = stats.fisher_exact(tab.values)
    h5[d] = {"p_correct_if_ctx": part[part.answer_in_context == 1].correct.mean(),
             "p_correct_if_no_ctx": part[part.answer_in_context == 0].correct.mean(),
             "odds_ratio": odds, "fisher_p": p, "share_ctx": part.answer_in_context.mean()}
RESULTS["h5"] = h5
rprint(h5)

# %% [markdown]
# # 4. Корреляции параметров и метрик

# %% [code]
# === 1) Спирмен по всей сетке ===
CORR_COLS = ["top_k", "chunk_size", "context_words", "q_len", "lex_overlap", "answer_len",
             "score_margin_r", "top1_score_r", "reader_score", "recall_at_k", "mrr_at_k", "answer_in_context", "f1", "correct"]
CORR_PLOT = {"top_k": "top-k", "chunk_size": "чанк", "q_len": "длина вопроса", "lex_overlap": "пересечение",
             "top1_score_r": "оценка ретривера", "reader_score": "уверенность ридера", "recall_at_k": "полнота",
             "answer_in_context": "ответ в контексте", "f1": "F1", "correct": "правильный"}
answers["top1_score_r"] = answers.groupby(["dataset", "retriever", "chunk_size"])["top1_score"].rank(pct=True)
answers["score_margin_r"] = answers.groupby(["dataset", "retriever", "chunk_size"])["score_margin"].rank(pct=True)
corr = {}
for d in DATASETS:
    part = answers[answers.dataset == d][CORR_COLS]
    rho = part.corr(method="spearman")
    corr[d] = rho
    save_table(rho.round(3), f"spearman_{d}")
fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
for ax, d in zip(axes, DATASETS):
    sub = corr[d].loc[list(CORR_PLOT), list(CORR_PLOT)].rename(index=CORR_PLOT, columns=CORR_PLOT)
    sns.heatmap(sub, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, cbar=False, annot_kws={"size": 10})
    ax.set_title(f"Корреляции Спирмена, {DATASET_RU[d]}")
fig.tight_layout()
save_fig(fig, "spearman")

# %% [code]
# === 2) Значимость ключевых пар с поправкой Холма ===
KEY_PAIRS = [("top_k", "recall_at_k"), ("top_k", "f1"), ("chunk_size", "f1"), ("context_words", "f1"),
             ("lex_overlap", "recall_at_k"), ("lex_overlap", "f1"), ("q_len", "f1"), ("top1_score_r", "recall_at_k"),
             ("score_margin_r", "f1"), ("reader_score", "correct"), ("recall_at_k", "f1")]
rows = []
for d in DATASETS:
    part = answers[answers.dataset == d]
    per_q = part.groupby("qid")
    for a, b in KEY_PAIRS:
        r, p = stats.spearmanr(part[a], part[b])
        rows.append({"dataset": d, "x": a, "y": b, "rho": r, "p": p})
key_corr = pd.DataFrame(rows)
key_corr["p_holm"] = multipletests(key_corr.p, method="holm")[1]
save_table(key_corr.set_index(["dataset", "x", "y"]).round(4), "key_corr")
rprint(key_corr.round(4))

# %% [code]
# === 3) Точечно-бисериальная: уверенность ридера и правильность в базовой конфигурации ===
pb = {}
for d in DATASETS:
    part = base[(base.dataset == d) & (base.retriever == CFG.base_retriever)]
    r, p = stats.pointbiserialr(part.correct, part.reader_score)
    pb[d] = {"r_pb": r, "p": p}
RESULTS["point_biserial"] = pb
rprint(pb)

# %% [markdown]
# # 5. Регрессия качества ~ параметры

# %% [code]
# === 1) OLS с кластерными ошибками по вопросу ===
reg = answers.copy()
reg["log2_k"] = np.log2(reg["top_k"])
reg["chunk"] = reg["chunk_size"].astype(str)
FORMULA = "f1 ~ C(retriever, Treatment('bm25')) * C(dataset) + log2_k * C(dataset) + C(chunk, Treatment('100')) + q_len + lex_overlap + answer_is_numeric"
ols = smf.ols(FORMULA, data=reg).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(reg["qid"])[0]})
rprint(ols.summary())
ols_tab = pd.DataFrame({"coef": ols.params, "se": ols.bse, "p": ols.pvalues,
                        "ci_lo": ols.conf_int()[0], "ci_hi": ols.conf_int()[1]})
save_table(ols_tab.round(4), "ols_f1")
RESULTS["ols"] = {"r2": ols.rsquared, "r2_adj": ols.rsquared_adj, "n": int(ols.nobs), "n_clusters": int(reg.qid.nunique())}

# %% [code]
# === 2) Доля дисперсии: параметры системы против свойств вопроса ===
sys_only = smf.ols("f1 ~ C(retriever) * C(dataset) + log2_k * C(dataset) + C(chunk)", data=reg).fit()
q_fe = smf.ols("f1 ~ C(qid)", data=reg).fit()
RESULTS["variance_split"] = {"r2_system_params": sys_only.rsquared, "r2_question_fixed_effects": q_fe.rsquared,
                             "r2_full_model": ols.rsquared}
for d in DATASETS:
    part = reg[reg.dataset == d]
    RESULTS["variance_split"][f"{d}_r2_system_params"] = smf.ols("f1 ~ C(retriever) + log2_k + C(chunk)", data=part).fit().rsquared
    RESULTS["variance_split"][f"{d}_r2_question"] = smf.ols("f1 ~ C(qid)", data=part).fit().rsquared
rprint(RESULTS["variance_split"])

# %% [code]
# === 3) Логистическая регрессия правильности (GEE, обменная корреляция внутри вопроса) ===
gee = smf.gee(
    "correct ~ C(retriever, Treatment('bm25')) + C(dataset) + log2_k + C(chunk, Treatment('100')) + q_len + lex_overlap + answer_is_numeric",
    groups="qid", data=reg, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
).fit()
rprint(gee.summary())
gee_tab = pd.DataFrame({"coef": gee.params, "odds_ratio": np.exp(gee.params), "se": gee.bse, "p": gee.pvalues,
                        "or_lo": np.exp(gee.conf_int()[0]), "or_hi": np.exp(gee.conf_int()[1])})
save_table(gee_tab.round(4), "gee_correct")

# %% [code]
# === 4) Диагностика: VIF и остатки ===
main_eff = smf.ols("f1 ~ C(retriever, Treatment('bm25')) + C(dataset) + log2_k + C(chunk, Treatment('100')) + q_len + lex_overlap + answer_is_numeric", data=reg).fit()
X = pd.DataFrame(main_eff.model.exog, columns=main_eff.model.exog_names)
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=X.columns[1:])
save_table(vif.round(2).to_frame("vif"), "vif")
rprint(vif.round(2))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].scatter(ols.fittedvalues, ols.resid, s=2, alpha=0.15)
axes[0].set_xlabel("предсказанный F1")
axes[0].set_ylabel("остаток")
sm.qqplot(ols.resid.sample(3000, random_state=CFG.seed), line="s", ax=axes[1], markersize=2)
fig.tight_layout()
save_fig(fig, "ols_diagnostics")

# %% [markdown]
# # 6. Классификация правильный / неправильный

# %% [code]
# === 1) Признаки и групповая кросс-валидация по вопросу ===
clf_df = answers.copy()
clf_df["is_hotpot"] = (clf_df.dataset == "hotpotqa").astype(int)
for r in RETRIEVERS:
    clf_df[f"ret_{r}"] = (clf_df.retriever == r).astype(int)
FEATURES_PRE = ["is_hotpot", "ret_bm25", "ret_dense", "ret_hybrid", "top_k", "chunk_size", "context_words",
                "q_len", "top1_score_r", "score_margin_r"]
FEATURES_PRE_NO_DS = [f for f in FEATURES_PRE if f != "is_hotpot"]
FEATURES_POST = FEATURES_PRE + ["reader_score", "pred_rank"]
y = clf_df["correct"].values
groups = clf_df["qid"].values
cv = GroupKFold(n_splits=5)


def cv_eval(model_factory, features):
    """
    Метрики на 5 групповых фолдах (вопрос целиком в одном фолде).

    :return: словарь средних метрик и out-of-fold предсказания
    """
    oof = np.zeros(len(y))
    for tr, te in cv.split(clf_df, y, groups):
        m = model_factory()
        m.fit(clf_df.iloc[tr][features].astype(float), y[tr])
        oof[te] = m.predict_proba(clf_df.iloc[te][features].astype(float))[:, 1]
    return {"roc_auc": roc_auc_score(y, oof), "pr_auc": average_precision_score(y, oof),
            "accuracy": accuracy_score(y, oof >= 0.5), "brier": brier_score_loss(y, oof)}, oof


MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "gboost": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=CFG.seed),
}
clf_rows, oofs = [], {}
majority = max(y.mean(), 1 - y.mean())
clf_rows.append({"model": "majority", "features": "-", "roc_auc": 0.5, "pr_auc": y.mean(), "accuracy": majority, "brier": y.mean() * (1 - y.mean())})
for fname, feats in [("pre_reader_no_dataset", FEATURES_PRE_NO_DS), ("pre_reader", FEATURES_PRE), ("with_reader", FEATURES_POST)]:
    for mname, factory in MODELS.items():
        res, oof = cv_eval(factory, feats)
        oofs[(mname, fname)] = oof
        clf_rows.append({"model": mname, "features": fname, **res})
clf_tab = save_table(pd.DataFrame(clf_rows).set_index(["model", "features"]), "classification")
rprint(clf_tab.round(4))

# %% [code]
# === 2) Важность признаков (перестановочная, на отложенном фолде) ===
tr, te = next(cv.split(clf_df, y, groups))
gb = MODELS["gboost"]().fit(clf_df.iloc[tr][FEATURES_POST].astype(float), y[tr])
pi = permutation_importance(gb, clf_df.iloc[te][FEATURES_POST].astype(float), y[te], scoring="roc_auc", n_repeats=5, random_state=CFG.seed)
FEATURE_RU = {"is_hotpot": "датасет HotpotQA", "ret_bm25": "ретривер BM25", "ret_dense": "ретривер плотный", "ret_hybrid": "ретривер гибрид",
              "top_k": "top-k", "chunk_size": "размер чанка", "context_words": "слов в контексте", "q_len": "длина вопроса",
              "top1_score_r": "оценка ретривера (перцентиль)", "score_margin_r": "отрыв 1-го от 2-го (перцентиль)",
              "reader_score": "уверенность ридера", "pred_rank": "ранг чанка с ответом"}
imp = pd.Series(pi.importances_mean, index=FEATURES_POST).sort_values(ascending=False)
save_table(imp.round(4).to_frame("auc_drop"), "feature_importance")
lr = MODELS["logreg"]().fit(clf_df[FEATURES_PRE].astype(float), y)
lr_coef = pd.Series(lr[-1].coef_[0], index=FEATURES_PRE).sort_values()
save_table(lr_coef.round(4).to_frame("std_coef"), "logreg_coef")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
imp.rename(FEATURE_RU).plot.barh(ax=axes[0])
axes[0].invert_yaxis()
axes[0].set_title("Падение ROC-AUC при перестановке (gboost)")
from sklearn.metrics import roc_curve
for (mname, fname), oof in oofs.items():
    fpr, tpr, _ = roc_curve(y, oof)
    names = {"logreg": "логрегрессия", "gboost": "бустинг", "pre_reader_no_dataset": "до ридера без датасета",
             "pre_reader": "до ридера", "with_reader": "с ридером"}
    axes[1].plot(fpr, tpr, label=f"{names[mname]}, {names[fname]}")
axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[1].legend(fontsize=7)
axes[1].set_title("ROC-кривые, out-of-fold")
axes[1].set_xlabel("доля ложных срабатываний")
axes[1].set_ylabel("доля найденных правильных")
fig.tight_layout()
save_fig(fig, "classification")

# %% [markdown]
# # 7. Динамика по итерациям

# %% [code]
# === 1) Средние по итерациям ===
it_mean = iters.groupby(["dataset", "mode", "iteration"])[["recall", "answer_in_context", "f1", "correct", "context_words"]].mean()
save_table(it_mean.round(4), "iterations_mean")
rprint(it_mean.round(3))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, metric in zip(axes, ["recall", "f1"]):
    for (d, mode), g in iters.groupby(["dataset", "mode"]):
        m = g.groupby("iteration")[metric].mean()
        mode_ru = "итеративный" if mode == "iterative" else "одношаговый"
        ax.plot(m.index, m.values, marker="o" if mode == "iterative" else "s", linestyle="-" if d == "hotpotqa" else "--", label=f"{DATASET_RU[d]}, {mode_ru}")
    ax.set_title(f"{METRIC_RU[metric].capitalize()} по шагам поиска")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("шаг (по 3 фрагмента на шаг)")
    ax.set_ylabel(METRIC_RU[metric])
axes[0].legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "iterations")

# %% [code]
# === 2) Тренд внутри итеративного режима и сравнение с одношаговым поиском ===
it_rows = []
for d in DATASETS:
    part = iters[(iters.dataset == d)]
    for mode in ["iterative", "single"]:
        for metric in ["recall", "f1"]:
            mat = part[part["mode"] == mode].pivot_table(index="qid", columns="iteration", values=metric)
            page_up = stats.page_trend_test(mat.values, ranked=False)
            page_down = stats.page_trend_test(mat.values[:, ::-1], ranked=False)
            fr = stats.friedmanchisquare(*[mat[c].values for c in mat.columns])
            lo, hi = paired_boot_ci((mat[4] - mat[1]).values)
            it_rows.append({"dataset": d, "mode": mode, "metric": metric, "page_up_p": page_up.pvalue, "page_down_p": page_down.pvalue,
                            "friedman_p": fr.pvalue, "delta_1_to_4": mat[4].mean() - mat[1].mean(), "delta_lo": lo, "delta_hi": hi})
    for it in [2, 3, 4]:
        for metric in ["recall", "f1"]:
            a = part[(part["mode"] == "iterative") & (part.iteration == it)].set_index("qid")[metric]
            b = part[(part["mode"] == "single") & (part.iteration == it)].set_index("qid")[metric].reindex(a.index)
            if metric == "recall" and d == "squad":
                tab = pd.crosstab(b.astype(int), a.astype(int)).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
                p, test = mcnemar(tab.values, exact=False, correction=True).pvalue, "mcnemar"
            else:
                p, test = wilcoxon_p(b, a), "wilcoxon"
            lo, hi = paired_boot_ci((a - b).values)
            it_rows.append({"dataset": d, "mode": f"iter_vs_single_t{it}", "metric": metric, "test": test,
                            "mean_iterative": a.mean(), "mean_single": b.mean(), "p": p, "delta_lo": lo, "delta_hi": hi})
it_tests = pd.DataFrame(it_rows)
cmp_mask = it_tests["mode"].str.startswith("iter_vs_single")
it_tests.loc[cmp_mask, "p_holm"] = it_tests[cmp_mask].groupby(["dataset", "metric"])["p"].transform(lambda s: multipletests(s, method="holm")[1])
it_tests = save_table(it_tests, "iterations_tests")
rprint(it_tests.round(4))

# %% [code]
# === 3) GEE: правильность ~ итерация × режим ===
gee_frames = []
for d in DATASETS:
    gee_it = smf.gee("correct ~ iteration * C(mode, Treatment('single'))", groups="qid", data=iters[iters.dataset == d],
                     family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    gee_frames.append(pd.DataFrame({"dataset": d, "coef": gee_it.params, "odds_ratio": np.exp(gee_it.params), "p": gee_it.pvalues}))
gee_it_tab = save_table(pd.concat(gee_frames).round(4), "gee_iterations")
rprint(gee_it_tab)

# %% [code]
# === 4) Сводка для отчета ===
RESULTS["h1"] = {f"{d}|{m}": {k: float(v) for k, v in r.items() if k != "pairs"} for (d, m), r in h1.items()}
with open(OUT_TAB / "results.json", "w") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=float)
rprint(RESULTS)
