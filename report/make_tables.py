from pathlib import Path

import pandas as pd


REPORT = Path(__file__).resolve().parent
TABLES = REPORT / "tables"
DATASET_NAMES = {"hotpotqa": "HotpotQA", "squad": "SQuAD"}
RETRIEVER_NAMES = {"bm25": "BM25", "dense": "Плотный (e5)", "hybrid": "Гибрид (RRF)"}


def num(x: float, digits: int = 3) -> str:
    """Число с десятичной запятой."""
    return f"{x:.{digits}f}".replace(".", ",")


def ci(mean: float, lo: float, hi: float) -> str:
    """Среднее и 95\\% интервал в одну ячейку LaTeX."""
    return f"{num(mean)} \\scriptsize[{num(lo)}; {num(hi)}]"


def desc_table() -> str:
    """Таблица описательной статистики базовой конфигурации (чанк 100, top-5)."""
    d = pd.read_csv(TABLES / "desc_base.csv")
    lines = []
    for ds in ["hotpotqa", "squad"]:
        for r in ["bm25", "dense", "hybrid"]:
            row = d[(d.dataset == ds) & (d.retriever == r)].iloc[0]
            lines.append(
                f"{DATASET_NAMES[ds]} & {RETRIEVER_NAMES[r]} & {ci(row.recall_at_k, row.recall_at_k_lo, row.recall_at_k_hi)}"
                f" & {ci(row.f1, row.f1_lo, row.f1_hi)}"
                f" & {ci(row.correct, row.correct_lo, row.correct_hi)} \\\\"
            )
        lines.append("\\midrule")
    return "\n".join(lines[:-1])


def grid_table(dataset: str, metric: str) -> str:
    """Таблица метрики по всей сетке: строки ретривер × чанк, колонки top-k."""
    m = pd.read_csv(TABLES / "metrics_by_config.csv")
    m = m[m.dataset == dataset]
    lines = []
    for r in ["bm25", "dense", "hybrid"]:
        for c in [50, 100, 200]:
            part = m[(m.retriever == r) & (m.chunk_size == c)].set_index("top_k")[metric]
            cells = " & ".join(num(part[k]) for k in [1, 3, 5, 10])
            lines.append(f"{RETRIEVER_NAMES[r]} & {c} & {cells} \\\\")
        lines.append("\\midrule")
    return "\n".join(lines[:-1])


def iterations_table() -> str:
    """Средние по итерациям: полнота и F1 для итеративного и одношагового режима."""
    it = pd.read_csv(TABLES / "iterations_mean.csv")
    lines = []
    for ds in ["hotpotqa", "squad"]:
        for t in [1, 2, 3, 4]:
            a = it[(it.dataset == ds) & (it["mode"] == "iterative") & (it.iteration == t)].iloc[0]
            b = it[(it.dataset == ds) & (it["mode"] == "single") & (it.iteration == t)].iloc[0]
            lines.append(
                f"{DATASET_NAMES[ds] if t == 1 else ''} & {t} & {3 * t} & {num(a.recall)} & {num(b.recall)}"
                f" & {num(a.f1)} & {num(b.f1)} \\\\"
            )
        lines.append("\\midrule")
    return "\n".join(lines[:-1])


if __name__ == "__main__":
    out = {
        "tab_desc.tex": desc_table(),
        "tab_grid_hotpot_f1.tex": grid_table("hotpotqa", "f1"),
        "tab_grid_squad_f1.tex": grid_table("squad", "f1"),
        "tab_grid_hotpot_recall.tex": grid_table("hotpotqa", "recall_at_k"),
        "tab_grid_squad_recall.tex": grid_table("squad", "recall_at_k"),
        "tab_iterations.tex": iterations_table(),
    }
    for name, body in out.items():
        (TABLES / name).write_text(body + "\n")
        print(f"Записана таблица: {name}")
