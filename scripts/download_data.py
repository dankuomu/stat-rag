import logging
from pathlib import Path

from datasets import load_dataset


DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"

logger = logging.getLogger(__name__)


def download() -> dict:
    """
    Скачать HotpotQA (distractor, train и dev) и SQuAD v1.1 (dev) в parquet.

    :return: число строк по файлам
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    jobs = {
        "hotpotqa_distractor_train.parquet": ("hotpotqa/hotpot_qa", "distractor", "train", HOTPOT_REVISION),
        "hotpotqa_distractor_dev.parquet": ("hotpotqa/hotpot_qa", "distractor", "validation", HOTPOT_REVISION),
        "squad_v1_dev.parquet": ("rajpurkar/squad", None, "validation", SQUAD_REVISION),
    }
    sizes = {}
    for fname, (repo, config, split, revision) in jobs.items():
        ds = load_dataset(repo, config, split=split, revision=revision)
        ds.to_parquet(DATA_RAW / fname)
        sizes[fname] = len(ds)
        logger.info(f"{fname}: {len(ds)} rows")
    return sizes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for name, n in download().items():
        print(f"{name}: {n} строк")
