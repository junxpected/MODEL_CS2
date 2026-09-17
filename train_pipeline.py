import os
import sys
import glob
import subprocess
import gc
import polars as pl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from parse_demo import parse_demo
from features import sample_round_states

DEMOS_DIR = "demos"
SAMPLES_DIR = "samples_cache"
PROCESSED_LOG = "processed_demos.txt"
MODEL_OUT = "wp_model_final.json"

KNOWN_MAPS = ["de_ancient", "de_anubis", "de_dust2", "de_inferno",
              "de_mirage", "de_nuke", "de_train", "de_vertigo"]


def ensure_dirs():
    os.makedirs(DEMOS_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)


def get_processed_set() -> set:
    if not os.path.exists(PROCESSED_LOG):
        return set()
    with open(PROCESSED_LOG) as f:
        return set(line.strip() for line in f if line.strip())


def mark_processed(filename: str):
    with open(PROCESSED_LOG, "a") as f:
        f.write(filename + "\n")


def decompress_zst_files():
    """Розпаковує всі .zst файли в demos/ у .dem поруч (якщо .dem ще нема)."""
    for zst_path in glob.glob(os.path.join(DEMOS_DIR, "*.zst")):
        dem_path = zst_path.rsplit(".zst", 1)[0]
        if not dem_path.endswith(".dem"):
            dem_path += ".dem"
        if os.path.exists(dem_path):
            continue
        print(f"Розпаковую: {os.path.basename(zst_path)}")
        result = subprocess.run(["zstd", "-d", "-q", zst_path, "-o", dem_path])
        if result.returncode != 0:
            print(f"  Не вдалось розпакувати {zst_path} — переконайся, що встановлено zstd (apt install zstd)")


def process_new_demos():
    """Парсить усі ще не оброблені .dem файли і зберігає їх семпли окремо."""
    processed = get_processed_set()
    dem_files = glob.glob(os.path.join(DEMOS_DIR, "*.dem"))
    new_files = [f for f in dem_files if os.path.basename(f) not in processed]

    if not new_files:
        print("Нових демок не знайдено — усі вже оброблені раніше.")
        return

    print(f"Знайдено {len(new_files)} нових демок для обробки.\n")

    for i, path in enumerate(new_files, 1):
        name = os.path.basename(path)
        print(f"[{i}/{len(new_files)}] {name}")
        try:
            parsed = parse_demo(path)
            map_name = parsed["header"].get("map_name", "unknown")
            n_rounds = len(parsed["rounds"])
            print(f"  Мапа: {map_name}, раундів: {n_rounds}")

            samples = sample_round_states(parsed, sample_every_n_ticks=64)
            samples = samples.with_columns([
                pl.lit(name).alias("source_demo"),
                pl.lit(map_name).alias("map_name"),
                pl.col("ct_health_sum").cast(pl.Float64),
                pl.col("t_health_sum").cast(pl.Float64),
                pl.col("ct_equip_value").cast(pl.Float64),
                pl.col("t_equip_value").cast(pl.Float64),
            ])
            out_path = os.path.join(SAMPLES_DIR, name.replace(".dem", ".parquet"))
            samples.write_parquet(out_path)
            print(f"  Семплів: {samples.height}")

            mark_processed(name)
            del parsed
            gc.collect()
        except Exception as e:
            print(f"  ПОМИЛКА: {e} — цю демку пропускаю, вона НЕ позначена як оброблена, спробує знову наступного разу")


def retrain_model():
    """Об'єднує всі накопичені семпли й перетреновує модель."""
    sample_files = glob.glob(os.path.join(SAMPLES_DIR, "*.parquet"))
    if not sample_files:
        print("Немає жодної обробленої демки — нема на чому тренувати.")
        return

    print(f"\nОб'єдную {len(sample_files)} файлів семплів...")
    dfs = [pl.read_parquet(f) for f in sample_files]
    combined = pl.concat(dfs)

    combined = combined.filter(pl.col("map_name").is_in(KNOWN_MAPS))
    combined = combined.to_dummies(columns=["map_name"])
    map_cols = [c for c in combined.columns if c.startswith("map_name_")]

    combined = combined.with_columns(
        (pl.col("source_demo") + "_" + pl.col("round_num").cast(pl.Utf8)).alias("round_key")
    )
    round_keys = combined["round_key"].unique().to_list()
    print(f"Всього семплів: {combined.height}, унікальних раундів: {len(round_keys)}")

    if len(round_keys) < 20:
        print("УВАГА: менше 20 раундів — модель буде ненадійною. Додай більше демок.")

    train_keys, test_keys = train_test_split(round_keys, test_size=0.2, random_state=42)
    train = combined.filter(pl.col("round_key").is_in(train_keys))
    test = combined.filter(pl.col("round_key").is_in(test_keys))

    features = ["ct_alive", "t_alive", "ct_health_sum", "t_health_sum",
                "ct_equip_value", "t_equip_value", "seconds_into_round"] + map_cols

    X_train, y_train = train.select(features).to_numpy(), train["ct_won"].to_numpy()
    X_test, y_test = test.select(features).to_numpy(), test["ct_won"].to_numpy()

    model = xgb.XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, eval_metric="logloss")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\nTrain раундів: {len(train_keys)}, Test раундів: {len(test_keys)}")
    print(f"Точність: {accuracy_score(y_test, preds):.1%}")
    print(f"AUC: {roc_auc_score(y_test, probs):.3f}")

    # фінальна модель тренується на ВСІХ даних (без відкладеного тесту) для реального використання
    X_all, y_all = combined.select(features).to_numpy(), combined["ct_won"].to_numpy()
    final_model = xgb.XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, eval_metric="logloss")
    final_model.fit(X_all, y_all)
    final_model.save_model(MODEL_OUT)
    print(f"\nМодель збережена: {MODEL_OUT}")


if __name__ == "__main__":
    ensure_dirs()
    decompress_zst_files()
    process_new_demos()
    retrain_model()
