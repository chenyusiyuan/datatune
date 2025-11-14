"""A command line demo to run the system up to data generation,
then visualize clusters with Ollama embeddings, and exit."""

import os

# Ensure Hugging Face traffic goes through mirror before importing HF-dependent libs.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
import logging
import time
from pathlib import Path
import shutil

import datasets
import pyfiglet
import torch
import transformers
import yaml
from datasets import load_from_disk
from termcolor import colored

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.prompt_based import PromptBasedDatasetGenerator
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from prompt2model.dataset_retriever import DescriptionDatasetRetriever
from prompt2model.prompt_parser import (
    MockPromptSpec,
    PromptBasedInstructionParser,
    TaskType,
)
from prompt2model.utils.logging_utils import get_formatted_logger

# ===== 可视化依赖 =====
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ========== 小工具 ==========
def line_print(s: str) -> None:
    print(s, flush=True)


def print_logo():
    figlet = pyfiglet.Figlet(width=200)
    words = ["Prompt", "2", "Model"]
    colors = ["red", "green", "blue"]
    ascii_art_parts = [figlet.renderText(word).split("\n") for word in words]
    max_height = max(len(part) for part in ascii_art_parts)
    for part in ascii_art_parts:
        while len(part) < max_height:
            part.append("")
    ascii_art_lines = []
    for lines in zip(*ascii_art_parts):
        colored_line = " ".join(
            colored(line, color) for line, color in zip(lines, colors)
        )
        ascii_art_lines.append(colored_line)
    ascii_art = "\n".join(ascii_art_lines)
    try:
        # 避免 tee 等非 TTY 场景报 ioctl 错
        term_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    except Exception:
        term_width = int(os.environ.get("COLUMNS", 120))
    centered_ascii_art = "\n".join(line.center(term_width) for line in ascii_art.split("\n"))
    line_print(centered_ascii_art)


def parse_model_size_limit(line: str, default_size=3e9) -> float:
    if len(line.strip()) == 0:
        return default_size
    model_units = {"B": 1e0, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "PB": 1e15}
    unit_disambiguations = {
        "KB": ["Kb", "kb", "kilobytes"],
        "MB": ["Mb", "mb", "megabytes"],
        "GB": ["Gb", "gb", "gigabytes"],
        "TB": ["Tb", "tb", "terabytes"],
        "PB": ["Pb", "pb", "petabytes"],
        "B": ["b", "bytes"],
    }
    unit_matched = False
    for unit, disambiguations in unit_disambiguations.items():
        for unit_name in [unit] + disambiguations:
            if line.strip().endswith(unit_name):
                unit_matched = True
                break
        if unit_matched:
            break
    if unit_matched:
        numerical_part = line.strip()[: -len(unit_name)].strip()
    else:
        numerical_part = line.strip()
    if not str.isdecimal(numerical_part):
        raise ValueError("Invalid input. Please enter a number (integer or with units).")
    scale_factor = model_units[unit] if unit_matched else 1
    return int(numerical_part) * scale_factor


def _flatten_generated_to_df(root: str) -> pd.DataFrame:
    """从 HF Dataset 磁盘目录读取，并映射到通用列: input/output/label/source/__text__"""
    if not root or not Path(root).exists():
        return pd.DataFrame(columns=["input","output","label","source","__text__"])
    ds = load_from_disk(root)
    cols = ds.column_names

    def pick(names):
        for n in names:
            if n in cols:
                return n
        return None
    in_col  = pick(["input","input_col","instruction","prompt","question","x","text"])
    out_col = pick(["output","output_col","response","answer","y"])
    in_col  = pick(["input","instruction","prompt","question","x","text"])
    out_col = pick(["output","response","answer","y"])
    df = ds.to_pandas().copy()
    if in_col is None:
        in_col = cols[0]
    if out_col is None and len(cols) > 1:
        out_col = cols[1]
    rename_map = {}
    if in_col != "input": rename_map[in_col] = "input"
    if out_col and out_col != "output": rename_map[out_col] = "output"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    if "label" not in df.columns:
        df["label"] = "synthetic"
    if "source" not in df.columns:
        df["source"] = "synthetic"
    df["output"] = df.get("output", "").astype(str)
    df["input"]  = df.get("input", "").astype(str)
    df["__text__"] = df["input"] + " || " + df["output"]
    return df[["input","output","label","source","__text__"]]


def _flatten_retrieved_to_df(root: str) -> pd.DataFrame:
    """读取检索数据（DatasetDict），取 train split。"""
    if (not root) or (not Path(root).exists()):
        return pd.DataFrame(columns=["input","output","label","source","__text__"])
    ddict = datasets.load_from_disk(root)
    if "train" not in ddict:
        return pd.DataFrame(columns=["input","output","label","source","__text__"])
    ds = ddict["train"]
    cols = ds.column_names

    def pick(names):
        for n in names:
            if n in cols:
                return n
        return None
    in_col  = pick(["input","input_col","instruction","prompt","question","x","text"])
    out_col = pick(["output","output_col","response","answer","y"])
    in_col  = pick(["input","instruction","prompt","question","x","text"])
    out_col = pick(["output","response","answer","y"])
    df = ds.to_pandas().copy()
    if in_col is None:
        in_col = cols[0]
    if out_col is None and len(cols) > 1:
        out_col = cols[1]
    rename_map = {}
    if in_col != "input": rename_map[in_col] = "input"
    if out_col and out_col != "output": rename_map[out_col] = "output"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    df["source"] = "retrieved"
    if "label" not in df.columns:
        df["label"] = "retrieved"
    df["output"] = df.get("output", "").astype(str)
    df["input"]  = df.get("input", "").astype(str)
    df["__text__"] = df["input"] + " || " + df["output"]
    return df[["input","output","label","source","__text__"]]


def _embed_with_ollama(texts, model="nomic-embed-text", host="http://localhost:11434"):
    """调用 Ollama Embeddings API 生成向量。"""
    import requests
    url = f"{host}/api/embeddings"
    embs = []
    for t in texts:
        r = requests.post(url, json={"model": model, "prompt": str(t)})
        r.raise_for_status()
        embs.append(r.json()["embedding"])
    return np.asarray(embs, dtype=np.float32)


def _reduce_and_plot(emb, df, color_col="label", method="umap", out_path="synth_clusters.png"):
    # 优先 UMAP，失败则 t-SNE
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42) if method.lower()=="umap" else None
    except Exception:
        reducer = None

    if reducer is None and method.lower()!="tsne":
        method = "tsne"
    if method.lower()=="tsne":
        from sklearn.manifold import TSNE
        xy = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42).fit_transform(emb)
    else:
        xy = reducer.fit_transform(emb)

    # 1) 支持中文的字体回退
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei", "WenQuanYi Zen Hei", "Arial Unicode MS"]
    # 2) 允许负号正常显示
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(9,7))
    labels = df[color_col].astype(str).fillna("NA").values if color_col in df.columns else np.array(["ALL"]*len(df))
    sources = df["source"].astype(str).fillna("unknown").values if "source" in df.columns else np.array(["unknown"]*len(df))

    uniq_labels  = sorted(list(set(labels)))
    marker_map   = {"retrieved":"o", "synthetic":"^", "unknown":"."}

    # 避免类别过多导致图例过长
    if len(uniq_labels) > 20:
        top20 = uniq_labels[:20]
        labels = np.array([l if l in top20 else "OTHER" for l in labels])
        uniq_labels = sorted(list(set(labels)))

    for lab in uniq_labels:
        lab_idx = (labels == lab)
        for src in sorted(list(set(sources))):
            idx = lab_idx & (sources == src)
            if not np.any(idx):
                continue
            plt.scatter(xy[idx,0], xy[idx,1], s=12, alpha=0.8,
                        marker=marker_map.get(src, "."),
                        label=f"{lab} ({src})")

    handles, texts = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(texts, handles))
    ncol = 1 if len(by_label) <= 12 else 2
    plt.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best", ncol=ncol)
    plt.title(f"Clusters colored by '{color_col}' ({method.upper()})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    line_print(f"[OK] cluster figure saved to: {out_path}")


def _normalize_io_columns(ds):
    """
    确保 HF Dataset 里有标准列名: input / output。
    若发现是 input_col / output_col，就重命名为 input / output。
    兼容其它常见别名（instruction/prompt/question/response/answer/y 等）
    """
    cols = set(ds.column_names)

    # 依次尝试匹配输入列
    in_candidates  = ["input", "input_col", "instruction", "prompt", "question", "x", "text"]
    out_candidates = ["output", "output_col", "response", "answer", "y"]

    in_col  = next((c for c in in_candidates  if c in cols), None)
    out_col = next((c for c in out_candidates if c in cols), None)

    if in_col is None or out_col is None:
        raise ValueError(f"Cannot find input/output columns in dataset. cols={ds.column_names}")

    # 如果已经是标准名就不动；否则重命名到标准名
    if in_col != "input":
        # 避免目的列已存在
        if "input" in cols:
            ds = ds.remove_columns(["input"])
        ds = ds.rename_column(in_col, "input")
    if out_col != "output":
        if "output" in set(ds.column_names):
            ds = ds.remove_columns(["output"])
        ds = ds.rename_column(out_col, "output")

    return ds


def _post_dedup_dataset(ds: datasets.Dataset) -> datasets.Dataset:
    """
    收尾去重：先标准化列名为 input/output；
    1) 对 (input, output) 完全重复去重；
    2) 对相同 input，保留最短 output（若长度相同，保留出现顺序靠前的）。
    """
    if len(ds) == 0:
        return ds

    # 统一列名
    ds = _normalize_io_columns(ds)

    df = ds.to_pandas()

    # 完全重复去重
    df = df.drop_duplicates(subset=["input", "output"], keep="first")

    # 相同 input -> 选最短 output
    df["__out_len__"] = df["output"].astype(str).map(len)
    df = df.sort_values(["input", "__out_len__"], ascending=[True, True])
    df = df.drop_duplicates(subset=["input"], keep="first").drop(columns="__out_len__")

    return datasets.Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)



def _export_sidecars(ds: datasets.Dataset, out_stem: Path):
    """便携的旁路导出：JSONL/CSV（不强依赖 HF 版本）。"""
    try:
        # JSONL
        jsonl_path = out_stem.with_suffix(".jsonl")
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in ds:
                f.write(json.dumps({k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                                    for k, v in r.items()}, ensure_ascii=False) + "\n")
        line_print(f"[OK] exported: {jsonl_path}")
    except Exception as e:
        line_print(f"[WARN] jsonl export failed: {e}")

    try:
        # CSV
        csv_path = out_stem.with_suffix(".csv")
        ds.to_pandas().to_csv(csv_path, index=False)
        line_print(f"[OK] exported: {csv_path}")
    except Exception as e:
        line_print(f"[WARN] csv export failed: {e}")


def main():
    # ===== 后端与模型（本地 Ollama） =====
    os.environ["P2M_BACKEND"] = "ollama"
    # 指令大模型（合成用）
    os.environ["P2M_GEN_MODEL"] = os.getenv("P2M_GEN_MODEL", "llama3.1")
    # 向量模型（可视化嵌入）
    os.environ["P2M_OLLAMA_EMBED_MODEL"] = os.getenv("P2M_OLLAMA_EMBED_MODEL", "nomic-embed-text")

    print_logo()

    # ===== Run 目录（安全落盘 + 可复现）=====
    out_root = Path(os.getenv("P2M_OUTPUT_DIR", "runs"))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    line_print(f"[RUN] output dir: {run_dir.resolve()}")

    # Save/Resume status（仍在工作根目录保留 status.yaml 以便恢复，同时也归档一份到 run_dir）
    if os.path.isfile("status.yaml"):
        with open("status.yaml", "r") as f:
            status = yaml.safe_load(f)
    else:
        status = {}

    while True:
        line_print("Do you want to start from scratch? (y/n)")
        answer = input().strip()
        if answer.lower() == "n":
            if os.path.isfile("status.yaml"):
                with open("status.yaml", "r") as f:
                    status = yaml.safe_load(f)
                    print(f"Current status:\n{json.dumps(status, indent=4)}", flush=True)
                    break
            else:
                status = {}
                break
        elif answer.lower() == "y":
            status = {}
            break
        else:
            continue

    propmt_has_been_parsed = status.get("prompt_has_been_parsed", False)
    dataset_has_been_retrieved = status.get("dataset_has_been_retrieved", False)
    dataset_has_been_generated = status.get("dataset_has_been_generated", False)

    # ===== 1) 解析任务说明 =====
    if not propmt_has_been_parsed:
        prompt_file = "prompt.yaml"
        loaded_from_file = False

        if prompt_file and Path(prompt_file).exists():
            with open(prompt_file, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f) if prompt_file.endswith((".yml", ".yaml")) else json.load(f)
            if "instruction" not in meta:
                raise ValueError(f"[prompt] 文件缺少 'instruction' 字段: {prompt_file}")
            instruction = meta["instruction"]
            examples = meta.get("examples", [])
            status["instruction"] = instruction
            status["examples"] = examples
            status["prompt_has_been_parsed"] = True
            with open("status.yaml", "w") as f:
                yaml.safe_dump(status, f)
            propmt_has_been_parsed = True
            loaded_from_file = True
            line_print(f"Loaded prompt from file: {prompt_file}")
            # 归档 prompt 到 run_dir
            try:
                shutil.copy2(prompt_file, run_dir / "prompt.yaml")
            except Exception:
                pass

        if not loaded_from_file:
            # 交互式输入
            prompt = ""
            line_print("Enter your task description and few-shot examples (or 'done' to finish):")
            time.sleep(0.5)
            while True:
                line = input()
                if line.strip().lower() == "done":
                    break
                prompt += line + "\n"

            line_print("Parsing prompt...")
            parser = PromptBasedInstructionParser(task_type=TaskType.TEXT_GENERATION)
            parser.parse_from_prompt(prompt)
            propmt_has_been_parsed = True
            status["instruction"] = parser.instruction
            status["examples"] = parser.examples
            status["prompt_has_been_parsed"] = True
            with open("status.yaml", "w") as f:
                yaml.safe_dump(status, f)
            # 归档到 run_dir
            with open(run_dir / "prompt.from_cli.txt", "w", encoding="utf-8") as f:
                f.write(prompt)

    # ===== 2) 检索数据（失败则回退到 few-shot）=====
    if propmt_has_been_parsed and not dataset_has_been_retrieved:
        retriever_logger = get_formatted_logger("DescriptionDatasetRetriever")
        retriever_logger.setLevel(logging.INFO)

        prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION, status["instruction"], status["examples"])
        line_print("Retrieving dataset...")
        line_print("Do you want to perform data transformation? (y/n)")
        line_print("Data transformation converts retrieved data into the desired format as per the prompt.")
        auto_transform_data = False
        total_num_points_to_transform = None
        while True:
            line = input().strip()
            if line.lower() == "y":
                auto_transform_data = True
                break
            elif line.lower() == "n":
                auto_transform_data = False
                break
            else:
                line_print("Invalid input. Please enter y or n.")

        if auto_transform_data:
            while True:
                line_print("Enter the number of data points you want to transform (the remaining will be discarded):")
                line = input().strip()
                try:
                    total_num_points_to_transform = int(line)
                except ValueError:
                    line_print("Invalid input. Please enter a number.")
                    continue
                if total_num_points_to_transform <= 0:
                    line_print("Invalid input. Please enter a number greater than 0.")
                    continue
                status["num_transform"] = total_num_points_to_transform
                break

        try:
            retriever = DescriptionDatasetRetriever(
                auto_transform_data=auto_transform_data,
                total_num_points_to_transform=total_num_points_to_transform,
            )
            retrieved_dataset_dict = retriever.retrieve_dataset_dict(prompt_spec)
            if retrieved_dataset_dict is None:
                raise RuntimeError("retriever returned None")
            line_print("[OK] remote retrieval succeeded.")
        except Exception as e:
            line_print(f"[WARN] remote retrieval init/fetch failed: {e}")
            line_print("[fallback] building dataset from prompt examples.")

            exs = status.get("examples", [])
            if not isinstance(exs, list):
                exs = []
            if auto_transform_data and total_num_points_to_transform is not None and len(exs) > total_num_points_to_transform:
                exs = exs[: total_num_points_to_transform]

            inputs  = [e.get("input", "")  for e in exs]
            outputs = [e.get("output", "") for e in exs]
            labels  = [e.get("label", "prompt_example") for e in exs]
            sources = ["prompt_example"] * len(inputs)

            from datasets import Dataset, DatasetDict
            retrieved_dataset_dict = DatasetDict({
                "train": Dataset.from_dict({
                    "input": inputs,
                    "output": outputs,
                    "label": labels,
                    "source": sources,
                })
            })

        # 保存到本次 run 目录
        ret_root = run_dir / "retrieved_dataset_dict"
        retrieved_dataset_dict.save_to_disk(str(ret_root))
        status["retrieved_dataset_dict_root"] = str(ret_root)
        dataset_has_been_retrieved = True
        status["dataset_has_been_retrieved"] = True
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)
        # 归档一份 status 到 run_dir
        with open(run_dir / "status.snapshot.yaml", "w") as f:
            yaml.safe_dump(status, f)

    # ===== 3) 合成数据（本地模型），并安全落盘 =====
    if propmt_has_been_parsed and dataset_has_been_retrieved and not dataset_has_been_generated:
        prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION, status["instruction"], status["examples"])
        generator_logger = get_formatted_logger("DatasetGenerator")
        generator_logger.setLevel(logging.INFO)

        line_print("The dataset generation has not finished.")
        time.sleep(0.3)
        line_print(f"Your input instruction:\n\n{prompt_spec.instruction}")
        time.sleep(0.3)
        line_print(f"Your input few-shot examples count: {len(prompt_spec.examples) if isinstance(prompt_spec.examples, list) else 'N/A'}")
        time.sleep(0.3)

        while True:
            line_print("Enter the number of examples you wish to generate (you can enter 0 to skip generation):")
            line = input().strip()
            try:
                num_expected = int(line)
                break
            except ValueError:
                line_print("Invalid input. Please enter a number.")

        # 默认更“去重友好”的超参（可通过环境变量覆盖）
        def _get_float_env(name, default):
            try:
                return float(os.getenv(name, default))
            except Exception:
                return default

        def _get_int_env(name, default):
            try:
                return int(os.getenv(name, default))
            except Exception:
                return default

        while True:
            line_print(f"Enter the initial temperature (default {os.getenv('P2M_INIT_TEMP','0.4')}):")
            line = input().strip()
            if line == "":
                initial_temperature = _get_float_env("P2M_INIT_TEMP", 0.4)
                break
            try:
                initial_temperature = float(line)
                assert 0 <= initial_temperature <= 2.0
                break
            except Exception:
                line_print("Invalid initial temperature. Enter float between 0 and 2.")

        while True:
            line_print(f"Enter the max temperature (default {os.getenv('P2M_MAX_TEMP','1.2')}):")
            line = input().strip()
            if line == "":
                max_temperature = _get_float_env("P2M_MAX_TEMP", 1.2)
                break
            try:
                max_temperature = float(line)
                assert 0 <= max_temperature <= 2.0
                break
            except Exception:
                line_print("Invalid max temperature. Enter float between 0 and 2.")

        # —— 重复度控制的关键参数（可调环境变量）——
        RESP_PER_REQ = _get_int_env("P2M_RESPONSES_PER_REQUEST", 1)          # 设 1 更稳妥
        PRES_PEN     = _get_float_env("P2M_PRESENCE_PENALTY", 0.6)           # 0.3~0.8 合理
        FREQ_PEN     = _get_float_env("P2M_FREQUENCY_PENALTY", 0.6)          # 0.3~0.8 合理
        MAX_BATCH    = _get_int_env("P2M_MAX_BATCH_SIZE", 3)                 # 小 batch 有助于多样性
        RPM          = _get_int_env("P2M_REQUESTS_PER_MINUTE", 80)

        gen_root = run_dir / "generated_dataset"

        if num_expected > 0:
            line_print("Starting to generate dataset. This may take a while...")
            time.sleep(0.3)
            unlimited_dataset_generator = PromptBasedDatasetGenerator(
                initial_temperature=initial_temperature,
                max_temperature=max_temperature,
                responses_per_request=RESP_PER_REQ,
                presence_penalty=PRES_PEN,
                frequency_penalty=FREQ_PEN,
                max_batch_size=MAX_BATCH,
                requests_per_minute=RPM,
                filter_duplicated_examples=True,
            )
            generated_dataset = unlimited_dataset_generator.generate_dataset_split(
                prompt_spec, num_expected, split=DatasetSplit.TRAIN
            )
            # 收尾再去重一遍（保险）
            generated_dataset = _post_dedup_dataset(generated_dataset)
            # 安全落盘（目录唯一 run_dir 保证不覆盖）
            generated_dataset.save_to_disk(str(gen_root))
            # 旁路导出 JSONL/CSV
            _export_sidecars(generated_dataset, gen_root / "generated_dataset")
        else:
            # 仍写入空占位（保持流程一致）
            gen_root.mkdir(parents=True, exist_ok=True)
            empty = datasets.Dataset.from_dict({"input": [], "output": []})
            empty.save_to_disk(str(gen_root))
            _export_sidecars(empty, gen_root / "generated_dataset")

        dataset_has_been_generated = True
        status["dataset_has_been_generated"] = True
        status["generated_dataset_root"] = str(gen_root)
        status["run_dir"] = str(run_dir)
        with open("status.yaml", "w") as f:
            yaml.safe_dump(status, f)
        with open(run_dir / "status.snapshot.yaml", "w") as f:
            yaml.safe_dump(status, f)
        # 记录一次 manifest，方便复现实验
        manifest = {
            "run_dir": str(run_dir.resolve()),
            "num_expected": num_expected,
            "initial_temperature": initial_temperature,
            "max_temperature": max_temperature,
            "responses_per_request": RESP_PER_REQ,
            "presence_penalty": PRES_PEN,
            "frequency_penalty": FREQ_PEN,
            "max_batch_size": MAX_BATCH,
            "requests_per_minute": RPM,
            "backend": os.getenv("P2M_BACKEND"),
            "gen_model": os.getenv("P2M_GEN_MODEL"),
            "embed_model": os.getenv("P2M_OLLAMA_EMBED_MODEL"),
            "timestamp": run_id,
        }
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        line_print("Data step finished. Now visualizing clusters...")

        # ===== 4) 聚类可视化（合成 + 检索合并）=====
        gen_df = _flatten_generated_to_df(status.get("generated_dataset_root"))
        ret_df = _flatten_retrieved_to_df(status.get("retrieved_dataset_dict_root"))
        mix_df = pd.concat([ret_df, gen_df], ignore_index=True)
        if len(mix_df) == 0:
            line_print("No data to visualize (both retrieved and generated are empty). Exit.")
            return

        ollama_model = os.getenv("P2M_OLLAMA_EMBED_MODEL", "nomic-embed-text")
        mix_emb = _embed_with_ollama(mix_df["__text__"].values, model=ollama_model)

        # 保存嵌入与元数据
        np.save(run_dir / "embeddings.npy", mix_emb)
        mix_df.to_csv(run_dir / "mix_meta.csv", index=False, encoding="utf-8")

        # 降维绘图（落盘到本次 run 目录）
        dr_method = os.getenv("P2M_DR_METHOD", "umap")
        out_path  = run_dir / os.getenv("P2M_OUT", "synth_clusters.png")
        color_col = "label"
        _reduce_and_plot(mix_emb, mix_df, color_col=color_col, method=dr_method, out_path=str(out_path))
        line_print(f"[DONE] everything saved under: {run_dir.resolve()}")
        return

    # ===== 已生成过数据：直接可视化一次并退出 =====
    run_dir_from_status = Path(status.get("run_dir", run_dir))
    gen_df = _flatten_generated_to_df(status.get("generated_dataset_root"))
    ret_df = _flatten_retrieved_to_df(status.get("retrieved_dataset_dict_root"))
    mix_df = pd.concat([ret_df, gen_df], ignore_index=True)
    if len(mix_df) == 0:
        line_print("No data to visualize (both retrieved and generated are empty). Exit.")
        return

    ollama_model = os.getenv("P2M_OLLAMA_EMBED_MODEL", "nomic-embed-text")
    mix_emb = _embed_with_ollama(mix_df["__text__"].values, model=ollama_model)
    np.save(run_dir_from_status / "embeddings.npy", mix_emb)
    mix_df.to_csv(run_dir_from_status / "mix_meta.csv", index=False, encoding="utf-8")
    dr_method = os.getenv("P2M_DR_METHOD", "umap")
    out_path  = run_dir_from_status / os.getenv("P2M_OUT", "synth_clusters.png")
    color_col = os.getenv("P2M_COLOR_COL", "label")
    _reduce_and_plot(mix_emb, mix_df, color_col=color_col, method=dr_method, out_path=str(out_path))
    line_print(f"[DONE] figure saved to: {out_path.resolve()}")
    return


if __name__ == "__main__":
    main()
