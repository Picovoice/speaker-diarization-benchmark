import argparse
import json
import os
from typing import *

import matplotlib.pyplot as plt

from benchmark import RESULTS_FOLDER
from dataset import Datasets
from engine import Engines

Color = Tuple[float, float, float]


def rgb_from_hex(x: str) -> Color:
    x = x.strip("# ")
    assert len(x) == 6
    return int(x[:2], 16) / 255, int(x[2:4], 16) / 255, int(x[4:], 16) / 255


BLACK = rgb_from_hex("#000000")
GREY1 = rgb_from_hex("#3F3F3F")
GREY2 = rgb_from_hex("#5F5F5F")
GREY3 = rgb_from_hex("#7F7F7F")
GREY4 = rgb_from_hex("#9F9F9F")
GREY5 = rgb_from_hex("#BFBFBF")
WHITE = rgb_from_hex("#FFFFFF")
BLUE = rgb_from_hex("#377DFF")

ENGINES = [
    Engines.AWS_TRANSCRIBE,
    Engines.AZURE_SPEECH_TO_TEXT,
    Engines.GOOGLE_SPEECH_TO_TEXT,
    Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED,
    Engines.PICOVOICE_FALCON,
    Engines.PYANNOTE,
]

ENGINE_ORDER_KEYS = {
    Engines.AWS_TRANSCRIBE: 1,
    Engines.AZURE_SPEECH_TO_TEXT: 2,
    Engines.GOOGLE_SPEECH_TO_TEXT: 3,
    Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED: 4,
    Engines.PICOVOICE_FALCON: 5,
    Engines.PYANNOTE: 6,
}

ENGINE_COLORS = {
    Engines.AWS_TRANSCRIBE: GREY5,
    Engines.AZURE_SPEECH_TO_TEXT: GREY4,
    Engines.GOOGLE_SPEECH_TO_TEXT: GREY3,
    Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED: GREY2,
    Engines.PICOVOICE_FALCON: BLUE,
    Engines.PYANNOTE: GREY1,
}

ENGINE_PRINT_NAMES = {
    Engines.AWS_TRANSCRIBE: "Amazon",
    Engines.AZURE_SPEECH_TO_TEXT: "Azure",
    Engines.GOOGLE_SPEECH_TO_TEXT: "Google",
    Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED: "Google\nEnhanced",
    Engines.PICOVOICE_FALCON: "Picovoice\nFalcon",
    Engines.PYANNOTE: "pyannote",
}

METRIC_NAME = [
    "diarization error rate",
    "jaccard error rate",
]


def _plot_accuracy(
        engine_list: List[Engines],
        result_path: str,
        save_path: str,
        show: bool) -> None:
    for metric in METRIC_NAME:
        fig, ax = plt.subplots(figsize=(6, 4))
        for engine_type in engine_list:
            engine_result_path = os.path.join(result_path, engine_type.value + ".json")
            if not os.path.exists(engine_result_path):
                continue

            with open(engine_result_path, "r") as f:
                results_json = json.load(f)

            engine_value = results_json[metric] * 100
            ax.bar(
                ENGINE_PRINT_NAMES.get(engine_type, engine_type.value),
                engine_value,
                width=0.5,
                color=ENGINE_COLORS.get(engine_type, WHITE),
                edgecolor="none",
                label=ENGINE_PRINT_NAMES.get(engine_type, engine_type.value),
            )
            ax.text(
                ENGINE_PRINT_NAMES.get(engine_type, engine_type.value),
                engine_value + 1,
                f"{engine_value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                color=ENGINE_COLORS.get(engine_type, BLACK),
            )

        ax.set_ylabel(metric.title(), fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        plot_path = os.path.join(save_path, metric.replace(" ", "_") + ".png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)

        if show:
            plt.show()

        plt.close()


def _plot_cpu(
        engine_list: List[Engines],
        result_path: str,
        save_path: str,
        show: bool) -> None:
    engines_results_cpu = dict()
    for engine_type in engine_list:
        engine_result_path = os.path.join(result_path, engine_type.value + "_cpu.json")
        if not os.path.exists(engine_result_path):
            continue

        with open(engine_result_path, "r") as f:
            results_json = json.load(f)

        engines_results_cpu[engine_type] = results_json

    fig, ax = plt.subplots(figsize=(6, 4))
    xlim = 0
    for engine_type, engine_value in engines_results_cpu.items():
        processing_capacity = engine_value["total_audio_time_sec"] / engine_value["total_process_time_sec"]
        xlim = max(xlim, processing_capacity)
        ax.barh(
            ENGINE_PRINT_NAMES.get(engine_type, engine_type.value),
            processing_capacity,
            # width=0.5,
            color=ENGINE_COLORS.get(engine_type, WHITE),
            edgecolor="none",
            label=ENGINE_PRINT_NAMES.get(engine_type, engine_type.value),
        )
        ax.text(
            processing_capacity + 3,
            ENGINE_PRINT_NAMES.get(engine_type, engine_type.value),
            f"{processing_capacity:.2f}\nhours",
            ha="center",
            va="center",
            fontsize=12,
            color=ENGINE_COLORS.get(engine_type, BLACK),
        )

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlim([0, xlim + 5])
    ax.set_xticks([])
    plt.title("Audio Processing Capacity per Core-Hour", fontsize=12)
    plt.savefig(os.path.join(save_path, "cpu_usage_comparison.png"))

    if show:
        plt.show()

    plt.close()


def _plot_mem(
        engine_list: List[Engines],
        result_path: str,
        save_path: str,
        show: bool) -> None:
    engines_results_mem = dict()

    for engine_type in engine_list:
        engine_result_path = os.path.join(result_path, engine_type.value + "_mem.json")
        if not os.path.exists(engine_result_path):
            continue

        with open(engine_result_path, "r") as f:
            results_json = json.load(f)

        engines_results_mem[engine_type] = results_json

    fig, ax = plt.subplots(figsize=(6, 4))
    xlim = 0
    for engine_type, engine_value in engines_results_mem.items():
        max_mem_usage = engine_value["max_mem_GiB"]
        xlim = max(xlim, max_mem_usage)
        ax.barh(
            ENGINE_PRINT_NAMES.get(engine_type, engine_type.value),
            max_mem_usage,
            # width=0.5,
            color=ENGINE_COLORS.get(engine_type, WHITE),
            edgecolor="none",
            label=ENGINE_PRINT_NAMES.get(engine_type, engine_type.value),
        )
        ax.text(
            max_mem_usage + 0.3,
            ENGINE_PRINT_NAMES.get(engine_type, engine_type.value),
            f"{max_mem_usage:.2f}GiB",
            ha="center",
            va="center",
            fontsize=12,
            color=ENGINE_COLORS.get(engine_type, BLACK),
        )

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    plt.xlim([0, xlim + 1])
    plt.title("Maximum Memory Usage", fontsize=12)
    plt.savefig(os.path.join(save_path, "mem_usage_comparison.png"))

    if show:
        plt.show()

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[ds.value for ds in Datasets], required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    dataset_name = args.dataset
    sorted_engines = sorted(ENGINES, key=lambda e: (ENGINE_ORDER_KEYS.get(e, 1), ENGINE_PRINT_NAMES.get(e, e.value)))

    save_path = os.path.join(RESULTS_FOLDER, "plots")
    os.makedirs(save_path, exist_ok=True)

    result_dataset_path = os.path.join(RESULTS_FOLDER, dataset_name)
    _plot_accuracy(sorted_engines, result_dataset_path, os.path.join(save_path, dataset_name), args.show)
    _plot_mem(sorted_engines, result_dataset_path, save_path, args.show)
    _plot_cpu(sorted_engines, result_dataset_path, save_path, args.show)


if __name__ == "__main__":
    main()

__all__ = [
    "Color",
    "plot_results",
    "rgb_from_hex",
]
