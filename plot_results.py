import argparse
import json
import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[ds.value for ds in Datasets], required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    dataset_name = args.dataset
    engines_results = dict()
    sorted_engines = sorted(ENGINES, key=lambda e: (ENGINE_ORDER_KEYS.get(e, 1), ENGINE_PRINT_NAMES.get(e, e.value)))
    for engine_type in sorted_engines:
        results_path = os.path.join(RESULTS_FOLDER, dataset_name, engine_type.value + ".json")
        if not os.path.exists(results_path):
            raise ValueError(f"No results file for engine `{engine_type.value}` on dataset `{dataset_name}`")

        with open(results_path, "r") as f:
            results_json = json.load(f)

        engines_results[engine_type] = results_json

    save_path = os.path.join(RESULTS_FOLDER, "plots")
    os.makedirs(save_path, exist_ok=True)

    for metric in METRIC_NAME:
        fig, ax = plt.subplots(figsize=(6, 4))
        for engine_type in sorted_engines:
            engine_value = engines_results[engine_type][metric] * 100
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
        plot_path = os.path.join(save_path, dataset_name, metric.replace(" ", "_") + ".png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)

        if args.show:
            plt.show()

        plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    max_process_time = 0
    for engine_type in sorted_engines:
        engine_value = engines_results[engine_type].get("realtime factor")
        if engine_value is None:
            continue
        process_time = 60 * engine_value
        max_process_time = max(max_process_time, process_time)
        ax.bar(
            ENGINE_PRINT_NAMES.get(engine_type, engine_type.value),
            process_time,
            width=0.5,
            color=ENGINE_COLORS.get(engine_type, WHITE),
            edgecolor="none",
            label=ENGINE_PRINT_NAMES.get(engine_type, engine_type.value),
        )
        ax.text(
            ENGINE_PRINT_NAMES.get(engine_type, engine_type.value),
            process_time + 0.1,
            f"{process_time:.1f} min",
            ha="center",
            va="bottom",
            fontsize=12,
            color=ENGINE_COLORS.get(engine_type, BLACK),
        )

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])
    plt.ylim([0, max_process_time + 10])
    plt.title("Process time for an hour of audio", fontsize=12)
    plt.savefig(os.path.join(save_path, "realtime_factor_comparison.png"))

    if args.show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()

__all__ = [
    "Color",
    "plot_results",
    "rgb_from_hex",
]
