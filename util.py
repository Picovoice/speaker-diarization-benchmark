from collections import defaultdict
from typing import Sequence, Dict, Tuple

import soundfile as sf
from pyannote.core import Segment, Annotation

RTTM = Dict[str, Sequence[Tuple[str, float, float]]]


def load_rttm(file: str) -> RTTM:
    rttm = defaultdict(list)
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split()
            file_id = parts[1]
            spk = parts[7]
            start = float(parts[3])
            end = start + float(parts[4])
            rttm[file_id].append((spk, start, end))
    return rttm


def rttm_to_annotation(rttm: RTTM) -> "Annotation":
    reference = Annotation()
    segments = list(rttm.values())[0]
    for segment in segments:
        label, start, end = segment
        reference[Segment(start, end)] = label
    return reference


def get_audio_length(file: str) -> float:
    data, samplerate = sf.read(file)
    return len(data) / samplerate


__all__ = [
    "RTTM",
    "load_rttm",
    "rttm_to_annotation",
    "get_audio_length",
]
