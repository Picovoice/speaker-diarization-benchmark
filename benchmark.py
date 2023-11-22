import argparse
import math
import os
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import psutil
from pyannote.metrics.diarization import (
    DiarizationErrorRate,
    JaccardErrorRate,
)
from tqdm import tqdm

from dataset import *
from engine import *
from util import load_rttm, rttm_to_annotation

DEFAULT_CACHE_FOLDER = os.path.join(os.path.dirname(__file__), "cache")
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")


class BenchmarkTypes(Enum):
    ACCURACY = "ACCURACY"
    CPU = "CPU"
    MEMORY = "MEMORY"


def _engine_params_parser(in_args: argparse.Namespace) -> Dict[str, Any]:
    kwargs_engine = dict()
    if in_args.engine == Engines.PICOVOICE_FALCON.value:
        if in_args.picovoice_access_key is None:
            raise ValueError(f"Engine {in_args.engine} requires --picovoice-access-key")
        kwargs_engine.update(access_key=in_args.picovoice_access_key)
    elif in_args.engine == Engines.PYANNOTE.value:
        if in_args.pyannote_auth_token is None:
            raise ValueError(f"Engine {in_args.engine} requires --pyannote-auth-token")
        kwargs_engine.update(auth_token=in_args.pyannote_auth_token)
    elif in_args.engine == Engines.AWS_TRANSCRIBE.value:
        if in_args.aws_profile is None:
            raise ValueError(f"Engine {in_args.engine} requires --aws-profile")
        os.environ["AWS_PROFILE"] = in_args.aws_profile
        if in_args.aws_s3_bucket_name is None:
            raise ValueError(f"Engine {in_args.engine} requires --aws-s3-bucket-name")
        kwargs_engine.update(bucket_name=in_args.aws_s3_bucket_name)
    elif in_args.engine in [Engines.GOOGLE_SPEECH_TO_TEXT.value, Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED.value]:
        if in_args.gcp_credentials is None:
            raise ValueError(f"Engine {in_args.engine} requires --gcp-credentials")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = in_args.gcp_credentials
        if in_args.gcp_bucket_name is None:
            raise ValueError(f"Engine {in_args.engine} requires --gcp-bucket-name")
        kwargs_engine.update(bucket_name=in_args.gcp_bucket_name)
    elif in_args.engine == Engines.AZURE_SPEECH_TO_TEXT.value:
        if in_args.azure_storage_account_name is None:
            raise ValueError(f"Engine {in_args.engine} requires --azure-storage-account-name")
        if in_args.azure_storage_account_key is None:
            raise ValueError(f"Engine {in_args.engine} requires --azure-storage-account-key")
        if in_args.azure_storage_container_name is None:
            raise ValueError(f"Engine {in_args.engine} requires --azure-storage-container-name")
        if in_args.azure_subscription_key is None:
            raise ValueError(f"Engine {in_args.engine} requires --azure-subscription-key")
        if in_args.azure_region is None:
            raise ValueError(f"Engine {in_args.engine} requires --azure-region")

        kwargs_engine.update(
            storage_account_name=in_args.azure_storage_account_name,
            storage_account_key=in_args.azure_storage_account_key,
            storage_container_name=in_args.azure_storage_container_name,
            subscription_key=in_args.azure_subscription_key,
            region=in_args.azure_region)

    return kwargs_engine


def _process_accuracy(engine: Engine, dataset: Dataset, verbose: bool = False) -> None:
    metric_der = DiarizationErrorRate(detailed=True, skip_overlap=True)
    metric_jer = JaccardErrorRate(detailed=True, skip_overlap=True)
    metrics = [metric_der, metric_jer]

    cache_folder = os.path.join(DEFAULT_CACHE_FOLDER, str(dataset), str(engine))
    print(f"Cache folder: {cache_folder}")
    os.makedirs(cache_folder, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_FOLDER, str(dataset)), exist_ok=True)
    try:
        for index in tqdm(range(dataset.size)):
            audio_path, audio_length, ground_truth = dataset.get(index)
            if verbose:
                print(f"Processing {audio_path}...")

            cache_path = os.path.join(cache_folder, f"{os.path.basename(audio_path)}_cached.rttm")

            if os.path.exists(cache_path):
                hypothesis = rttm_to_annotation(load_rttm(cache_path))
            else:
                hypothesis = engine.diarization(audio_path)

                with open(cache_path, "w") as f:
                    f.write(hypothesis.to_rttm())

            for metric in metrics:
                res = metric(ground_truth, hypothesis, detailed=True)
                if verbose:
                    print(f"{metric.name}: {res}")
    except KeyboardInterrupt as e:
        print("Stopping benchmark...")

    results = dict()
    for metric in metrics:
        results[metric.name] = abs(metric)
    results_path = os.path.join(RESULTS_FOLDER, str(dataset), f"{str(engine)}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    results_details_path = os.path.join(RESULTS_FOLDER, str(dataset), f"{str(engine)}.log")
    with open(results_details_path, "w") as f:
        for metric in metrics:
            f.write(f"{metric.name}:\n{str(metric)}")
            f.write("\n")


WorkerResult = namedtuple(
    'WorkerResult',
    [
        'total_audio_sec',
        'process_time_sec',
    ])


def _process_worker(
        engine_type: str,
        engine_params: Dict[str, Any],
        samples: Sequence[Tuple[str, str, float]]) -> WorkerResult:
    engine = Engine.create(Engines(engine_type), **engine_params)
    total_audio_sec = 0
    process_time = 0

    for sample in samples:
        audio_path, _, audio_length = sample
        total_audio_sec += audio_length
        tic = perf_counter()
        _ = engine.diarization(audio_path)
        toc = perf_counter()
        process_time += (toc - tic)

    engine.cleanup()
    return WorkerResult(total_audio_sec, process_time)


def _process_cpu_process_pool(
        engine: str,
        engine_params: Dict[str, Any],
        dataset: Dataset,
        num_samples: Optional[int] = None) -> None:
    num_workers = os.cpu_count()

    samples = list(dataset.samples[:])
    if num_samples is not None:
        samples = samples[:num_samples]

    chunk_size = math.floor(len(samples) / num_workers)
    futures = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            chunk = samples[i * chunk_size: (i + 1) * chunk_size]
            future = executor.submit(
                _process_worker,
                engine_type=engine,
                engine_params=engine_params,
                samples=chunk)
            futures.append(future)

    res = [f.result() for f in futures]
    total_audio_time_sec = sum([r.total_audio_sec for r in res])
    total_process_time_sec = sum([r.process_time_sec for r in res])

    results_path = os.path.join(RESULTS_FOLDER, str(dataset), f"{str(engine)}_cpu.json")
    results = {
        "total_audio_time_sec": total_audio_time_sec,
        "total_process_time_sec": total_process_time_sec,
        "num_workers": num_workers,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[ds.value for ds in Datasets], required=True)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--label-folder", required=True)
    parser.add_argument("--engine", choices=[en.value for en in Engines], required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--aws-profile")
    parser.add_argument("--aws-s3-bucket-name")
    parser.add_argument("--azure-region")
    parser.add_argument("--azure-storage-account-key")
    parser.add_argument("--azure-storage-account-name")
    parser.add_argument("--azure-storage-container-name")
    parser.add_argument("--azure-subscription-key")
    parser.add_argument("--gcp-bucket-name")
    parser.add_argument("--gcp-credentials")
    parser.add_argument("--nemo-model-config")
    parser.add_argument("--picovoice-access-key")
    parser.add_argument("--pyannote-auth-token")
    parser.add_argument("--type", choices=[bt.value for bt in BenchmarkTypes], required=True)
    parser.add_argument("--num-samples", type=int)
    args = parser.parse_args()

    engine_args = _engine_params_parser(args)

    dataset = Dataset.create(Datasets(args.dataset), data_folder=args.data_folder, label_folder=args.label_folder)
    print(f"Dataset: {dataset}")

    engine = Engine.create(Engines(args.engine), **engine_args)
    print(f"Engine: {engine}")

    if args.type == BenchmarkTypes.ACCURACY.value:
        _process_accuracy(engine, dataset, verbose=args.verbose)
    elif args.type == BenchmarkTypes.CPU.value:
        if not engine.is_offline():
            raise ValueError(f"CPU benchmark is only supported for offline engines")
        _process_cpu_process_pool(
            engine=args.engine,
            engine_params=engine_args,
            dataset=dataset,
            num_samples=args.num_samples)
    elif args.type == BenchmarkTypes.MEMORY.value:
        if not engine.is_offline():
            raise ValueError(f"Memory benchmark is only supported for offline engines")
        print("Please make sure the `mem_monitor.py` script is running and then press enter to continue...")
        input()
        _process_cpu_process_pool(
            engine=args.engine,
            engine_params=engine_args,
            dataset=dataset,
            num_samples=args.num_samples)


if __name__ == "__main__":
    main()

__all__ = [
    "RESULTS_FOLDER",
]
