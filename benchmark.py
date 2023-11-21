import argparse
import math
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

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
import random

WorkerResult = namedtuple(
    'WorkerResult',
    [
        'total_audio_sec',
        'process_time_per_core_sec',
    ])


def process_core_hour(
        engine_type: Engines,
        engine_params: Dict[str, Any],
        samples: Sequence[Tuple[str, str, float]]) -> WorkerResult:
    print("Creating engine...")
    engine = Engine.create(engine_type, **engine_params)
    print("Engine created.")
    total_audio_sec = 0

    tic = perf_counter()
    for sample in samples:
        audio_path, _, audio_length = sample
        print(f"Processing {audio_path}...")
        total_audio_sec += audio_length
        _ = engine.diarization(audio_path)

    toc = perf_counter()

    process_time = toc - tic
    print(f"Total audio time: {total_audio_sec} sec")
    # engine.cleanup()
    return WorkerResult(total_audio_sec, process_time)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[ds.value for ds in Datasets], required=True)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--label-folder", required=True)
    parser.add_argument("--engine", choices=[en.value for en in Engines], required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--only-rtf", action="store_true")
    parser.add_argument("--rtf-num-examples", type=int, default=None)
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
    args = parser.parse_args()

    dataset = Dataset.create(Datasets(args.dataset), data_folder=args.data_folder, label_folder=args.label_folder)
    print(f"Dataset: {dataset}")

    kwargs_engine = dict()
    if args.engine == Engines.PICOVOICE_FALCON.value:
        if args.picovoice_access_key is None:
            raise ValueError(f"Engine {args.engine} requires --picovoice-access-key")
        kwargs_engine.update(access_key=args.picovoice_access_key)
    elif args.engine == Engines.PYANNOTE.value:
        if args.pyannote_auth_token is None:
            raise ValueError(f"Engine {args.engine} requires --pyannote-auth-token")
        kwargs_engine.update(auth_token=args.pyannote_auth_token)
    elif args.engine == Engines.AWS_TRANSCRIBE.value:
        if args.aws_profile is None:
            raise ValueError(f"Engine {args.engine} requires --aws-profile")
        # TODO:
        # os.environ["AWS_PROFILE"] = args.aws_profile
        if args.aws_s3_bucket_name is None:
            raise ValueError(f"Engine {args.engine} requires --aws-s3-bucket-name")
        kwargs_engine.update(bucket_name=args.aws_s3_bucket_name)
    elif args.engine in [Engines.GOOGLE_SPEECH_TO_TEXT.value, Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED.value]:
        if args.gcp_credentials is None:
            raise ValueError(f"Engine {args.engine} requires --gcp-credentials")
        # TODO:
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.gcp_credentials
        if args.gcp_bucket_name is None:
            raise ValueError(f"Engine {args.engine} requires --gcp-bucket-name")
        kwargs_engine.update(bucket_name=args.gcp_bucket_name)
    elif args.engine == Engines.AZURE_SPEECH_TO_TEXT.value:
        if args.azure_storage_account_name is None:
            raise ValueError(f"Engine {args.engine} requires --azure-storage-account-name")
        if args.azure_storage_account_key is None:
            raise ValueError(f"Engine {args.engine} requires --azure-storage-account-key")
        if args.azure_storage_container_name is None:
            raise ValueError(f"Engine {args.engine} requires --azure-storage-container-name")
        if args.azure_subscription_key is None:
            raise ValueError(f"Engine {args.engine} requires --azure-subscription-key")
        if args.azure_region is None:
            raise ValueError(f"Engine {args.engine} requires --azure-region")

        kwargs_engine.update(
            storage_account_name=args.azure_storage_account_name,
            storage_account_key=args.azure_storage_account_key,
            storage_container_name=args.azure_storage_container_name,
            subscription_key=args.azure_subscription_key,
            region=args.azure_region)

    engine = Engine.create(Engines(args.engine), **kwargs_engine)
    print(f"Engine: {engine}")

    results_folder = os.path.join(RESULTS_FOLDER, args.dataset)
    if not args.only_rtf:

        metric_der = DiarizationErrorRate(detailed=True, skip_overlap=True)
        metric_jer = JaccardErrorRate(detailed=True, skip_overlap=True)
        metrics = [metric_der, metric_jer]

        cache_folder = os.path.join(DEFAULT_CACHE_FOLDER, args.dataset, args.engine)
        os.makedirs(cache_folder, exist_ok=True)

        try:
            for index in tqdm(range(dataset.size)):
                audio_path, audio_length, ground_truth = dataset.get(index)
                if args.verbose:
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
                    if args.verbose:
                        print(f"{metric.name}: {res}")
        except KeyboardInterrupt as e:
            print("Stopping benchmark...")
        finally:
            engine.cleanup()

        results = dict()
        for metric in metrics:
            results[metric.name] = abs(metric)
        results_path = os.path.join(results_folder, f"{args.engine}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        results_details_path = os.path.join(results_folder, f"{args.engine}.log")
        with open(results_details_path, "w") as f:
            for metric in metrics:
                f.write(f"{metric.name}: {str(metric)}")
                f.write("\n")

    if engine.is_offline():
        # num_workers = psutil.cpu_count(logical=False)
        num_workers = 2
        # if engine.is_single_threaded():
        #     num_workers = psutil.cpu_count(logical=False)
        # else:
        #     num_workers = 1

        print(f"Number of workers: {num_workers}")
        samples = list(dataset.samples[:])
        # randomize the order of samples
        random.shuffle(samples)
        if args.rtf_num_examples is not None:
            samples = samples[:args.rtf_num_examples]

        chunk = math.floor(len(samples) / num_workers)
        futures = list()

        # NOTE: this is required for the ``fork`` method to work
        # processes = []
        # for rank in range(num_workers):
        #     p = mp.Process(target=process_core_hour,
        #                     args=(Engines(args.engine), kwargs_engine, samples[rank * chunk:(rank + 1) * chunk]))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
        with ProcessPoolExecutor(num_workers) as executor:
            for i in range(num_workers):
                future = executor.submit(
                    process_core_hour,
                    engine_type=Engines(args.engine),
                    engine_params=kwargs_engine,
                    samples=samples[i * chunk:(i + 1) * chunk]
                )
                futures.append(future)

        res = [x.result() for x in futures]
        total_audio_sec = sum([x.total_audio_sec for x in res])
        total_processing_sec = sum([x.process_time_per_core_sec for x in res])

        # if num_workers > 1:
        #     with ProcessPoolExecutor(num_workers) as executor:
        #         for i in range(num_workers):
        #             future = executor.submit(
        #                 process_core_hour,
        #                 engine_type=Engines(args.engine),
        #                 engine_params=kwargs_engine,
        #                 samples=samples[i * chunk:(i + 1) * chunk]
        #             )
        #             futures.append(future)
        #
        #     res = [x.result() for x in futures]
        #     total_audio_sec = sum([x.total_audio_sec for x in res])
        #     total_processing_sec = sum([x.process_time_per_core_sec for x in res])
        # else:
        #     res = process_core_hour(
        #         engine_type=Engines(args.engine),
        #         engine_params=kwargs_engine,
        #         samples=samples,
        #     )
        #     total_audio_sec = res.total_audio_sec
        #     total_processing_sec = res.process_time_per_core_sec

        results_path = os.path.join(results_folder, f"{args.engine}_rtf.json")
        results = dict()
        results["num_cpu"] = os.cpu_count()
        results["total_audio_sec"] = total_audio_sec
        results["total_processing_sec"] = total_processing_sec
        results["core_hours"] = total_processing_sec / 3600
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

__all__ = [
    "RESULTS_FOLDER",
]
