import argparse
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

    metric_der = DiarizationErrorRate(detailed=True, skip_overlap=True)
    metric_jer = JaccardErrorRate(detailed=True, skip_overlap=True)
    metrics = [metric_der, metric_jer]

    cache_folder = os.path.join(DEFAULT_CACHE_FOLDER, args.dataset, args.engine)
    os.makedirs(cache_folder, exist_ok=True)

    total_audio_length = 0
    total_processing_time = 0
    try:
        for index in tqdm(range(dataset.size())):
            audio_path, audio_length, ground_truth = dataset.get(index)
            if args.verbose:
                print(f"Processing {audio_path}...")
            total_audio_length += audio_length

            cache_path = os.path.join(cache_folder, f"{os.path.basename(audio_path)}_cached.rttm")
            rtf_path = os.path.join(cache_folder, f"{os.path.basename(audio_path)}_cached.rtf")

            if os.path.exists(cache_path):
                hypothesis = rttm_to_annotation(load_rttm(cache_path))
                if os.path.exists(rtf_path):
                    with open(rtf_path) as f:
                        total_processing_time += float(f.read())
            else:
                if engine.is_offline():
                    tic = perf_counter()
                    hypothesis = engine.diarization(audio_path)
                    toc = perf_counter()
                    total_processing_time += toc - tic
                    with open(rtf_path, "w") as f:
                        f.write(str(toc - tic))
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

    results_folder = os.path.join(RESULTS_FOLDER, args.dataset)
    os.makedirs(results_folder, exist_ok=True)

    results = dict()
    if engine.is_offline():
        results["realtime factor"] = total_processing_time / total_audio_length
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


if __name__ == "__main__":
    main()

__all__ = [
    "RESULTS_FOLDER",
]
