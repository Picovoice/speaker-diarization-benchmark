import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import *

import boto3
import pvfalcon
import requests
import swagger_client
import torch
from azure.storage.blob import BlobServiceClient, ResourceTypes, AccountSasPermissions, generate_account_sas
from google.cloud import speech
from google.cloud import storage
from google.protobuf.json_format import MessageToDict
from omegaconf import OmegaConf
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from simple_diarizer.diarizer import Diarizer

from util import load_rttm, rttm_to_annotation

# NUM_THREADS = os.cpu_count()
# os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
# os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
# torch.set_num_threads(NUM_THREADS)
# torch.set_num_interop_threads(NUM_THREADS)
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Engines(Enum):
    AWS_TRANSCRIBE = "AWS_TRANSCRIBE"
    AZURE_SPEECH_TO_TEXT = "AZURE_SPEECH_TO_TEXT"
    GOOGLE_SPEECH_TO_TEXT = "GOOGLE_SPEECH_TO_TEXT"
    GOOGLE_SPEECH_TO_TEXT_ENHANCED = "GOOGLE_SPEECH_TO_TEXT_ENHANCED"
    PICOVOICE_FALCON = "PICOVOICE_FALCON"
    PYANNOTE = "PYANNOTE"
    SIMPLE_DIARIZER = "SIMPLE_DIARIZER"


class Engine(object):
    def diarization(self, path: str) -> "Annotation":
        raise NotImplementedError()

    def cleanup(self) -> None:
        raise NotImplementedError()

    def is_offline(self) -> bool:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    @classmethod
    def create(cls, x: Engines, **kwargs: Any) -> "Engine":
        try:
            subclass = {
                Engines.AWS_TRANSCRIBE: AWSTranscribeEngine,
                Engines.AZURE_SPEECH_TO_TEXT: AzureSpeechToTextEngine,
                Engines.GOOGLE_SPEECH_TO_TEXT: GoogleSpeechToTextEngine,
                Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED: GoogleSpeechToTextEnhancedEngine,
                Engines.PICOVOICE_FALCON: PicovoiceFalconEngine,
                Engines.PYANNOTE: PyAnnoteEngine,
                Engines.SIMPLE_DIARIZER: SimpleDiarizerEngine,
            }[x]
        except KeyError:
            raise ValueError(f"cannot create `{cls.__name__}` of type `{x.value}`")
        return subclass(**kwargs)


class PicovoiceFalconEngine(Engine):
    def __init__(self, access_key: str) -> None:
        # TODO: remove hard-coded paths
        zoo_dev_path = os.path.expanduser("~/work/gitlab/zoo-dev")
        model_path = os.path.join(zoo_dev_path, "res/falcon/param/falcon_params.pv")
        library_path = os.path.join(zoo_dev_path, "build/release/x86_64/src/falcon/libpv_falcon.so")

        self._falcon = pvfalcon.Falcon(
            access_key=access_key,
            model_path=model_path,
            library_path=library_path,
        )
        super().__init__()

    def diarization(self, path: str) -> "Annotation":
        segments = self._falcon.process_file(path)
        return self._segments_to_annotation(segments)

    @staticmethod
    def _segments_to_annotation(segments):
        annotation = Annotation()
        for segment in segments:
            start = segment.start_sec
            end = segment.end_sec
            annotation[Segment(start, end)] = segment.speaker_tag

        return annotation.support()

    def cleanup(self) -> None:
        self._falcon.delete()

    def is_offline(self) -> bool:
        return True

    def __str__(self):
        return Engines.PICOVOICE_FALCON.value


class PyAnnoteEngine(Engine):
    def __init__(self, auth_token: str, use_gpu: bool = False) -> None:
        if use_gpu and torch.cuda.is_available():
            torch_device = torch.device("cuda")
        else:
            torch_device = torch.device("cpu")

        self._pretrained_pipeline = Pipeline.from_pretrained(
            checkpoint_path="pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token,
        )
        self._pretrained_pipeline.to(torch_device)
        super().__init__()

    def diarization(self, path: str) -> "Annotation":
        return self._pretrained_pipeline(path)

    def cleanup(self) -> None:
        self._pretrained_pipeline = None

    def is_offline(self) -> bool:
        return True

    def __str__(self) -> str:
        return Engines.PYANNOTE.value


class AWSTranscribeEngine(Engine):
    def __init__(self, bucket_name: str) -> None:
        self._bucket_name = bucket_name

        # TODO: remove hard-coded paths
        self._aws_access_key = "AKIA47ZYJCPNTC73E3NZ"
        self._aws_secret_access_key = "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"
        self._region_name = "us-east-2"

        self._storage = boto3.client(
            "s3",
            aws_access_key_id=self._aws_access_key,
            aws_secret_access_key=self._aws_secret_access_key,
            region_name=self._region_name,
        )

        self._transcribe = boto3.client(
            "transcribe",
            aws_access_key_id=self._aws_access_key,
            aws_secret_access_key=self._aws_secret_access_key,
            region_name=self._region_name,
        )
        super().__init__()

    def diarization(self, path: str) -> "Annotation":
        blob_name = os.path.basename(path)
        temp_path = os.path.join(tempfile.mkdtemp(), blob_name)

        self._storage.upload_file(Filename=path, Bucket=self._bucket_name, Key=blob_name)

        self._transcribe_blob(blob_name=blob_name, results_path=temp_path)

        with open(temp_path) as f:
            transcript = json.load(f)["results"]
        print(transcript)
        return self._transcript_to_annotation(transcript)

    @staticmethod
    def _transcript_to_annotation(transcript: Dict) -> "Annotation":
        segments = transcript["speaker_labels"]["segments"]
        annotation = Annotation()
        for segment in segments:
            start = float(segment["start_time"])
            end = float(segment["end_time"])
            annotation[Segment(start, end)] = segment["speaker_label"]

        return annotation.support()

    def _transcribe_blob(self, blob_name: str, results_path: str) -> None:
        completed = False

        job_name = uuid.uuid4().hex
        response = self._transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            LanguageCode="en-US",
            OutputBucketName=self._bucket_name,
            OutputKey=blob_name + ".json",
            Media={"MediaFileUri": "s3://" + self._bucket_name + "/" + blob_name},
            MediaFormat="wav",
            Settings={
                "ShowSpeakerLabels": True,
                "MaxSpeakerLabels": 9,
            },
        )

        if response["TranscriptionJob"]["TranscriptionJobStatus"] != "IN_PROGRESS":
            completed = True

        while not completed:
            time.sleep(2)

            response = self._transcribe.get_transcription_job(
                TranscriptionJobName=job_name,
            )

            if response["TranscriptionJob"]["TranscriptionJobStatus"] != "IN_PROGRESS":
                completed = True

        self._storage.download_file(Filename=results_path, Bucket=self._bucket_name, Key=blob_name + ".json")

    def is_offline(self) -> bool:
        return False

    def cleanup(self) -> None:
        pass

    def __str__(self):
        return Engines.AWS_TRANSCRIBE.value


class GoogleSpeechToTextEngine(Engine):
    _diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=1,
        max_speaker_count=20,
    )
    _config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_word_time_offsets=True,
        diarization_config=_diarization_config,
    )

    def __init__(self, bucket_name: str) -> None:
        self._speech_client = speech.SpeechClient()
        self._storage_client = storage.Client()
        self._bucket_name = bucket_name
        self._bucket = self._storage_client.bucket(bucket_name)

    def diarization(self, path: str) -> "Annotation":
        blob_name = os.path.basename(path)
        self._upload_audio_to_storage(path, blob_name)
        response = self._transcribe_from_storage(path)
        transcript = response["results"]
        return self._transcript_to_annotation(transcript)

    @staticmethod
    def _transcript_to_annotation(transcript: List[Dict]) -> "Annotation":
        words = transcript[-1]["alternatives"][0]["words"]
        annotation = Annotation()
        for word in words:
            start = float(word["startTime"][:-1])
            end = float(word["endTime"][:-1])
            annotation[Segment(start, end)] = word["speakerTag"]

        return annotation.support()

    def _transcribe_from_storage(self, path: str) -> Dict:
        audio = speech.RecognitionAudio(uri=f"gs://{self._bucket_name}/{os.path.basename(path)}")

        operation = self._speech_client.long_running_recognize(config=self._config, audio=audio)
        response = operation.result(timeout=600)
        response_dict = MessageToDict(response._pb)
        return response_dict

    def _upload_audio_to_storage(self, source_file_name: str, blob_name: str) -> None:
        blob = self._bucket.blob(blob_name)
        stats = storage.Blob(bucket=self._bucket, name=blob_name).exists(self._storage_client)
        if not stats:
            blob.upload_from_filename(source_file_name)

    def is_offline(self) -> bool:
        return False

    def cleanup(self) -> None:
        pass

    def __str__(self):
        return Engines.GOOGLE_SPEECH_TO_TEXT.value


class GoogleSpeechToTextEnhancedEngine(GoogleSpeechToTextEngine):
    _diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=1,
        max_speaker_count=20,
    )
    _config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_word_time_offsets=True,
        diarization_config=_diarization_config,
        model="latest_long",
        use_enhanced=True,
    )

    def __str__(self):
        return Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED.value


class SimpleDiarizerEngine(Engine):
    def __init__(self, use_gpu: bool = False) -> None:
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

        self._diar = Diarizer(embed_model="ecapa", cluster_method="sc")

    def diarization(self, path: str) -> "Annotation":
        segments = self._diar.diarize(path)
        return self._segments_to_annotation(segments)

    @staticmethod
    def _segments_to_annotation(segments: Sequence[Dict[str, Union[int, float]]]) -> "Annotation":
        annotation = Annotation()
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            annotation[Segment(start, end)] = segment["label"]

        return annotation.support()

    def cleanup(self) -> None:
        pass

    def is_offline(self) -> bool:
        return True

    def __str__(self) -> str:
        return Engines.SIMPLE_DIARIZER.value


class NvidiaNeMoEngine(Engine):
    def __init__(self, model_config: str) -> None:
        self._model_config_path = model_config

    def diarization(self, path: str) -> "Annotation":
        from nemo.collections.asr.models import ClusteringDiarizer

        manifest_path = tempfile.TemporaryDirectory().name

        meta = {
            "audio_filepath": path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        with open(manifest_path, "w") as fp:
            json.dump(meta, fp)
            fp.write("\n")

        output_folder = tempfile.TemporaryDirectory().name
        os.makedirs(output_folder, exist_ok=True)

        config = OmegaConf.load(self._model_config_path)
        config.diarizer.manifest_filepath = (manifest_path,)
        config.diarizer.out_dir = output_folder
        config.diarizer.oracle_vad = False
        config.diarizer.clustering.parameters.oracle_num_speakers = False

        sd_model = ClusteringDiarizer(cfg=config)
        sd_model.diarize()

        output_path = os.path.join(output_folder, "pred_rttms", os.path.basename(path).replace(".wav", ".rttm"))
        rttm = load_rttm(output_path)

        return rttm_to_annotation(rttm)

    def cleanup(self) -> None:
        pass

    def is_offline(self) -> bool:
        return False

    def __str__(self) -> str:
        return Engines.NVIDIA_NEMO.value


class AzureSpeechToTextEngine(Engine):
    def __init__(
            self,
            storage_account_key: str,
            storage_account_name: str,
            storage_container_name: str,
            subscription_key: str,
            region: str,
    ) -> None:
        self._storage_account_key = storage_account_key
        self._storage_account_name = storage_account_name
        self._connection_string = \
            (f"DefaultEndpointsProtocol=https;"
             f"AccountName={storage_account_name};"
             f"AccountKey={storage_account_key};"
             f"EndpointSuffix=core.windows.net")
        self._container_name = storage_container_name
        self._subscription_key = subscription_key
        self._service_region = region

    def _upload_to_blob_storage(self, file_path: str, blob_name: str) -> None:
        blob_service_client = BlobServiceClient.from_connection_string(self._connection_string)
        blob_client = blob_service_client.get_blob_client(container=self._container_name, blob=blob_name)

        if blob_client.exists():
            return

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data)

    @staticmethod
    def _transcribe_from_single_blob(uri: str, properties: swagger_client.TranscriptionProperties):
        transcription_definition = swagger_client.Transcription(
            display_name="diariazation",
            description="no-description",
            locale="en-US",
            content_urls=[uri],
            properties=properties,
        )
        return transcription_definition

    @staticmethod
    def _paginate(api, paginated_object):
        yield from paginated_object.values
        typename = type(paginated_object).__name__
        auth_settings = ["api_key"]
        while paginated_object.next_link:
            link = paginated_object.next_link[len(api.api_client.configuration.host):]
            paginated_object, status, headers = api.api_client.call_api(
                link, "GET", response_type=typename, auth_settings=auth_settings
            )

            if status == 200:
                yield from paginated_object.values
            else:
                raise Exception(f"could not receive paginated data: status {status}")

    def diarization(self, path: str) -> "Annotation":
        blob_name = os.path.basename(path)
        self._upload_to_blob_storage(path, blob_name)
        transcripts = self._transcribe(blob_name)

        return self._transcript_to_annotation(transcripts)

    def _transcribe(self, blob_name: str) -> Dict:
        transcripts = {"result": []}
        sas_token = generate_account_sas(
            account_name=self._storage_account_name,
            account_key=self._storage_account_key,
            resource_types=ResourceTypes(service=True, container=True, object=True),
            permission=AccountSasPermissions(read=True, write=True, list=True, delete=True),
            expiry=datetime.utcnow() + timedelta(hours=1),
        )

        blob_service_client = BlobServiceClient.from_connection_string(self._connection_string)
        blob_client = blob_service_client.get_blob_client(container=self._container_name, blob=blob_name)
        blob_url = f"{blob_client.url}?{sas_token}"

        configuration = swagger_client.Configuration()
        configuration.api_key["Ocp-Apim-Subscription-Key"] = self._subscription_key
        configuration.host = f"https://{self._service_region}.api.cognitive.microsoft.com/speechtotext/v3.1"
        client = swagger_client.ApiClient(configuration)
        api = swagger_client.CustomSpeechTranscriptionsApi(api_client=client)
        properties = swagger_client.TranscriptionProperties()
        properties.word_level_timestamps_enabled = True
        properties.display_form_word_level_timestamps_enabled = True
        properties.diarization_enabled = True
        properties.diarization = swagger_client.DiarizationProperties(
            swagger_client.DiarizationSpeakersProperties(min_count=1, max_count=20)
        )
        transcription_definition = self._transcribe_from_single_blob(blob_url, properties)
        (
            created_transcription,
            status,
            headers,
        ) = api.transcriptions_create_with_http_info(transcription=transcription_definition)
        transcription_id = headers["location"].split("/")[-1]

        completed = False

        while not completed:
            time.sleep(5)

            transcription = api.transcriptions_get(transcription_id)

            if transcription.status in ("Failed", "Succeeded"):
                completed = True

            if transcription.status == "Succeeded":
                pag_files = api.transcriptions_list_files(transcription_id)
                for file_data in self._paginate(api, pag_files):
                    if file_data.kind != "Transcription":
                        continue
                    results_url = file_data.links.content_url
                    results = requests.get(results_url)
                    transcripts["result"].append(json.loads(results.content.decode("utf-8")))
            elif transcription.status == "Failed":
                raise Exception(f"Transcription failed: {transcription.properties.error.message}")

        return transcripts

    @staticmethod
    def _transcript_to_annotation(transcript: Dict) -> "Annotation":
        annotation = Annotation()
        pages = transcript["result"]
        for page in pages:
            for segment in page["recognizedPhrases"]:
                start = float(segment["offsetInTicks"] / 10000000)
                end = start + float(segment["durationInTicks"] / 10000000)
                annotation[Segment(start, end)] = segment["speaker"]
        return annotation.support()

    def cleanup(self) -> None:
        pass

    def is_offline(self) -> bool:
        return False

    def __str__(self):
        return Engines.AZURE_SPEECH_TO_TEXT.value
