import glob
import os.path
from enum import Enum
from typing import *

from pyannote.core import Annotation

from util import load_rttm, rttm_to_annotation, get_audio_length

Sample = Tuple[str, str, float]


class Datasets(Enum):
    VOX_CONVERSE = "VoxConverse"


class Dataset:
    @property
    def size(self) -> int:
        raise NotImplementedError()

    @property
    def samples(self) -> Sequence[Sample]:
        raise NotImplementedError()

    def get(self, index: int) -> Tuple[str, float, Annotation]:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    @classmethod
    def create(cls, x: Datasets, data_folder: str, **kwargs: Any) -> "Dataset":
        try:
            subclass = {
                Datasets.VOX_CONVERSE: VoxConverse,
            }[x]
        except KeyError:
            raise ValueError(f"cannot create `{cls.__name__}` of type `{x.value}`")
        return subclass(data_folder, **kwargs)


class VoxConverse(Dataset):
    def __init__(self, data_folder: str, label_folder: str, only_en: bool = True) -> None:
        en_audio_files = {
            "aepyx.wav", "aggyz.wav", "aiqwk.wav", "aorju.wav", "auzru.wav", "bjruf.wav", "bmsyn.wav", "bvqnu.wav",
            "bvyvm.wav", "bxcfq.wav", "cadba.wav", "cawnd.wav", "clfcg.wav", "cpebh.wav", "cqfmj.wav", "crorm.wav",
            "crylr.wav", "cvofp.wav", "dgvwu.wav", "dkabn.wav", "dlast.wav", "dohag.wav", "dxbbt.wav", "dxokr.wav",
            "dzsef.wav", "dzxut.wav", "eazeq.wav", "eddje.wav", "eguui.wav", "eoyaz.wav", "epygx.wav", "erslt.wav",
            "eucfa.wav", "euqef.wav", "ezxso.wav", "fpfvy.wav", "fqrnu.wav", "fvhrk.wav", "fxnwf.wav", "fyqoe.wav",
            "fzwtp.wav", "gcfwp.wav", "gfneh.wav", "gkiki.wav", "gmmwm.wav", "gtjow.wav", "gtnjb.wav", "gukoa.wav",
            "gwloo.wav", "gylzn.wav", "gyomp.wav", "hcyak.wav", "heolf.wav", "hhepf.wav", "hqhrb.wav", "iabca.wav",
            "iacod.wav", "ibrnm.wav", "ifwki.wav", "iiprr.wav", "iowob.wav", "isrps.wav", "isxwc.wav", "jbowg.wav",
            "jdrwl.wav", "jeymh.wav", "jgiyq.wav", "jjvkx.wav", "jrfaz.wav", "jttar.wav", "jwggf.wav", "jxpom.wav",
            "jxydp.wav", "kajfh.wav", "kgjaa.wav", "kmjvh.wav", "kmunk.wav", "kpjud.wav", "kvkje.wav", "kzmyi.wav",
            "laoyl.wav", "lbfnx.wav", "ledhe.wav", "leneg.wav", "lhuly.wav", "lilfy.wav", "ljpes.wav", "lkikz.wav",
            "lpola.wav", "ltgmz.wav", "lubpm.wav", "luobn.wav", "mbzht.wav", "mclsr.wav", "mjmgr.wav", "mkhie.wav",
            "mqtep.wav", "msbyq.wav", "mupzb.wav", "mxdpo.wav", "mxduo.wav", "myjoe.wav", "neiye.wav", "nitgx.wav",
            "nlvdr.wav", "nprxc.wav", "nqcpi.wav", "nqyqm.wav", "ocfop.wav", "ofbxh.wav", "olzkb.wav", "ooxlj.wav",
            "oqwpd.wav", "otmpf.wav", "ouvtt.wav", "pccww.wav", "pgtkk.wav", "pkwrt.wav", "poucc.wav", "ppexo.wav",
            "pxqme.wav", "pzxit.wav", "qadia.wav", "qajyo.wav", "qeejz.wav", "qlrry.wav", "qoarn.wav", "qwepo.wav",
            "qxana.wav", "ralnu.wav", "rarij.wav", "rmvsh.wav", "rpkso.wav", "rsypp.wav", "rxulz.wav", "ryken.wav",
            "sbrmv.wav", "sebyw.wav", "sfdvy.wav", "svxzm.wav", "swbnm.wav", "sxqvt.wav", "thnuq.wav", "tiido.wav",
            "tkhgs.wav", "tkybe.wav", "tnjoh.wav", "tpnyf.wav", "tpslg.wav", "tvtoe.wav", "uedkc.wav", "uevxo.wav",
            "uicid.wav", "upshw.wav", "uqxlg.wav", "usqam.wav", "vdlvr.wav", "vgaez.wav", "vgevv.wav", "vncid.wav",
            "vtzqw.wav", "vuewy.wav", "vzuru.wav", "wcxfk.wav", "wdvva.wav", "wemos.wav", "wibky.wav", "wlfsf.wav",
            "wprog.wav", "wwvcs.wav", "wwzsk.wav", "xggbk.wav", "xkmqx.wav", "xlsme.wav", "xlyov.wav", "xmyyy.wav",
            "xqxkt.wav", "xtdcl.wav", "xtzoq.wav", "xvxwv.wav", "ybhwz.wav", "ygrip.wav", "ylgug.wav", "ytmef.wav",
            "ytula.wav", "yukhy.wav", "zedtj.wav", "zehzu.wav", "zowse.wav", "zqidv.wav", "zsgto.wav", "zzsba.wav",
            "zztbo.wav",
        }
        self._samples = list()

        files = glob.iglob(os.path.join(data_folder, "*.wav"))
        for file in files:
            name = os.path.basename(file)
            if only_en and name not in en_audio_files:
                continue
            label_path = os.path.join(label_folder, name.replace(".wav", ".rttm"))
            if not os.path.exists(label_path):
                raise ValueError(f"cannot find label file `{label_path}`")
            audio_length = get_audio_length(file)
            self._samples.append((file, label_path, audio_length))

    @property
    def size(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> Sequence[Sample]:
        return self._samples

    def get(self, index: int) -> Tuple[str, float, Annotation]:
        audio_path, label_path, audio_length = self._samples[index]
        rttm = load_rttm(label_path)
        label = rttm_to_annotation(rttm)
        label.uri = os.path.basename(audio_path)
        return audio_path, audio_length, label

    def __str__(self) -> str:
        return "VoxConverse"


__all__ = [
    "Datasets",
    "Dataset",
    "Sample"
]
