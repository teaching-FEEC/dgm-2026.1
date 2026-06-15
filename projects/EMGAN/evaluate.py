#!/usr/bin/env python3
"""
Avaliacao de multiplos checkpoints do EMG generator (STE-GAN).

Para cada modelo (EMG_GEN_DIR):
  - gera o EMG fake a partir de cada amostra
  - calcula o envelope (retifica + media, como no artigo) e o Env. CC real vs gerado
  - calcula a distancia (L1/L2) entre o SU original (sample[SU_KEY]) e o SU predito
  - encoda o EMG fake -> speech units (SU) -> sintetiza audio (soft-vc) -> salva
  - transcreve (whisper) e calcula WER / CER
  - tudo salvo, por utterance, em results.csv

A baseline "original" (EMG real -> encoder -> SU -> audio -> WER/CER) NAO depende
do generator (o emg encoder e comum), entao roda UMA vez e e salva a parte em
baseline_metrics.csv, com os audios em baseline/audio/.

Erros sao capturados por utterance: a linha e gravada mesmo assim, com a mensagem
de erro na coluna "error", e o loop segue para a proxima.

Rode a partir da raiz do repositorio ste-gan (onde fica a pasta `ste_gan/`).
"""

import sys
import os
import csv
import json
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from tqdm import tqdm
import jiwer

from scipy import signal # addition

# ======================================================================
# CONFIG  -- edite aqui
# ======================================================================

# Raiz do repositorio ste-gan (onde esta a pasta `ste_gan/`).
REPO_ROOT = Path(".")

# >>> Raiz dos experimentos (a pasta `exp/`). <<<
EXP_ROOT = Path("/workspace/EMG/dgm-2026.1/projects/EMGAN/exp")

# >>> Experimentos a testar (os nomes do `ls`). [] = descobre TODOS automaticamente. <<<
# O nome de cada item vira o nome da pasta de saida (stegan e suas variacoes).
EXPERIMENTS = [
    "ste-gan_mtd0_neurorvq0",
    "ste-gan_mtd0_neurorvq1",
    "ste-gan_mtd0_neurorvq7",
    "ste-gan_mtd0_neurorvq15",
    "ste-gan_mtd15_neurorvq0",
    "ste-gan_mtd15_neurorvq1",
    "ste-gan_mtd15_neurorvq7",
    "ste-gan_mtd15_neurorvq15",
]
GEN_CKPT_NAME = "best_netG.pt"     # nome do checkpoint dentro da subpasta do modelo
GEN_CONFIG_NAME = "config.yaml"    # nome do config dentro da subpasta do modelo


# EMG encoder (comum a todos os modelos)
EMG_ENC_DIR = "/workspace/EMG/data_models_OG_stegan/exp/emg_encoder/EMGEncoderTransformer_voiced_only__seq_len__200__data_gaddy_complete"
EMG_ENC_CHECKPOINT = f"{EMG_ENC_DIR}/best_val_loss_model.pt"
EMG_ENC_CONFIG_PATH = f"{EMG_ENC_DIR}/config.yaml"

# Dados
PARTITION = "valid"
DATA_ROOT = Path("/workspace/EMG/data_models_OG_stegan/data/gaddy_complete")

# Saida
OUTPUT_ROOT = Path("./eval_outputs")
RESULTS_CSV = OUTPUT_ROOT / "results.csv"            # rms + su_dist + wer + cer (por modelo/utterance)
BASELINE_CSV = OUTPUT_ROOT / "baseline_metrics.csv"  # wer/cer do EMG real (uma vez)
SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"            # medias por modelo (no fim)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")  # descomente para forcar CPU

# Parametros
# Envelope como no artigo (Scheck & Schultz, Interspeech 2023): retifica e
# aplica filtro de media com janela de 50 ms. A janela em AMOSTRAS depende da
# taxa de amostragem do EMG (o artigo usa 800 Hz -> 40 amostras).
EMG_FS = 800            # taxa de amostragem do EMG em Hz (confira a do seu sinal!)
ENV_WINDOW_MS = 50      # janela do envelope em ms (artigo: 50 ms)
ENV_WINDOW = max(1, round(EMG_FS * ENV_WINDOW_MS / 1000.0))  # janela em amostras

WHISPER_MODEL = "medium.en"

MAX_UTTERANCES = None   # None = todas; ou um int para testes rapidos
RESUME = True           # pula (modelo, utt) ja presentes no results.csv

# ======================================================================
# Imports do ste-gan (precisa do REPO_ROOT no path)
# ======================================================================
sys.path.append(str(REPO_ROOT.resolve()))
from ste_gan.models.generator import init_emg_generator          # noqa: E402
from ste_gan.models.emg_encoder import load_emg_encoder          # noqa: E402
from ste_gan.data.emg_dataset import EMGDataset                  # noqa: E402
from ste_gan.utils.common import fix_state_dict                  # noqa: E402

# ======================================================================
# Helpers
# ======================================================================

JIWER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
])


def envelope(emg: torch.Tensor, window: int) -> torch.Tensor:
    """Envelope como no artigo (Scheck & Schultz, 2023): retifica (|x|) e
    aplica um filtro de media com a janela dada. emg: [T, C] -> [T, C]."""
    rect = emg.abs()
    env = (
        F.avg_pool1d(rect.T.unsqueeze(0), window, stride=1, padding=window // 2)
        .squeeze(0)
        .T[: len(emg)]
    )
    return env


def envelope_metrics(real_emg: torch.Tensor, pred_emg: torch.Tensor, window: int):
    """Metricas de envelope real vs gerado.

    Retorna:
      env_cc        -> Envelope Correlation Coefficient medio entre canais (metrica do artigo)
      cc_per_channel-> CC por canal [list]
      l1_per_channel-> L1 por canal [list]   (mantido para a sua analise)
      l1_total      -> L1 medio
      l2            -> L2 por frame, media
    """
    env_real = envelope(real_emg, window)
    env_pred = envelope(pred_emg, window)
    # alinha comprimentos (real e pred podem diferir)
    n = min(env_real.shape[0], env_pred.shape[0])
    env_real = env_real[:n]
    env_pred = env_pred[:n]

    # Env. CC: correlacao de Pearson por canal, depois media entre canais
    cc_per_channel = []
    for c in range(env_real.shape[1]):
        a = env_real[:, c] - env_real[:, c].mean()
        b = env_pred[:, c] - env_pred[:, c].mean()
        denom = torch.sqrt((a * a).sum() * (b * b).sum())
        r = (a * b).sum() / denom if denom > 0 else torch.tensor(float("nan"))
        cc_per_channel.append(float(r))
    env_cc = float(np.nanmean(cc_per_channel))

    l1_per_channel = torch.mean(torch.abs(env_real - env_pred), dim=0)
    l1_total = l1_per_channel.mean().item()
    l2 = torch.sqrt(torch.sum((env_real - env_pred) ** 2, dim=1)).mean().item()
    return env_cc, cc_per_channel, l1_per_channel.detach().cpu().tolist(), l1_total, l2

def coherence(real_emg: torch.Tensor, pred_emg: torch.Tensor, intervals: list):

    dic = {}
    win_length = 200 # for application of Hann
    hop_length = 100
    n_fft = win_length # number of fft size

    for key in list(intervals):
        mean_cohe = 0
        for channel in range(8): # loop over each channel
            # Ensure signals are 1D float tensors
            x = real_emg[:,channel]
            y = pred_emg[:,channel]
            
            # Generate standard Hann window
            window = torch.hann_window(win_length, device=x.device)
            
            # Compute short-time Fourier transforms (STFT)
            X = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True, center=False)
            Y = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True, center=False)

            # resulting window
            cohe_win = X.shape[0]
            
            # Calculate Auto Power Spectral Densities (PSD) by averaging across time frames
            Pxx = torch.mean(X.abs() ** 2, dim=-1)
            Pyy = torch.mean(Y.abs() ** 2, dim=-1)
            
            # Calculate Cross Power Spectral Density (CPSD)
            Pxy = torch.mean(X * torch.conj(Y), dim=-1)
            
            # Compute Magnitude-Squared Coherence - for each window
            coherence = (Pxy.abs() ** 2) / (Pxx * Pyy + 1e-8)

            new_res = (EMG_FS/2)/cohe_win # new resolution
            mean_cohe += coherence[int(key[0]/new_res):int(key[1]/new_res)].mean().item() # aggregates coherences accross channels

        dic['cohe_'+str(key[0])+'-'+str(key[1])+'_Hz'] = mean_cohe/8

    return list(dic.values())

def mse_psd(real_emg: torch.Tensor, pred_emg: torch.Tensor, intervals: list):

    dic = {}
    win_length = 200 # for application of Hann
    hop_length = 100
    n_fft = win_length # number of fft size

    for key in list(intervals):
        mean_cohe = 0
        for channel in range(8): # loop over each channel
            # Ensure signals are 1D float tensors
            x = real_emg[:,channel].numpy()
            y = pred_emg[:,channel].numpy()
            
            # nperseg controls the segment length (affects frequency resolution vs smoothing)
            _, psd_x = signal.welch(
                x=x, 
                fs=EMG_FS, 
                window='hann', 
                nperseg=win_length, 
                noverlap=hop_length, 
                scaling='density'
            )

            _, psd_y = signal.welch(
                x=y, 
                fs=EMG_FS, 
                window='hann', 
                nperseg=win_length, 
                noverlap=hop_length, 
                scaling='density'
            )

            # normalization
            psd_x_normalized = (psd_x[int(key[0]/new_res):int(key[1]/new_res)] - psd_x.mean())/psd_x.std()
            psd_y_normalized = (psd_y[int(key[0]/new_res):int(key[1]/new_res)] - psd_y.mean())/psd_y.std()

            new_res = (EMG_FS/2)/psd_x.shape[0] # new resolution
            mean_cohe += np.mean((psd_y_normalized - psd_x_normalized)**2) # aggregates coherences accross channels

        dic['cohe_'+str(key[0])+'-'+str(key[1])+'_Hz'] = mean_cohe/8

    return list(dic.values())

def _to_2d(su: torch.Tensor) -> torch.Tensor:
    """Normaliza SU para [T, D] em CPU (remove dim de batch se houver)."""
    su = su.detach().cpu()
    if su.dim() == 3:
        su = su.squeeze(0)
    return su


def su_distance(su_orig: torch.Tensor, su_pred: torch.Tensor):
    """Distancia entre SU original (ground-truth) e SU predito.

    Ambos -> [T, D]; alinha o tempo pelo minimo (como no RMS).
    Retorna (su_l1[float], su_l2[float]).
    """
    a = _to_2d(su_orig)
    b = _to_2d(su_pred)
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(f"dim do SU difere: orig {tuple(a.shape)} vs pred {tuple(b.shape)}")
    n = min(a.shape[0], b.shape[0])
    a = a[:n]
    b = b[:n]
    su_l1 = torch.mean(torch.abs(a - b)).item()
    su_l2 = torch.sqrt(torch.sum((a - b) ** 2, dim=1)).mean().item()
    return su_l1, su_l2


def encode_su(emg: torch.Tensor, netenc, device) -> torch.Tensor:
    """EMG [T, C] -> speech units [1, T', D].

    O encoder retorna uma TUPLA (o SU e' o primeiro elemento), por isso o
    notebook usava `pred_su[0]`.
    """
    emg = emg.to(device)
    with torch.no_grad():
        out = netenc(emg.unsqueeze(0))  # entrada [1, T, C]
    su = out[0] if isinstance(out, (tuple, list)) else out
    return su


def su_to_audio(su: torch.Tensor, acoustic, hifigan) -> torch.Tensor:
    """su: [T, C] ou [1, T, C] -> waveform 1D @ 16kHz."""
    dev = next(acoustic.parameters()).device
    su = su.to(dev)
    if su.dim() == 2:
        su = su.unsqueeze(0)  # garante [1, T, C]
    with torch.inference_mode():
        mel = acoustic.generate(su).transpose(1, 2)
        target = hifigan(mel)
    return target.squeeze()


def transcribe(audio: torch.Tensor, whisper_model) -> str:
    audio = audio.detach().cpu().float()
    return whisper_model.transcribe(audio)["text"]


def wer_cer(reference: str, hypothesis: str):
    ref = JIWER_TRANSFORM(reference)
    hyp = JIWER_TRANSFORM(hypothesis)
    return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)


def save_audio(path: Path, audio: torch.Tensor, sr: int = 16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.detach().cpu().numpy().astype(np.float32), sr)


def save_emg(path: Path, emg: torch.Tensor):
    """Salva o EMG gerado [T, C] como .npy float32 (recarrega com np.load)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), emg.detach().cpu().numpy().astype(np.float32))


class CsvWriter:
    """CSV append-only com flush a cada linha (sobrevive a crashes)."""

    def __init__(self, path: Path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames)
        if new_file:
            self._w.writeheader()
            self._f.flush()

    def write(self, row: dict):
        self._w.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._f.flush()

    def close(self):
        self._f.close()


def load_done_keys(path: Path):
    """Le pares (model_name, utt_id) ja gravados, para resume."""
    done = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row.get("model_name", ""), row.get("utt_id", "")))
    return done


def find_model_dir(exp_dir: Path) -> Path:
    """Acha a subpasta que contem best_netG.pt dentro de um experimento.

    Aceita tanto a pasta do experimento (.../exp/ste-gan_mtd0_neurorvq0) quanto
    a propria subpasta do modelo (.../gaddy_voiced_EMGGeneratorGanTTS_...).
    """
    if (exp_dir / GEN_CKPT_NAME).exists():
        return exp_dir
    matches = sorted(exp_dir.glob(f"*/{GEN_CKPT_NAME}"))
    if not matches:
        matches = sorted(exp_dir.glob(f"**/{GEN_CKPT_NAME}"))
    if not matches:
        raise FileNotFoundError(f"'{GEN_CKPT_NAME}' nao encontrado em {exp_dir}")
    if len(matches) > 1:
        print(f"[aviso] varios '{GEN_CKPT_NAME}' em {exp_dir.name}; usando {matches[0].parent}")
    return matches[0].parent


# ======================================================================
# Main
# ======================================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device = {DEVICE}")

    # ---- dados ----
    data = EMGDataset(DATA_ROOT, partition=PARTITION)
    indices = list(range(len(data)))
    if MAX_UTTERANCES is not None:
        indices = indices[:MAX_UTTERANCES]
    print(f"[setup] {len(indices)} utterances ({PARTITION})")

    # ---- modelos comuns (carregados uma vez) ----
    print("[setup] carregando EMG encoder ...")
    enc_cfg = OmegaConf.load(EMG_ENC_CONFIG_PATH)
    netenc = load_emg_encoder(enc_cfg, DEVICE, EMG_ENC_CHECKPOINT)
    netenc.eval()

    print("[setup] carregando soft-vc (acoustic + hifigan) ...")
    acoustic = torch.hub.load("bshall/acoustic-model:main", "hubert_soft", trust_repo=True).to(DEVICE)
    hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft", trust_repo=True).to(DEVICE)
    acoustic.eval()
    hifigan.eval()

    print(f"[setup] carregando whisper ({WHISPER_MODEL}) ...")
    import whisper
    whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)

    # ==================================================================
    # PASSO 1: baseline (EMG real -> encoder -> SU -> audio). Roda UMA vez.
    # ==================================================================
    baseline_dir = OUTPUT_ROOT / "baseline" / "audio"
    baseline_done = load_done_keys(BASELINE_CSV) if RESUME else set()
    baseline_writer = CsvWriter(
        BASELINE_CSV,
        fieldnames=["utt_id", "reference", "transcription_orig",
                    "wer_orig", "cer_orig", "audio_path_orig", "error"],
    )
    print("\n[baseline] EMG real -> encoder -> audio (comum a todos os modelos)")
    for idx in tqdm(indices, desc="baseline"):
        try:
            sample = data[idx]
            utt_id = sample["UTT_ID"]
            if RESUME and ("", utt_id) in baseline_done:
                continue

            real_emg = sample["REAL_EMG"].to(DEVICE)
            real_su = sample["SPEECH_UNITS"].unsqueeze(0).to(DEVICE)

            audio_orig = su_to_audio(real_su, acoustic, hifigan)
            audio_path = baseline_dir / f"{utt_id}.wav"
            save_audio(audio_path, audio_orig)

            transcription = transcribe(audio_orig, whisper_model)
            wer, cer = wer_cer(sample["TRANSCRIPTION"], transcription)

            baseline_writer.write({
                "utt_id": utt_id,
                "reference": sample["TRANSCRIPTION"],
                "transcription_orig": transcription,
                "wer_orig": f"{wer:.4f}",
                "cer_orig": f"{cer:.4f}",
                "audio_path_orig": str(audio_path.resolve()),
                "error": "",
            })
        except Exception as e:
            utt_id = locals().get("utt_id", f"idx_{idx}")
            baseline_writer.write({
                "utt_id": utt_id,
                "error": f"{type(e).__name__}: {e} | {traceback.format_exc().splitlines()[-1]}",
            })
            tqdm.write(f"[baseline][ERRO] {utt_id}: {e}")
    baseline_writer.close()

    # ==================================================================
    # PASSO 2: por modelo
    # ==================================================================
    results_fields = [
        "model_name", "model_checkpoint", "utt_id", "reference",
        "env_cc", "env_cc_per_channel",
        "env_l1_total", "env_l2", "env_l1_per_channel",
        "su_l1", "su_l2",
        "transcription_gen", "wer_gen", "cer_gen",
        "audio_path_gen", "emg_path_gen", "error",
        "cohe" # adicionado
    ]
    results_done = load_done_keys(RESULTS_CSV) if RESUME else set()
    results_writer = CsvWriter(RESULTS_CSV, fieldnames=results_fields)

    # monta a lista de experimentos
    if EXPERIMENTS:
        exp_dirs = [EXP_ROOT / e for e in EXPERIMENTS]
    else:
        exp_dirs = [p for p in sorted(EXP_ROOT.iterdir()) if p.is_dir()]

    for exp_dir in exp_dirs:
        model_name = exp_dir.name  # <- nome de saida = nome do experimento (stegan e variacoes)
        print(f"\n[modelo] {model_name}")

        # resolve a subpasta que contem o checkpoint
        try:
            gen_dir = find_model_dir(exp_dir)
            ckpt = gen_dir / GEN_CKPT_NAME
            cfg_path = gen_dir / GEN_CONFIG_NAME
        except FileNotFoundError as e:
            results_writer.write({
                "model_name": model_name,
                "model_checkpoint": str(exp_dir),
                "utt_id": "__MODEL_LOAD__",
                "error": f"{type(e).__name__}: {e}",
            })
            tqdm.write(f"[modelo][ERRO] {model_name}: {e}")
            continue

        # carrega o generator (erro aqui -> pula o modelo inteiro, registrado)
        try:
            gcfg = OmegaConf.load(cfg_path)
            netG = init_emg_generator(gcfg)
            state_dict = torch.load(ckpt, map_location=DEVICE)
            netG.load_state_dict(fix_state_dict(state_dict))
            netG.to(DEVICE)
            netG.eval()
        except Exception as e:
            results_writer.write({
                "model_name": model_name,
                "model_checkpoint": str(ckpt),
                "utt_id": "__MODEL_LOAD__",
                "error": f"{type(e).__name__}: {e} | {traceback.format_exc().splitlines()[-1]}",
            })
            tqdm.write(f"[modelo][ERRO ao carregar] {model_name}: {e}")
            continue

        model_audio_dir = OUTPUT_ROOT / model_name / "audio"
        model_emg_dir = OUTPUT_ROOT / model_name / "emg"

        for idx in tqdm(indices, desc=model_name[:40]):
            utt_id = f"idx_{idx}"
            try:
                sample = data[idx]
                utt_id = sample["UTT_ID"]
                if RESUME and (model_name, utt_id) in results_done:
                    continue

                real_emg = sample["REAL_EMG"].to(DEVICE)

                # --- gera EMG fake ---
                with torch.no_grad():
                    pred_emg = netG.generate_from_data_dict(sample, DEVICE)  # [T, C]
                pred_emg = pred_emg.to(DEVICE)

                # --- salva o EMG gerado [T, C] (.npy) ---
                emg_path = model_emg_dir / f"{utt_id}.npy"
                save_emg(emg_path, pred_emg)

                # --- envelope real vs gerado (artigo: retifica + media; metrica Env. CC) ---
                env_cc, cc_pc, l1_pc, l1_total, l2 = envelope_metrics(
                    real_emg.detach().cpu(), pred_emg.detach().cpu(), ENV_WINDOW
                )

                # --- COHERENCE ---
                dict_cohe = {(20, 250): []} # frequency intervals for coherence metric
                list_cohe = mse_psd(
                    real_emg.detach().cpu(), pred_emg.detach().cpu(), dict_cohe.keys()
                )

                # --- EMG fake -> SU predito ---
                pred_su = encode_su(pred_emg, netenc, DEVICE)  # [1, T', D]

                # --- distancia SU original (ground-truth) vs SU predito ---
                su_l1, su_l2 = su_distance(sample["SPEECH_UNITS"], pred_su)

                # --- SU predito -> audio ---
                audio_gen = su_to_audio(pred_su, acoustic, hifigan)
                audio_path = model_audio_dir / f"{utt_id}.wav"
                save_audio(audio_path, audio_gen)

                # --- transcricao + WER/CER ---
                transcription = transcribe(audio_gen, whisper_model)
                wer, cer = wer_cer(sample["TRANSCRIPTION"], transcription)

                results_writer.write({
                    "model_name": model_name,
                    "model_checkpoint": str(ckpt),
                    "utt_id": utt_id,
                    "reference": sample["TRANSCRIPTION"],
                    "env_cc": f"{env_cc:.6f}",
                    "env_cc_per_channel": json.dumps([round(x, 6) for x in cc_pc]),
                    "env_l1_total": f"{l1_total:.6f}",
                    "env_l2": f"{l2:.6f}",
                    "env_l1_per_channel": json.dumps([round(x, 6) for x in l1_pc]),
                    "su_l1": f"{su_l1:.6f}",
                    "su_l2": f"{su_l2:.6f}",
                    "transcription_gen": transcription,
                    "wer_gen": f"{wer:.4f}",
                    "cer_gen": f"{cer:.4f}",
                    "audio_path_gen": str(audio_path.resolve()),
                    "emg_path_gen": str(emg_path.resolve()),
                    "error": "",
                    "cohe": f"{list_cohe[0]:.4f}",
                })
            except Exception as e:
                results_writer.write({
                    "model_name": model_name,
                    "model_checkpoint": str(ckpt),
                    "utt_id": utt_id,
                    "error": f"{type(e).__name__}: {e} | {traceback.format_exc().splitlines()[-1]}",
                })
                tqdm.write(f"[{model_name}][ERRO] {utt_id}: {e}")

        # libera memoria do generator antes do proximo
        del netG
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    results_writer.close()

    # ==================================================================
    # PASSO 3: resumo (medias por modelo)
    # ==================================================================
    write_summary()
    print(f"\n[ok] resultados em: {RESULTS_CSV}")
    print(f"[ok] baseline  em: {BASELINE_CSV}")
    print(f"[ok] resumo    em: {SUMMARY_CSV}")


def write_summary():
    """Agrega medias de envelope/SU/WER/CER por modelo (ignora linhas com erro)."""
    if not RESULTS_CSV.exists():
        return
    agg = {}  # model_name -> dict de listas
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            m = row["model_name"]
            d = agg.setdefault(m, {"env_cc": [], "env_l1": [], "env_l2": [],
                                   "su_l1": [], "su_l2": [], "wer": [], "cer": [],
                                   "cohe": [], "n": 0})
            try:
                d["env_cc"].append(float(row["env_cc"]))
                d["env_l1"].append(float(row["env_l1_total"]))
                d["env_l2"].append(float(row["env_l2"]))
                d["su_l1"].append(float(row["su_l1"]))
                d["su_l2"].append(float(row["su_l2"]))
                d["wer"].append(float(row["wer_gen"]))
                d["cer"].append(float(row["cer_gen"]))
                d["cohe"].append(float(row["cohe"]))
                d["n"] += 1
            except (ValueError, KeyError):
                pass

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_name", "n_ok", "mean_env_cc", "mean_env_l1", "mean_env_l2",
                    "mean_su_l1", "mean_su_l2", "mean_wer", "mean_cer", "mean_cohe"])
        for m, d in agg.items():
            w.writerow([m, d["n"], f"{mean(d['env_cc']):.4f}",
                        f"{mean(d['env_l1']):.6f}", f"{mean(d['env_l2']):.6f}",
                        f"{mean(d['su_l1']):.6f}", f"{mean(d['su_l2']):.6f}",
                        f"{mean(d['wer']):.4f}", f"{mean(d['cer']):.4f}",
                        f"{mean(d['cohe']):.4f}"])


if __name__ == "__main__":
    main()
