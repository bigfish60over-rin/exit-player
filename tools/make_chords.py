"""EXIT PLAYER 用 chords.json を PC で作るプログラム。

使い方（C:\\dev\\tab-player の Python 環境を使う）:
  C:\\dev\\tab-player\\envs\\main\\Scripts\\python.exe tools\\make_chords.py features music 作業フォルダ
  C:\\dev\\tab-player\\envs\\main\\Scripts\\python.exe tools\\make_chords.py decode 作業フォルダ music chords.json

features : 曲ごとに「拍・小節の頭（beat_this）」と「伴奏・ベースの音の成分（demucs で歌とドラムを除いたもの）」を
           作業フォルダに保存する（重い処理。1回だけ）
decode   : 保存した成分からコードを判定して chords.json を書く（軽い処理）

判定する種類: メジャー / m / 7 / m7 / M7 / sus4 / dim(m♭5) / m7b5(m7♭5)
chords.json の形（アプリが読む形）:
  {"曲名": {"size": mp3のバイト数, "bpm": 数値, "bars": [小節の頭の秒...],
            "chords": [[開始秒, 終了秒, 根音(0=C..11=B / -1=なし), 種類], ...]}}
"""
import os, sys, json, subprocess
import numpy as np

TAB = r"C:\dev\tab-player"
NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
QUAL = {"": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10], "M7": [0, 4, 7, 11],
        "sus4": [0, 5, 7], "dim": [0, 3, 6], "m7b5": [0, 3, 6, 10]}
SR = 22050
HOP = 512


def load_audio(path, sr=44100):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


# ------------------------------------------------------------------ features
def features(music, cache):
    import torch, librosa
    from pathlib import Path
    from beat_this.inference import Audio2Beats
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    os.makedirs(cache, exist_ok=True)
    a2b = Audio2Beats(checkpoint_path="final0", device="cuda", dbn=False)
    model = get_model("htdemucs", repo=Path(os.path.join(TAB, "models", "demucs"))).to("cuda").eval()
    for fn in sorted(os.listdir(music)):
        if not fn.lower().endswith(".mp3"):
            continue
        title = fn[:-4]
        out = os.path.join(cache, title + ".npz")
        if os.path.exists(out):
            continue
        print("features:", title, flush=True)
        a = load_audio(os.path.join(music, fn))
        beats, downs = a2b(a.mean(0), 44100)
        import random
        random.seed(0); torch.manual_seed(0)
        x = torch.from_numpy(a)[None]
        ref = x.mean(1); mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            st = apply_model(model, (x - mean) / std, device="cuda", shifts=1, split=True, overlap=0.25)[0]
        st = (st * std + mean).cpu().numpy()
        src = model.sources  # drums, bass, other, vocals
        other = librosa.resample(st[src.index("other")].mean(0), orig_sr=44100, target_sr=SR)
        bass = librosa.resample(st[src.index("bass")].mean(0), orig_sr=44100, target_sr=SR)
        mix = librosa.resample(a.mean(0), orig_sr=44100, target_sr=SR)
        tre = librosa.feature.chroma_cqt(y=other, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C2"),
                                         n_octaves=5, bins_per_octave=36, norm=None)
        bas = librosa.feature.chroma_cqt(y=bass, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C1"),
                                         n_octaves=3, bins_per_octave=36, norm=None)
        n = min(tre.shape[1], bas.shape[1])
        rms = librosa.feature.rms(y=mix, frame_length=2048, hop_length=HOP)[0][:n]
        np.savez_compressed(out, tre=tre[:, :n], bas=bas[:, :n], rms=rms, beats=np.array(beats, float),
                            downs=np.array(downs, float), dur=len(mix) / SR)
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ decode
# 4和音・sus4・dim は「その音がはっきり鳴っている時だけ」選ぶよう、少し不利にしておく
PRIOR = {"": 0.0, "m": 0.0, "7": -0.055, "m7": -0.055, "M7": -0.10, "sus4": -0.12, "dim": -0.08, "m7b5": -0.08}
WB = 0.60         # ベース音が根音と同じ時のおまけ
PEN_DOWN = 0.22    # 小節の頭でコードが変わる時の罰
PEN_BEAT = 0.40    # 小節の途中でコードが変わる時の罰


def templates():
    T, names = [], []
    for r in range(12):
        for q, iv in QUAL.items():
            t = np.zeros(12)
            for i, v in enumerate(iv):
                t[(r + v) % 12] = [1.0, 1.0, 0.8, 1.0][i]
            T.append(t / np.linalg.norm(t)); names.append((r, q))
    return np.array(T), names


def beat_grid(beats, dur):
    b = [float(x) for x in beats if 0 < x < dur]
    if not b:
        return np.arange(0, dur, 0.5).tolist() + [dur]
    return [0.0] + b + [dur]


def sync(feat, grid):
    t = np.arange(feat.shape[1]) * HOP / SR
    out = []
    for a, b in zip(grid[:-1], grid[1:]):
        m = (t >= a) & (t < b)
        out.append(feat[:, m].mean(1) if m.any() else np.zeros(feat.shape[0]))
    return np.array(out)  # (beats, 12)


def decode_song(npz):
    d = np.load(npz)
    dur = float(d["dur"]); grid = beat_grid(d["beats"], dur)
    tre = sync(d["tre"], grid); bas = sync(d["bas"], grid)
    rms = sync(d["rms"][None], grid)[:, 0]
    T, names = templates(); S = len(names)
    B = len(grid) - 1
    med = np.median(rms[rms > 0]) if (rms > 0).any() else 0
    em = np.zeros((B, S + 1))
    for i in range(B):
        v = tre[i]; v = v / (v.max() + 1e-9); v = np.maximum(0, v - 0.15)
        v = v / (np.linalg.norm(v) + 1e-9)
        bb = bas[i]; bb = bb / (bb.sum() + 1e-9)
        sc = T @ v
        for s, (r, q) in enumerate(names):
            sc[s] += WB * bb[r] + PRIOR[q]
        em[i, :S] = sc
        em[i, S] = 1.5 if rms[i] < med * 0.08 else -1
    downs = set(np.round(d["downs"], 2).tolist())
    isdown = [round(g, 2) in downs for g in grid[:-1]]
    prev = em[0].copy(); back = np.zeros((B, S + 1), int)
    for i in range(1, B):
        pen = PEN_DOWN if isdown[i] else PEN_BEAT
        bi = int(prev.argmax()); jump = prev[bi] - pen
        stay = prev >= jump
        back[i] = np.where(stay, np.arange(S + 1), bi)
        prev = np.where(stay, prev, jump) + em[i]
    path = np.zeros(B, int); path[-1] = int(prev.argmax())
    for i in range(B - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    segs = []
    for i, s in enumerate(path):
        if segs and segs[-1][2] == s:
            segs[-1][1] = grid[i + 1]
        else:
            segs.append([grid[i], grid[i + 1], s])
    chords = []
    for a, b, s in segs:
        r, q = (-1, "") if s == S else names[s]
        chords.append([round(a, 2), round(b, 2), r, q])
    bpm = 60 / np.median(np.diff(d["beats"])) if len(d["beats"]) > 2 else 0
    return {"bpm": round(float(bpm), 1), "bars": [round(float(x), 2) for x in d["downs"]], "chords": chords}


def decode(cache, music, out):
    res = {}
    for fn in sorted(os.listdir(music)):
        if not fn.lower().endswith(".mp3"):
            continue
        title = fn[:-4]
        npz = os.path.join(cache, title + ".npz")
        if not os.path.exists(npz):
            print("特徴がありません:", title); continue
        r = decode_song(npz)
        res[title] = {"size": os.path.getsize(os.path.join(music, fn)), **r}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, separators=(",", ":"))
    print("wrote", out, len(res), "songs")


if __name__ == "__main__":
    if sys.argv[1] == "features":
        features(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "decode":
        decode(sys.argv[2], sys.argv[3], sys.argv[4])
