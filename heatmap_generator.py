import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
import sounddevice as sd
import time
import threading # 音声再生の停止管理用に使うかも

# 計算中の警告（特にlog10(0)）を一旦無視する場合
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- 設定 ---
INPUT_AUDIO_PATH = 'she-chiptune_superstar.wav' # ★要変更: 入力するステレオ音声ファイルのパス
OUTPUT_VIDEO_PATH = 'output_heatmap.mp4' # ★要変更: 出力する動画ファイルのパス

# STFTパラメータ
N_PERSEG = 1024  # FFTのポイント数（周波数解像度）
N_OVERLAP = N_PERSEG // 2 # オーバーラップさせるサンプル数（時間解像度）
WINDOW = 'hann' # 窓関数

# ヒートマップ・動画パラメータ
N_PAN_BINS = 3 # パンの分割数（仮実装用: 0=Left, 1=Center, 2=Right）
PAN_LABELS = ['Left', 'Center', 'Right'] # パン軸のラベル
TARGET_FPS = 30 # 動画のフレームレート
CMAP = 'magma' # ヒートマップの色
DB_RANGE = 60 # 表示するダイナミックレンジ（dB）
# --- 設定ここまで ---

# 音声ファイル読み込み
try:
    audio_data, sample_rate = sf.read(INPUT_AUDIO_PATH, dtype='float32')
    print(f"オーディオファイル読み込み完了: {INPUT_AUDIO_PATH}")
    print(f"サンプルレート: {sample_rate} Hz, 長さ: {audio_data.shape[0]/sample_rate:.2f} 秒")
    if audio_data.ndim != 2 or audio_data.shape[1] != 2:
        raise ValueError("ステレオ音声ファイルではありません。")
    audio_L = audio_data[:, 0]
    audio_R = audio_data[:, 1]
except Exception as e:
    print(f"エラー: オーディオファイルの読み込みに失敗しました - {e}")
    exit()

print("STFTを実行中...")
freqs, times, Zxx_L = signal.stft(audio_L, fs=sample_rate, window=WINDOW, nperseg=N_PERSEG, noverlap=N_OVERLAP)
_, _, Zxx_R = signal.stft(audio_R, fs=sample_rate, window=WINDOW, nperseg=N_PERSEG, noverlap=N_OVERLAP)
print("STFT完了.")

n_freqs = len(freqs)
n_times = len(times)
print(f"周波数ビン数: {n_freqs}, 時間フレーム数: {n_times}")

print("ヒートマップデータを計算中（仮のパン分布）...")
# [周波数ビン数, パンビン数, 時間フレーム数] の配列を初期化
all_heatmap_data = np.zeros((n_freqs, N_PAN_BINS, n_times))

for t in range(n_times):
    for f in range(n_freqs):
        L = Zxx_L[f, t] # 左チャンネルの複素数値
        R = Zxx_R[f, t] # 右チャンネルの複素数値

        # Mid/Side 計算
        M = (L + R) / 2.0
        S = (L - R) / 2.0

        # パワー計算 (|複素数|^2)
        P_M = np.abs(M)**2
        P_S = np.abs(S)**2

        # --- 仮のパン分布マッピング (3 Bins) ---
        # ここはもっと洗練させられるけど、今回はシンプルに
        all_heatmap_data[f, 0, t] = P_S / 2.0 # Left Bin (Side成分の半分)
        all_heatmap_data[f, 1, t] = P_M       # Center Bin (Mid成分)
        all_heatmap_data[f, 2, t] = P_S / 2.0 # Right Bin (Side成分の半分)
        # --- ここまで ---

# dBスケールに変換 & 正規化
# 微小値を足して log10(0) を回避
all_heatmap_data_db = 10 * np.log10(all_heatmap_data + 1e-10)

# 全体の最大値を基準にする
max_db = np.max(all_heatmap_data_db)
min_db = max_db - DB_RANGE # 表示する最小dB値

# min_db と max_db の間でクリップ（範囲外の値を丸める）
all_heatmap_data_db_clipped = np.clip(all_heatmap_data_db, min_db, max_db)
print("ヒートマップデータ計算完了.")

# --- 音声再生と同期のための準備 ---
current_audio_frame = 0 # 現在再生中のオーディオフレーム位置
playback_active = threading.Event() # 再生中フラグ (停止管理用)

# 音声再生コールバック関数
def audio_callback(outdata, frames, time_info, status):
    global current_audio_frame
    if status:
        print(status) # エラー表示

    chunk_start = current_audio_frame
    chunk_end = chunk_start + frames
    
    # 再生範囲が音声データの長さを超えたら停止
    if chunk_end > len(audio_data):
        outdata.fill(0) # 無音データを詰める
        # playback_active.clear() # 再生終了を示す（メインスレッドから停止させるので不要かも）
        # raise sd.CallbackStop # コールバック停止
        # ↑ CallbackStop を使うとウィンドウが閉じる前に音が止まる可能性があるので注意
        #   代わりに、再生位置が進まないようにするか、無音を返し続ける
        print("再生終了")
        # 実際のフレーム数を返す
        valid_frames = len(audio_data) - chunk_start
        if valid_frames > 0:
             outdata[:valid_frames] = audio_data[chunk_start:chunk_start + valid_frames]
             outdata[valid_frames:] = 0 # 残りは無音
        else:
             outdata.fill(0)
        current_audio_frame = len(audio_data) # 再生位置を最後に固定
    else:
        # 音声データを outdata にコピー
        outdata[:] = audio_data[chunk_start:chunk_end]
        current_audio_frame = chunk_end # 再生位置を更新

print("アニメーションの準備中...")
fig, ax = plt.subplots(figsize=(10, 6)) # 図のサイズはお好みで

# 周波数軸を対数にするか線形にするか (対数が一般的)
use_log_freq = True
if use_log_freq:
    ax.set_yscale('log')
    # 表示する周波数範囲（低すぎると対数表示で見にくいので下限を設定）
    min_display_freq = 20 # Hz
    max_display_freq = sample_rate / 2
    ax.set_ylim(min_display_freq, max_display_freq)
else:
    ax.set_ylim(freqs[0], freqs[-1])


# pcolormesh を使う準備 (imshowより座標の扱いが正確)
# X軸 (パン): ビンの境界を定義 (-1.5, -0.5, 0.5, 1.5 のようなイメージ)
pan_edges = np.linspace(-N_PAN_BINS/2.0, N_PAN_BINS/2.0, N_PAN_BINS + 1)
# Y軸 (周波数): ビンの境界を定義 (freqs を使う)
freq_edges = np.concatenate(([freqs[0] - (freqs[1]-freqs[0])/2], (freqs[:-1]+freqs[1:])/2, [freqs[-1]+(freqs[-1]-freqs[-2])/2]))

# 最初のフレームデータ (周波数 x パン)
initial_data = all_heatmap_data_db_clipped[:, :, 0]

# pcolormesh で初期描画 (X, Y は境界を指定)
# データは (周波数, パン) の形なので、X=pan_edges, Y=freq_edges になる
quadmesh = ax.pcolormesh(pan_edges, freq_edges, initial_data, cmap=CMAP, vmin=min_db, vmax=max_db, shading='flat')

# 軸ラベルとか設定
ax.set_xlabel("Pan")
# X軸の目盛りをラベルに設定
ax.set_xticks(np.arange(N_PAN_BINS) - N_PAN_BINS/2.0 + 0.5)
ax.set_xticklabels(PAN_LABELS)

ax.set_ylabel("Frequency (Hz)")
fig.colorbar(quadmesh, label="Level (dB)")
title = ax.set_title(f"Time: {times[0]:.2f}s")
plt.tight_layout() # レイアウトをよしなに調整
print("アニメーション準備完了.")

# --- Step 5: アニメーション更新関数定義 (変更) ---
# アニメーション更新関数 (音声同期版)
def update_sync(frame): # 引数 frame は使わない
    # 現在の音声再生時間 (コールバックで更新される global 変数から)
    # 再生が終了したら更新を止める
    if current_audio_frame >= len(audio_data) :
        # 必要ならここでアニメーションを止める処理を入れる（今はループ or 最終フレーム表示）
        # animation_active.set() みたいなので制御？ FuncAnimation の repeat=False なら勝手に止まる
        pass # 最終フレームを表示し続けるか、何もしない

    current_playback_time = current_audio_frame / sample_rate

    # その時間に最も近い分析フレームのインデックスを探す
    if current_playback_time <= times[-1]:
        time_index = np.argmin(np.abs(times - current_playback_time))
    else:
        time_index = n_times - 1 # 再生が分析時間を超えたら最後のフレーム

    # ヒートマップデータを取得して更新
    heatmap_frame_data = all_heatmap_data_db_clipped[:, :, time_index]
    quadmesh.set_array(heatmap_frame_data.ravel())
    # タイトルも更新 (再生時間基準)
    title.set_text(f"Audio Time: {current_playback_time:.2f}s (Frame: {time_index})")

    return quadmesh, title

# --- Step 6 の代わり: 音声ストリームとアニメーション表示 ---
print("音声ストリームとアニメーション表示を開始します...")

# アニメーションオブジェクト生成 (update 関数を変更)
# interval は画面更新の頻度。音声同期なので、あまり重要ではないが、描画負荷との兼ね合い。33ms = 約30fps
ani = FuncAnimation(fig, update_sync, interval=33, blit=False, repeat=False) # repeat=False 推奨

# 音声出力ストリームを開く
try:
    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=audio_data.shape[1], # ステレオ
        callback=audio_callback,
        finished_callback=playback_active.set # 再生終了時にイベントをセット (使わないかも)
    )
    # ストリームを開始し、表示 (plt.show() はブロッキングされる)
    with stream:
        print("再生中... ウィンドウを閉じると停止します。")
        plt.show() # ウィンドウ表示（ユーザーが閉じるまで待機）
    
    print("表示ウィンドウが閉じられました。")
    # stream は with ブロックを抜けたら自動的に stop/close されるはず

except Exception as e:
    print(f"エラー: 音声ストリームの開始または再生中にエラーが発生しました - {e}")

# 必要ならここで再生位置をリセット
current_audio_frame = 0 
print("プログラム終了。")
