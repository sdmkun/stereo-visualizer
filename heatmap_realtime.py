import numpy as np
# import soundfile as sf # 不要になる
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
import sounddevice as sd
import time
# import threading # 不要になる
import queue # リアルタイムデータ受け渡し用に queue を使う

# --- 設定 ---
# マイク入力用の設定を追加
INPUT_DEVICE_INDEX = 89 # Noneならデフォルト入力デバイス
BUFFER_DURATION_SEC = 1.0 # 音声データを溜めておくバッファの長さ（秒） ※STFTの時間解像度や応答性に影響
BLOCK_DURATION_MSEC = 50 # マイクから音声を受け取る間隔（ミリ秒）

# STFTパラメータ (リアルタイム用に調整が必要かも)
N_PERSEG = 1024
N_OVERLAP = N_PERSEG // 2 # リアルタイムでSTFTする場合、オーバーラップ管理がキモ
WINDOW = 'hann'

# ヒートマップ・表示パラメータ (FPSは描画速度に依存)
N_PAN_BINS = 21 # ★リアルタイムでは少なめから試すのが吉 (例: 11 or 21)
TARGET_FPS = 20 # ★描画と計算が間に合う範囲で設定 (例: 15-30)
CMAP = 'magma'
DB_RANGE = 60
PAN_LABELS = np.linspace(-1, 1, 5) # X軸の目盛り表示例 (-1, -0.5, 0, 0.5, 1)

use_log_freq = True  # True: 対数スケール, False: 線形スケール

# グローバル変数 / データキュー
audio_buffer = None
sample_rate = 0
data_queue = queue.Queue() # スレッド間でデータを安全に受け渡すためのキュー
last_heatmap_data = None # 表示更新用に最後のデータを保持

# --- 音声入力コールバック ---
def audio_callback(indata, frames, time_info, status):
    """マイク入力があるたびに呼び出され、データをキューに入れる"""
    if status:
        print(status)
    # 入力データ (L/R) をキューに入れる
    data_queue.put(indata.copy())

# --- パン分布計算関数 ---
# (Step 3 の計算ロジックを関数化しておくと見通しが良い)
def calculate_pan_distribution(Zxx_L_slice, Zxx_R_slice, n_freqs, n_pan_bins):
    heatmap_slice = np.zeros((n_freqs, n_pan_bins))
    pan_bin_centers = np.linspace(-1.0, 1.0, n_pan_bins)
    bin_width = 2.0 / (n_pan_bins - 1) if n_pan_bins > 1 else 1.0
    sigma = bin_width * 1.0 # ★調整ポイント
    epsilon = 1e-10

    for f in range(n_freqs):
        L = Zxx_L_slice[f]
        R = Zxx_R_slice[f]
        mag_L = np.abs(L) + epsilon
        mag_R = np.abs(R) + epsilon
        phi = np.arctan(mag_R / mag_L)
        p = np.clip(1.0 - (4.0 / np.pi) * phi, -1.0, 1.0)
        P_total = mag_L**2 + mag_R**2
        weights = np.exp(-((pan_bin_centers - p)**2) / (2 * sigma**2))
        sum_weights = np.sum(weights) + epsilon
        normalized_weights = weights / sum_weights
        heatmap_slice[f, :] = P_total * normalized_weights
    return heatmap_slice

# --- アニメーション更新関数 ---
# (FuncAnimation から呼び出される)
def update_realtime(frame):
    global audio_buffer, last_heatmap_data
    
    new_data_block = None
    # キューから可能な限りデータを取得してバッファを更新
    while not data_queue.empty():
        new_data_block = data_queue.get_nowait()
        if audio_buffer is None:
             # 最初のデータでバッファを初期化
             buffer_size = int(BUFFER_DURATION_SEC * sample_rate)
             audio_buffer = np.zeros((buffer_size, new_data_block.shape[1]), dtype='float32')
        
        # 古いデータを捨てて新しいデータを末尾に追加（リングバッファ）
        shift_size = new_data_block.shape[0]
        audio_buffer = np.roll(audio_buffer, -shift_size, axis=0)
        audio_buffer[-shift_size:, :] = new_data_block

    # バッファがまだ満たされていない、または新しいデータがなければ更新しない
    if audio_buffer is None or new_data_block is None:
         # まだ描画できるデータがない場合は、最後に成功したデータを再描画
         if last_heatmap_data is not None:
             quadmesh.set_array(last_heatmap_data.ravel())
             return quadmesh, title
         else:
             return [] # 何も返さない (あるいは初期画像を返す)

    # --- 最新のバッファデータで STFT と計算 ---
    # バッファ全体でSTFTを実行し、最新のタイムスライスのみを使用
    # (注意：これは計算効率が良くない。ストリーミングSTFTが理想)
    try:
        audio_L_buf = audio_buffer[:, 0]
        audio_R_buf = audio_buffer[:, 1]
        
        # STFTを実行（ここでは buffer 全体に対して行い、最後のフレームだけ使う簡易版）
        # overlap を正しく扱うには、もっと工夫が必要（状態を保持するなど）
        _freqs, _times, Zxx_L = signal.stft(audio_L_buf, fs=sample_rate, window=WINDOW, nperseg=N_PERSEG, noverlap=N_OVERLAP)
        _, _, Zxx_R = signal.stft(audio_R_buf, fs=sample_rate, window=WINDOW, nperseg=N_PERSEG, noverlap=N_OVERLAP)
        
        # 最新のタイムスライスを取得
        Zxx_L_latest = Zxx_L[:, -1]
        Zxx_R_latest = Zxx_R[:, -1]
        n_freqs_current = Zxx_L.shape[0] # STFT結果の周波数ビン数を取得

        # パン分布計算
        heatmap_now = calculate_pan_distribution(Zxx_L_latest, Zxx_R_latest, n_freqs_current, N_PAN_BINS)

        # dB変換 & 正規化 (固定範囲を使う)
        heatmap_db = 10 * np.log10(heatmap_now + 1e-10)
        # リアルタイムでは最大値が変動するので、固定のmax/minを使うのが無難
        # 例えば、想定される入力レベルから決める (ここでは仮に 0dB を最大とする)
        max_db_realtime = 0
        min_db_realtime = max_db_realtime - DB_RANGE
        heatmap_clipped = np.clip(heatmap_db, min_db_realtime, max_db_realtime)
        
        # 表示更新
        # quadmesh の Y軸 (freq_edges) が STFT の結果と合っているか注意
        # もし n_freqs_current が初期値と違う場合は再設定が必要だが、通常は同じはず
        quadmesh.set_array(heatmap_clipped.ravel())
        quadmesh.set_clim(vmin=min_db_realtime, vmax=max_db_realtime) # 色の範囲も更新
        title.set_text("Real-time Input") # タイトル変更
        
        last_heatmap_data = heatmap_clipped # 最後に成功したデータを保持

    except Exception as e:
        print(f"エラー: update 中 - {e}")
        # エラーが起きても最後に成功したデータを表示し続ける
        if last_heatmap_data is not None:
             quadmesh.set_array(last_heatmap_data.ravel())
             return quadmesh, title
        else:
             return []

    return quadmesh, title

# --- メイン処理 ---
try:
    # デフォルトのサンプルレートとデバイス情報を取得
    sd.check_input_settings(device=INPUT_DEVICE_INDEX, channels=2) # ステレオ入力可能かチェック
    device_info = sd.query_devices(device=INPUT_DEVICE_INDEX, kind='input')
    sample_rate = int(device_info['default_samplerate'])
    print(f"入力デバイス: {device_info['name']}, サンプルレート: {sample_rate} Hz")

    # ブロックサイズ計算
    blocksize = int(sample_rate * BLOCK_DURATION_MSEC / 1000)

    # Matplotlib の準備 (周波数軸は開始時点では不明なので、後で設定か、仮設定)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # STFT を一度ダミーデータで実行して freqs を取得（表示のため）
    dummy_signal = np.zeros(N_PERSEG)
    freqs, _, _ = signal.stft(dummy_signal, fs=sample_rate, nperseg=N_PERSEG)
    n_freqs_init = len(freqs)
    
    # pcolormesh の Y軸 (周波数) の境界を計算
    if n_freqs_init > 1:
        freq_step = freqs[1]-freqs[0]
        freq_edges = np.concatenate(([freqs[0] - freq_step/2], (freqs[:-1]+freqs[1:])/2, [freqs[-1]+freq_step/2]))
    elif n_freqs_init == 1:
        freq_edges = np.array([freqs[0] * 0.8, freqs[0] * 1.2]) # 適当な幅
    else:
        freq_edges = np.array([0, 1]) # データがない場合

    pan_edges = np.linspace(-(N_PAN_BINS+1)/2.0, (N_PAN_BINS+1)/2.0, N_PAN_BINS + 1) # pcolormesh用境界

    # 初期表示データ (ゼロ)
    initial_data = np.zeros((n_freqs_init, N_PAN_BINS))
    min_db_realtime_init = -DB_RANGE # 初期表示の最小dB
    max_db_realtime_init = 0       # 初期表示の最大dB

    quadmesh = ax.pcolormesh(pan_edges, freq_edges, initial_data, cmap=CMAP, vmin=min_db_realtime_init, vmax=max_db_realtime_init, shading='flat')

    # 軸設定
    ax.set_xlabel("Pan")
    ax.set_xticks(np.linspace(-(N_PAN_BINS)/2.0 + 0.5, (N_PAN_BINS)/2.0 - 0.5, len(PAN_LABELS)))
    ax.set_xticklabels([f"{p:.1f}" for p in PAN_LABELS]) # -1 から 1 のラベルを表示
    
    ax.set_ylabel("Frequency (Hz)")
    if use_log_freq: # 設定変数 use_log_freq を定義しておく必要あり
        ax.set_yscale('log')
        min_display_freq = 20
        max_display_freq = sample_rate / 2
        ax.set_ylim(min_display_freq, max_display_freq)
    else:
         # 最初の freqs 配列を元に Y リミット設定
        if n_freqs_init > 0:
             ax.set_ylim(freqs[0], freqs[-1])
        else:
             ax.set_ylim(0, sample_rate/2)


    fig.colorbar(quadmesh, label="Level (dB)")
    title = ax.set_title("Real-time Input (Waiting for data...)")
    plt.tight_layout()
    
    # アニメーションオブジェクト生成
    ani = FuncAnimation(fig, update_realtime, interval=1000/TARGET_FPS, blit=False, cache_frame_data=False) # interval はミリ秒

    # マイク入力ストリームを開始
    with sd.InputStream(
            device=INPUT_DEVICE_INDEX, 
            channels=2, # ステレオ入力
            samplerate=sample_rate, 
            blocksize=blocksize, 
            dtype='float32', 
            callback=audio_callback):
        
        print("リアルタイム解析を開始しました。ウィンドウを閉じると終了します。")
        plt.show() # アニメーション表示開始（ユーザーが閉じるまで待機）

    print("プログラム終了。")

except Exception as e:
    print(f"エラーが発生しました: {e}")
    # traceback.print_exc() # 詳細なエラー表示が必要な場合