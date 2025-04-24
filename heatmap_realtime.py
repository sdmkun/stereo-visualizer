import numpy as np
# import soundfile as sf # 不要
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
import sounddevice as sd
import time
import queue # リアルタイムデータ受け渡し用に queue を使う
# import threading # 不要
import traceback # エラー詳細表示用

# 計算中の警告（特にlog10(0)）を一旦無視する場合
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- 設定 ---
# マイク入力用の設定を追加
INPUT_DEVICE_INDEX = 89 # Noneならデフォルト入力デバイス
BUFFER_DURATION_SEC = 0.5 # ★バッファ長。短いほど応答性は良いが、低周波数の解像度が下がる可能性。
BLOCK_DURATION_MSEC = 20 # マイクから音声を受け取る間隔（ミリ秒）

# STFTパラメータ (リアルタイム用に調整が必要)
N_PERSEG = 1024  # FFTポイント数。減らすと時間解像度↑周波数解像度↓
N_OVERLAP = N_PERSEG * 3 // 4 # ★オーバーラップを増やす (例: 75%)。計算負荷↑
WINDOW = 'hann'

# ヒートマップ・表示パラメータ
N_PAN_BINS = 21 # パン分割数。減らすと計算負荷↓
TARGET_FPS = 30 # ★目標FPS。高くすると計算が間に合わない可能性↑
CMAP = 'magma'
DB_RANGE = 60
PAN_LABELS = np.linspace(-1, 1, 5) # X軸の目盛り表示例 (-1, -0.5, 0, 0.5, 1)
use_log_freq = True  # True: 対数スケール, False: 線形スケール
# --- 設定ここまで ---

# グローバル変数 / データキュー
audio_buffer = None
sample_rate = 0
data_queue = queue.Queue()
last_heatmap_data = None
n_freqs_init = 0 # 初期化時に周波数ビン数を保持
freqs = None # 周波数軸データも保持

# --- 音声入力コールバック ---
def audio_callback(indata, frames, time_info, status):
    """マイク入力があるたびに呼び出され、データをキューに入れる"""
    if status:
        print(status, flush=True) # エラー表示
    data_queue.put(indata.copy())

# --- パン分布計算関数 ---
# (変更なし)
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
def update_realtime(frame):
    global audio_buffer, last_heatmap_data

    new_data_received = False
    # キューから可能な限りデータを取得してバッファを更新
    while not data_queue.empty():
        try:
            new_data_block = data_queue.get_nowait()
            new_data_received = True # 新しいデータを受け取ったフラグ

            if audio_buffer is None:
                 # 最初のデータでバッファを初期化
                 buffer_size = int(BUFFER_DURATION_SEC * sample_rate)
                 if buffer_size < N_PERSEG:
                      print(f"警告: バッファ長 ({BUFFER_DURATION_SEC}s) が STFTセグメント長 ({N_PERSEG/sample_rate:.3f}s) より短いです。長くしてください。")
                      buffer_size = N_PERSEG # 最低限必要なサイズに
                 audio_buffer = np.zeros((buffer_size, new_data_block.shape[1]), dtype='float32')

            # 古いデータを捨てて新しいデータを末尾に追加（リングバッファ）
            shift_size = new_data_block.shape[0]
            if shift_size > audio_buffer.shape[0]:
                 # 入力ブロックがバッファより大きい場合 (エラーケース)
                 print(f"警告: 入力ブロックサイズ ({shift_size}) がバッファサイズ ({audio_buffer.shape[0]}) を超えています。")
                 # バッファを最新データで埋める (一部データ損失)
                 audio_buffer = new_data_block[-audio_buffer.shape[0]:, :]
            else:
                 audio_buffer = np.roll(audio_buffer, -shift_size, axis=0)
                 audio_buffer[-shift_size:, :] = new_data_block

        except queue.Empty:
            break # キューが空になったらループを抜ける
        except Exception as e:
            print(f"エラー: キュー処理中 - {e}", flush=True)
            break

    # バッファがまだ満たされていない場合は更新しない
    if audio_buffer is None:
         return [] # 何も描画しない

    # --- 新しいデータが来た場合のみ STFT と計算を実行 ---
    # またはバッファが満たされていて、まだ計算結果がない場合も実行
    if new_data_received or last_heatmap_data is None:
        try:
            # バッファに十分なデータがあるか確認
            # STFTは少なくとも N_PERSEG の長さが必要
            if audio_buffer.shape[0] < N_PERSEG:
                 # print("バッファデータ不足、計算スキップ")
                 return [] # データ不足なら計算しない

            # STFTを実行（バッファ全体に対して。効率は悪い）
            audio_L_buf = audio_buffer[:, 0]
            audio_R_buf = audio_buffer[:, 1]

            _freqs, _times, Zxx_L = signal.stft(audio_L_buf, fs=sample_rate, window=WINDOW, nperseg=N_PERSEG, noverlap=N_OVERLAP)
            _, _, Zxx_R = signal.stft(audio_R_buf, fs=sample_rate, window=WINDOW, nperseg=N_PERSEG, noverlap=N_OVERLAP)

            if Zxx_L.shape[1] == 0: # STFT結果が空の場合がある
                 # print("警告: STFT結果が空です。")
                 # 最後に成功したデータを表示し続ける
                 if last_heatmap_data is not None:
                     quadmesh.set_array(last_heatmap_data.ravel())
                     return quadmesh, title
                 else:
                     return []

            # 最新のタイムスライスを取得
            Zxx_L_latest = Zxx_L[:, -1]
            Zxx_R_latest = Zxx_R[:, -1]
            n_freqs_current = Zxx_L.shape[0]

            # 周波数ビン数が初期値と異なる場合のエラーチェック (通常は発生しないはず)
            if n_freqs_current != n_freqs_init:
                 print(f"エラー: 周波数ビン数が変化しました ({n_freqs_init} -> {n_freqs_current})")
                 # ここでリセット処理などが必要になるが、今回は単純にスキップ
                 return []

            # パン分布計算
            heatmap_now = calculate_pan_distribution(Zxx_L_latest, Zxx_R_latest, n_freqs_current, N_PAN_BINS)

            # dB変換 & 正規化 (固定範囲を使う)
            heatmap_db = 10 * np.log10(heatmap_now + 1e-10)
            max_db_realtime = 0 # 仮の最大値 (入力レベルに応じて調整が必要かも)
            min_db_realtime = max_db_realtime - DB_RANGE
            heatmap_clipped = np.clip(heatmap_db, min_db_realtime, max_db_realtime)

            # 最後に計算成功したデータを保持
            last_heatmap_data = heatmap_clipped

        except Exception as e:
            print(f"エラー: update 中の計算エラー - {e}", flush=True)
            # 計算エラーが起きても、最後に成功したデータを表示し続ける
            if last_heatmap_data is None:
                 # 初回計算でエラーの場合、最小dB値で埋めたデータを表示
                 last_heatmap_data = np.full((n_freqs_init, N_PAN_BINS), min_db_realtime)

    # --- 描画更新 ---
    # (計算が行われなかった場合も、前回成功したデータで描画)
    if last_heatmap_data is not None:
        quadmesh.set_array(last_heatmap_data.ravel())
        quadmesh.set_clim(vmin=min_db_realtime, vmax=max_db_realtime) # 色範囲も更新
        title.set_text("Real-time Input")
        return quadmesh, title
    else:
        # 描画できるデータがまだない場合
        return []

# --- メイン処理 ---
try:
    # デフォルトのサンプルレートとデバイス情報を取得
    sd.check_input_settings(device=INPUT_DEVICE_INDEX, channels=2) # ステレオ入力可能かチェック
    device_info = sd.query_devices(device=INPUT_DEVICE_INDEX, kind='input')
    sample_rate = int(device_info['default_samplerate'])
    print(f"入力デバイス: {device_info['name']}, サンプルレート: {sample_rate} Hz")

    # ブロックサイズ計算
    blocksize = int(sample_rate * BLOCK_DURATION_MSEC / 1000)

    # Matplotlib の準備
    fig, ax = plt.subplots(figsize=(10, 6)) # Figureサイズはここで調整可能

    # STFT を一度ダミーデータで実行して freqs を取得（表示のため）
    dummy_signal = np.zeros(N_PERSEG)
    freqs, _, _ = signal.stft(dummy_signal, fs=sample_rate, nperseg=N_PERSEG)
    n_freqs_init = len(freqs) # グローバル変数に保存

    # pcolormesh の Y軸 (周波数) の境界を計算
    if n_freqs_init > 1:
        freq_step = freqs[1]-freqs[0]
        freq_edges = np.concatenate(([freqs[0] - freq_step/2], (freqs[:-1]+freqs[1:])/2, [freqs[-1]+freq_step/2]))
        # Y軸の下限がゼロ以下にならないように調整
        freq_edges[0] = max(freq_edges[0], 1e-1) # ゼロや負の値を避ける (logスケール用)
    elif n_freqs_init == 1:
        freq_edges = np.array([max(freqs[0] * 0.8, 1e-1), freqs[0] * 1.2]) # 適当な幅、ゼロ回避
    else: # 周波数ビンがゼロの場合 (N_PERSEGが小さすぎるなど)
        freq_edges = np.array([1e-1, 1]) # ダミー、ゼロ回避
        print("警告: 周波数ビン数がゼロです。N_PERSEGを確認してください。")

    # pcolormesh の X軸 (パン) の境界を計算
    # -1.0 から 1.0 の範囲を N_PAN_BINS 個に分割するための境界
    # (pcolormesh は N+1 個の境界を必要とする)
    pan_edges = np.linspace(-1.0 - (1.0/N_PAN_BINS), 1.0 + (1.0/N_PAN_BINS), N_PAN_BINS + 1)

    # 初期表示データ (最小dB値で埋める)
    initial_data = np.full((n_freqs_init, N_PAN_BINS), -DB_RANGE)
    min_db_realtime_init = -DB_RANGE
    max_db_realtime_init = 0

    quadmesh = ax.pcolormesh(pan_edges, freq_edges, initial_data, cmap=CMAP, vmin=min_db_realtime_init, vmax=max_db_realtime_init, shading='flat')

    # 軸設定
    ax.set_xlabel("Pan")
    # X軸の目盛りを設定 (-1 から 1 の範囲で表示)
    # pcolormeshのX座標はビンのインデックスではないので注意
    # -1.0 から 1.0 の範囲にラベルを配置する
    pan_tick_values = np.linspace(-1.0, 1.0, len(PAN_LABELS))
    # 対応する pcolormesh の座標を計算 (pan_edges の中心あたり)
    # pan_tick_coords = np.linspace(pan_edges[0] + (pan_edges[1]-pan_edges[0])/2 , pan_edges[-1] - (pan_edges[1]-pan_edges[0])/2, len(PAN_LABELS))
    # ↑複雑なので、単純に値で指定する
    ax.set_xticks(pan_tick_values) # X軸の目盛り位置を -1.0 から 1.0 で指定
    ax.set_xticklabels([f"{p:.1f}" for p in PAN_LABELS]) # -1 から 1 のラベルを表示

    ax.set_ylabel("Frequency (Hz)")
    if use_log_freq:
        ax.set_yscale('log')
        min_display_freq = 20
        max_display_freq = sample_rate / 2
        # Y軸の下限が 0 以下にならないようにクリップ
        y_min_limit = max(min_display_freq, freq_edges[0]) # freq_edges[0]が計算済みのはず
        ax.set_ylim(y_min_limit, max_display_freq)
    else:
        if n_freqs_init > 0:
             ax.set_ylim(freqs[0], freqs[-1])
        else:
             ax.set_ylim(0, sample_rate/2)

    # ★★★ X軸（パン軸）の表示を反転させる ★★★
    ax.invert_xaxis()

    # ★★★ アスペクト比を自動に設定 ★★★
    ax.set_aspect('auto')

    fig.colorbar(quadmesh, label="Level (dB)")
    title = ax.set_title("Real-time Input (Waiting for data...)")
    plt.tight_layout() # アスペクト比設定の後に実行

    # アニメーションオブジェクト生成
    ani = FuncAnimation(fig, update_realtime, interval=max(1, 1000//TARGET_FPS), blit=False, cache_frame_data=False) # interval はミリ秒, 0以下にならないように

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

except sd.PortAudioError as e:
    print(f"エラー: オーディオデバイスに問題があります - {e}")
    print("マイクが接続され、ステレオ入力が有効になっているか確認してください。")
    print("利用可能なデバイス:")
    print(sd.query_devices())
except Exception as e:
    print(f"予期せぬエラーが発生しました: {e}")
    traceback.print_exc() # 詳細なエラー表示
