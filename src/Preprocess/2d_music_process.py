import numpy as np
import scipy.signal as signal
import scipy.fftpack as fft
import time
import pyaudio
import queue
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 模拟参数配置
fs = 44100  # 采样频率 (Hz)
T = 40e-3  # Chirp持续时间 (s)
N_mic = 2  # 麦克风数量
duration = 0.04  # 每次采集的时间 (40ms)
bandpass_low = 1000  # 带通滤波器下限
bandpass_high = 20000  # 带通滤波器上限
K = 60  # 时域波束形成的子矩阵大小
d_mic = 0.05  # 麦克风间距
B = 2000  # 带宽
vs = 343  # 声速 (m/s)
chirp_freq_start = 18e3  # 起始频率
chirp_freq_end = chirp_freq_start + B  # 结束频率

# 信号队列，用于存储接收到的音频数据
rx_signal_queue = queue.Queue()

# 设置标志变量，控制是否使用真实数据
use_real_data = 1  # 设置为 False 使用模拟数据，设置为 True 使用真实音频数据

def measure_background_reflection(p, chirp_signal, duration=2):
    """
    测量背景反射信号，同时发射 Chirp 信号，记录静态背景噪声模板
    :param p: PyAudio实例
    :param chirp_signal: 生成的 Chirp 信号
    :param duration: 测量时长
    :return: 背景噪声模板信号
    """
    print("正在测量背景反射，请保持环境静止...")

    # 打开音频流
    stream_out = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=fs,
                        output=True)
    stream_in = p.open(format=pyaudio.paInt16,
                       channels=N_mic,
                       rate=fs,
                       input=True,
                       frames_per_buffer=int(fs * T))

    background_frames = []
    num_iterations = int(duration / T)

    for _ in range(num_iterations):
        # 发射 Chirp 信号
        stream_out.write(chirp_signal.astype(np.float32).tobytes())

        # 接收反射信号
        data = stream_in.read(int(fs * T))
        signals = np.frombuffer(data, dtype=np.int16).reshape(-1, N_mic)
        background_frames.append(signals)

    # 关闭音频流
    stream_out.close()
    stream_in.close()

    # 计算平均背景模板
    background_noise_template = np.mean(background_frames, axis=0)
    print("背景反射测量完成！")
    return background_noise_template
# 生成模拟音频数据：chirp信号（模拟发送信号和接收信号）
def generate_chirp_signal(T, fs, f0, B):
    t = np.linspace(0, T, int(T * fs))
    chirp_signal = np.cos(2 * np.pi * f0 * t + (B - f0) * t ** 2 / (2 * T))
    return chirp_signal, t

# 模拟接收到的信号（反射信号），此处假设有1个反射源
def simulate_received_signal(chirp_signal, noise_factor=0.5, num_sources=1):
    # 模拟信号的反射，假设有1个反射源
    t = np.linspace(0, T, len(chirp_signal))
    reflection_1 = np.cos(2 * np.pi * (chirp_freq_start + 500) * t)  # 第一反射源
    received_signal = reflection_1  # 总接收信号是一个源的反射

    # 加噪声
    noise = noise_factor * np.random.randn(len(t))
    received_signal += noise

    # 为四个麦克风模拟接收信号（假设每个麦克风接收到的信号相同，存在一个相位偏移）
    signals = np.zeros((len(t), N_mic))  # 每列是一个麦克风的接收信号
    for i in range(N_mic):
        phase_shift = 2 * np.pi * i * 0.1  # 假设不同麦克风的相位偏移
        signals[:, i] = received_signal * np.cos(phase_shift)
    return signals

# 2D MUSIC算法（计算伪谱图）
def music_algorithm(cov_matrix, num_sources):
    """ 2D MUSIC Algorithm """
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigvals)
    noise_subspace = eigvecs[:, sorted_indices[:-num_sources]]  # 选择噪声子空间
    P = np.zeros((60, 60))  # 假设角度范围 [-30, 30] 和距离范围 [2m, 3m]
    for i, theta in enumerate(np.linspace(-30, 30, 60)):
        for j, distance in enumerate(np.linspace(0.1, 0.4, 60)):
            # 构建方向向量
            steering_vector = np.exp(-1j * 2 * np.pi * distance * np.sin(np.deg2rad(theta)) / vs)

            # 确保点积返回标量
            dot_product = np.dot(noise_subspace.T, steering_vector)
            # 确保dot_product是标量，取第一个元素
            if np.ndim(dot_product) > 0:  # 如果返回结果是多维的
                dot_product = dot_product.flatten()[0]  # 使用 .flatten() 展平，并提取第一个元素

            P[i, j] = 1 / np.abs(dot_product) ** 2  # 使用绝对值计算伪谱图
    return P

# 播放信号的音频发射模块（循环播放）
def audio_transmit(tx_signal, p):
    stream_out = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=fs,
                        output=True,
                        frames_per_buffer=len(tx_signal))

    while True:
        stream_out.write(tx_signal.astype(np.float32).tobytes())
        time.sleep(T)  # 播放每个周期的信号

# 实时接收信号并将其存入队列
def audio_receive(p):
    stream_in = p.open(format=pyaudio.paInt16,
                       channels=N_mic,
                       rate=fs,
                       input=True,
                       frames_per_buffer=int(T * fs))

    while True:
        rx_data = stream_in.read(int(T * fs))  # 获取实时音频数据
        rx_data = np.frombuffer(rx_data, dtype=np.int16)
        signals = np.reshape(rx_data, (len(rx_data) // N_mic, N_mic))
        rx_signal_queue.put(signals)  # 将接收到的信号放入队列

# 音频数据处理模块
def process_audio_data(p,background_noise, ax, scat, line):
    start_time = time.time()

    count = 0  # 初始化计数器，用来控制处理的次数
    trajectory = []  # 用来存储目标的运动轨迹

    # 生成模拟的chirp信号
    tx_signal, t = generate_chirp_signal(T, fs, chirp_freq_start, B)

    # 启动音频发射和接收线程
    transmit_thread = threading.Thread(target=audio_transmit, args=(tx_signal, p))
    receive_thread = threading.Thread(target=audio_receive, args=(p,))

    transmit_thread.daemon = True  # 设置为守护线程，主线程退出时会自动退出
    receive_thread.daemon = True  # 设置为守护线程，主线程退出时会自动退出

    transmit_thread.start()
    receive_thread.start()

    while count < 50:  # 控制最多处理5次
        # 从队列中获取接收到的信号
        if not rx_signal_queue.empty():
            received_signal = rx_signal_queue.get()

            # 计算每次信号的处理时间
            signal_process_start = time.time()

            # 1. 带通滤波器，去除带外噪声
            nyquist = 0.5 * fs
            low = bandpass_low / nyquist
            high = bandpass_high / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_signal = signal.lfilter(b, a, received_signal)

            # 2. 傅里叶变换去负频率
            freq_signal = fft.fft(filtered_signal, axis=0)  # 对每列（每个麦克风）进行傅里叶变换
            freq_signal = freq_signal[:len(freq_signal) // 2, :]  # 去除负频率部分
            #去除背景噪声
            received_signal = received_signal - background_noise

            # 3. 乘法操作：发射信号和接收到的信号相乘
            chirp_signal = np.cos(
                2 * np.pi * chirp_freq_start * t + (chirp_freq_end - chirp_freq_start) * t ** 2 / (2 * T))
            multiplied_signal = filtered_signal * chirp_signal[:, np.newaxis]  # 对所有麦克风的信号进行乘法

            # 4. 低通滤波器：提取正弦波
            low_cut = 3000 / nyquist  # 低通滤波器截止频率
            b, a = signal.butter(4, low_cut, btype='low')
            low_pass_signal = signal.lfilter(b, a, multiplied_signal, axis=0)

            # 5. 时域波束形成：加权信号
            weighted_signal = np.zeros_like(low_pass_signal, dtype=complex)
            for i in range(N_mic):
                weighted_signal[:, i] = low_pass_signal[:, i] * np.exp(
                    -1j * 2 * np.pi * d_mic * i * np.linspace(0, 1, len(low_pass_signal)) / vs)

            # 6. 计算协方差矩阵
            cov_matrix = np.cov(weighted_signal.T)

            # 7. 2D MUSIC伪谱图估计
            P = music_algorithm(cov_matrix, 1)  # 假设估计1个信号源


            # 8. 从伪谱图中提取距离和角度
            angle, distance = np.unravel_index(np.argmax(P), P.shape)
            angle = np.deg2rad(angle - 30)  # [-30, 30] 转换为角度
            distance = 0.1 + distance * 0.3/60  # [2, 3] 距离调整

            # 9. 计算目标位置
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)

            # 更新目标轨迹
            trajectory.append((x, y))

            # 更新图形
            if len(trajectory) > 1:
                line.set_data([point[0] for point in trajectory], [point[1] for point in trajectory])

            # 绘制目标当前位置
            scat.set_offsets((x, y))
            plt.draw()

            signal_process_end = time.time()
            print(f"Processed in {signal_process_end - signal_process_start:.4f} seconds")
            count += 1

        time.sleep(0.04)  # 每次处理间隔时间

# 设置并显示动画
def update_plot():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Target Position and Trajectory")

    # 初始目标位置
    scat = ax.scatter([], [], c='r', s=100, label="Target")
    line, = ax.plot([], [], label="Trajectory")

    return fig, ax, scat, line

if __name__ == '__main__':
    # 初始化PyAudio
    p = pyaudio.PyAudio()
    chirp_signal, t = generate_chirp_signal(T, fs, chirp_freq_start, B)
    # 背景噪声提取
    background_noise = measure_background_reflection(p, chirp_signal, duration=2)

    input("按 Enter 键开始动作捕捉...")  # 用户交互确认
    # 设置图形
    fig, ax, scat, line = update_plot()

    # 启动音频处理和更新图形
    process_audio_data(p, background_noise,ax, scat, line)
    plt.show()  # 在主线程中调用plt.show()
