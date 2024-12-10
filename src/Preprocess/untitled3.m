% 超高速2D MUSIC算法优化：发射与接收
function ultra_fast_2d_music()
    % 关闭不必要的警告和性能开销
    warning('off', 'all');
    feature('numThreads', feature('numCores'));  % 利用所有CPU核心

    % 参数配置
    PARAMS = struct(... 
        'f0', 18e3, ...       % 起始频率 (Hz)
        'B', 2000, ...        % 带宽 (Hz)
        'T', 40e-3, ...       % Chirp持续时间 (s)
        'fs', 44e3, ...       % 采样频率 (Hz)
        'vs', 343, ...        % 声速 (m/s)
        'N_mic', 4, ...       % 麦克风数量
        'd_mic', 0.05, ...    % 麦克风间距 (m)
        'SNR', 10, ...        % 信噪比 (dB)
        'n_iter', 10 ...      % 迭代次数（发射和接收信号的次数）
    );

    % 初始化存储伪谱图的矩阵
    P_all = [];

    % 启动音频设备
    audioReader = audioDeviceReader('SamplesPerFrame', round(PARAMS.fs*PARAMS.T), 'NumChannels', PARAMS.N_mic, 'SampleRate', PARAMS.fs);

    % 无限循环，直到手动停止
    iter = 1; % 计数迭代次数
    while true
        % 生成发射信号：40ms的线性频率增加信号
        [tx_signal, t] = generate_chirp_signal(PARAMS);

        % 向接收器发送信号并接收
        disp(['开始接收第 ', num2str(iter), ' 组信号...']);
        rx_signals = audioReader();  % 从设备读取接收信号
        
        % 计算伪谱图（MUSIC算法）
        tic;
        [P, theta_range, d_range] = fast_2d_music(rx_signals, PARAMS);
        computation_time = toc;

        % 将伪谱图 P 储存到全局存储
        P_all = cat(3, P_all, P);

        % 输出计算时间
        fprintf('第 %d 组信号计算时间: %.4f 秒\n', iter, computation_time);

        % 可视化每组伪谱图
        visualize_results(P, d_range, theta_range, iter);
        
        iter = iter + 1;  % 更新迭代次数
        
        % 检查用户是否按下Ctrl+C停止程序（或者设置其他终止条件）
        if ~ishandle(1)  % 监测图形窗口关闭
            disp('程序已停止。');
            break;  % 退出循环
        end
        
        % 确保设备被正确释放
        release(audioReader);
    end
end

% 快速啁啾信号生成
function [tx_signal, t] = generate_chirp_signal(PARAMS)
    t = 0:1/PARAMS.fs:PARAMS.T-1/PARAMS.fs;  % 时间向量
    tx_signal = chirp(t, PARAMS.f0, PARAMS.T, PARAMS.f0 + PARAMS.B, 'linear');  % 线性调频信号
end

% 2D MUSIC算法（伪谱计算）
function [P, theta_range, d_range] = fast_2d_music(rx_signals, PARAMS)
    % 减少搜索空间
    theta_range = linspace(-90, 90, 180);  % 角度范围
    d_range = linspace(1, 5, 100);  % 距离范围
    
    % 协方差矩阵快速估计
    R = rx_signals * rx_signals' / size(rx_signals, 2);  % 协方差矩阵
    
    % 特征值分解
    [V, ~] = eigs(R, 2, 'smallestreal');
    noise_subspace = V;
    
    % 初始化伪谱图矩阵
    P = zeros(length(theta_range), length(d_range));
    
    % 计算伪谱图
    parfor i = 1:length(theta_range)
        local_P = zeros(1, length(d_range));
        
        for j = 1:length(d_range)
            % 构建方向向量
            steering_vector = build_steering_vector(theta_range(i), d_range(j), PARAMS);
            
            % 计算伪谱值
            local_P(j) = 1 / (norm(noise_subspace' * steering_vector)^2 + eps);
        end
        
        P(i, :) = local_P;
    end
end

% 构建方向向量
function steering_vector = build_steering_vector(theta, distance, PARAMS)
    lambda = PARAMS.vs / PARAMS.f0;  % 波长
    u = exp(-1j * 2 * pi * PARAMS.d_mic * cos(deg2rad(theta)) / lambda * (0:PARAMS.N_mic-1)');
    v = exp(1j * 4 * pi * PARAMS.B * distance / (PARAMS.T * PARAMS.vs) * (0:length(u)-1)');
    steering_vector = u .* v;  % 元素级乘法
end

% 结果可视化
function visualize_results(P, d_range, theta_range, iter)
    figure(1);  % 在同一窗口中更新伪谱图
    imagesc(d_range, theta_range, 10*log10(abs(P)));  % 以对数形式显示伪谱图
    xlabel('距离 (m)');
    ylabel('角度 (°)');
    title(['2D MUSIC 反射物体定位 - 第 ', num2str(iter), ' 组']);
    colorbar;
    axis xy;
    drawnow;
end

% 主程序入口
ultra_fast_2d_music();
