import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

import time  # 用于统计时间

class CustomStage(nn.Module):
    def __init__(self, input_channels, input_height, input_width, pooling_size, h1_hidden, h2_hidden, output_dim, ns, nc):
        super(CustomStage, self).__init__()
        
        # 保存输入尺寸和池化参数
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.pooling_size = pooling_size
        
        # 动态计算池化后的尺寸
        self.pooling = nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_size)
        pooled_height = input_height // pooling_size
        pooled_width = input_width // pooling_size
        input_size = input_channels * pooled_height * pooled_width

        # 初始化各层
        self.fc_h1 = nn.Linear(input_size, h1_hidden)  # H1 层
        self.fc_h2 = nn.Linear(h1_hidden + ns * nc, h2_hidden)  # H2 层
        self.fc_dist = nn.Linear(h2_hidden, output_dim)  # 输出层 O 的 Dist
        self.fc_aoa = nn.Linear(h2_hidden, output_dim)  # 输出层 O 的 AoA
        
        # Context 层初始化 (默认 ns x nc 输出，后续由用户自定义)
        self.context_layer = nn.Linear(h2_hidden, ns * nc)
        self.ns = ns
        self.nc = nc

    def forward(self, x, context):
        """
        输入:
        x: (batch_size, channels, height, width) - 当前阶段的输入图片
        context: (batch_size, ns, nc) - 上一阶段的上下文输出
        """
        batch_size = x.size(0)
        
        # Pooling 层
        x = self.pooling(x)  # (batch_size, channels, new_height, new_width)
        x = x.view(batch_size, -1)  # 展平为 (batch_size, input_size)
        
        # H1 层
        h1_out = F.relu(self.fc_h1(x))  # (batch_size, h1_hidden)

        # 将 H1 的输出和 Context 拼接
        context_flattened = context.view(batch_size, -1)  # 展平 Context 为 (batch_size, ns*nc)
        h2_input = torch.cat((h1_out, context_flattened), dim=1)  # 拼接后的输入

        # H2 层
        h2_out = F.relu(self.fc_h2(h2_input))  # (batch_size, h2_hidden)

        # 输出层 O
        dist = self.fc_dist(h2_out)  # (batch_size, output_dim)
        aoa = self.fc_aoa(h2_out)  # (batch_size, output_dim)

        # 上下文层 C
        context_output = self.context_layer(h2_out).view(batch_size, self.ns, self.nc)  # (batch_size, ns, nc)

        return dist, aoa, context_output


class MultiStageRNN(nn.Module):
    def __init__(self, num_stages, input_channels, input_height, input_width, pooling_size, h1_hidden, h2_hidden, output_dim, ns, nc):
        super(MultiStageRNN, self).__init__()
        self.num_stages = num_stages
        
        # 构建多个阶段
        self.stages = nn.ModuleList([
            CustomStage(input_channels, input_height, input_width, pooling_size, h1_hidden, h2_hidden, output_dim, ns, nc)
            for _ in range(num_stages)
        ])

    def forward(self, x):
        """
        输入:
        x: (batch_size, seq_len, channels, height, width) - 输入图片序列
        """
        batch_size, seq_len, channels, height, width = x.size()
        context = torch.zeros(batch_size, self.stages[0].ns, self.stages[0].nc, device=x.device)  # 初始化 Context
        
        outputs = []
        
        for i in range(self.num_stages):
            dist_list, aoa_list = [], []
            for t in range(seq_len):
                # 当前时间步输入
                img = x[:, t, :, :, :]  # (batch_size, channels, height, width)
                
                # 当前阶段计算
                dist, aoa, context = self.stages[i](img, context)
                
                dist_list.append(dist.unsqueeze(1))  # 保留时间维度
                aoa_list.append(aoa.unsqueeze(1))

            # 拼接当前阶段的所有时间步结果
            dist_all = torch.cat(dist_list, dim=1)  # (batch_size, seq_len, output_dim)
            aoa_all = torch.cat(aoa_list, dim=1)  # (batch_size, seq_len, output_dim)
            
            outputs.append((dist_all, aoa_all))
        
        return outputs


# 定义参数
num_stages = 100  # 阶段数
input_channels = 1  # 输入图片通道数
input_height = 128  # 输入图片高度
input_width = 128  # 输入图片宽度
pooling_size = 4  # 池化窗口大小 BxB
h1_hidden = 128  # H1 隐藏层大小
h2_hidden = 64  # H2 隐藏层大小
output_dim = 1  # Dist 和 AoA 的输出大小
ns, nc = 4, 4  # 上下文输出尺寸 (ns x nc)

# 模型初始化
model = MultiStageRNN(num_stages, input_channels, input_height, input_width, pooling_size, h1_hidden, h2_hidden, output_dim, ns, nc)

# 示例输入
batch_size = 1
seq_len = 1
x = torch.randn(batch_size, seq_len, input_channels, input_height, input_width)  # (batch_size, seq_len, channels, height, width)

# 前向传播
start_time = time.time()
outputs = model(x)
end_time = time.time()
# 打印输出
for i, (dist, aoa) in enumerate(outputs):
    print(f"Stage {i+1} - Dist shape: {dist.shape}, AoA shape: {aoa.shape}")
print("total time: ", end_time - start_time)