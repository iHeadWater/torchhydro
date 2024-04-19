import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModule(nn.Module):
    """
    定义CNN模块
    CNN模块用于从输入的多源数据中提取空间特征。这可以通过定义一个简单的CNN层来实现。
    """

    def __init__(self, in_channels, out_channels):
        super(CNNModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class TransformerModule(nn.Module):
    """
    定义transformer模块
    transformer模块用于处理时间和跨源依赖性。
    """

    def __init__(self, input_dim, num_heads, num_layers):
        super(TransformerModule, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )

    def forward(self, src, tgt):
        return self.transformer(src, tgt)


class CNNTransformerModule(nn.Module):
    """
    整合CNN和transformer模块
    整合CNN和变压器模块，确保在每个特征提取级别都应用注意力机制
    """

    def __init__(self, num_sources, cnn_output_dim, transformer_heads, transformer_layers):
        super(CNNTransformerModule, self).__init__()
        self.cnn_modules = nn.ModuleList([CNNModule(1, cnn_output_dim) for _ in range(num_sources)])
        self.transformer = TransformerModule(cnn_output_dim, transformer_heads, transformer_layers)
        self.fc = nn.Linear(cnn_output_dim, cnn_output_dim)  # 保持维度与LSTM输入维度相同

    def forward(self, x):
        cnn_outputs = [cnn(x[:, i:i + 1]) for i, cnn in
                       enumerate(self.cnn_modules)]  # 假设x的形状是[batch_size, num_sources, height, width]
        combined = torch.cat(cnn_outputs, dim=1)  # 沿特征维合并
        combined = combined.permute(2, 3, 0, 1).contiguous()  # 调整为变压器所需的维度[seq_len, batch, features]
        transformer_output = self.transformer(combined, combined)
        transformer_output = transformer_output.permute(2, 0, 1, 3).contiguous()
        out = self.fc(transformer_output[:, :, :, 0])  # 取序列的第一个元素作为输出
        return out.squeeze(-1)  # 确保输出是一个二维张量 [batch_size, seq_len]


class FusionModel(nn.Module):
    """
    创建降雨融合模型
    该模型将CNNTransformer和dpl4xaj模块整合在一起，确保数据流从CNNTransformer到dpl4xaj。
    整合CNN和变压器模块，确保在每个特征提取级别都应用注意力机制
    """

    def __init__(self, CNNTransformer_model, dpl4xaj_model):
        super(FusionModel, self).__init__()
        self.CNNTransformer_model = CNNTransformer_model
        self.dpl4xaj_model = dpl4xaj_model

    def forward(self, x):
        CNNTransformer_output = self.CNNTransformer_model(x)
        CNNTransformer_output = CNNTransformer_output.unsqueeze(1)  # 增加一个序列长度的维度
        dpl4xaj_output = self.dpl4xaj_model(CNNTransformer_output)
        return dpl4xaj_output
