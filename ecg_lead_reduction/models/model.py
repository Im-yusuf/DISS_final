import torch
import torch.nn as nn

from ecg_lead_reduction.core.config import (
    CNN_LSTM_FILTERS, CNN_LSTM_KERNEL, DROPOUT_RATE,
    LSTM_DROPOUT, LSTM_HIDDEN, LSTM_LAYERS,
    RESNET_BASE_FILTERS, RESNET_KERNEL_SIZE, RESNET_NUM_BLOCKS,
    SE_REDUCTION, USE_SE_BLOCK,
)


class SEBlock1D(nn.Module):

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:

        batch_size, channel_count, _ = features.shape
        channel_weights = self.squeeze(features).view(batch_size, channel_count)
        channel_weights = self.excitation(channel_weights).view(batch_size, channel_count, 1)
        return features * channel_weights


class ResidualBlock1D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 15, stride: int = 1,
                 dropout: float = 0.3,
                 use_se: bool = False, se_reduction: int = 16):
        super().__init__()
        same_padding = (kernel_size - 1) // 2


        self.bn1   = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=same_padding, bias=False)

        self.bn2     = nn.BatchNorm1d(out_channels)
        self.relu2   = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv1d(out_channels, out_channels, kernel_size,
                                 stride=1, padding=same_padding, bias=False)


        self.se = SEBlock1D(out_channels, se_reduction) if use_se else nn.Identity()


        self.skip: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        identity = self.skip(features)

        block_output = self.bn1(features)
        block_output = self.relu1(block_output)
        block_output = self.conv1(block_output)

        block_output = self.bn2(block_output)
        block_output = self.relu2(block_output)
        block_output = self.dropout(block_output)
        block_output = self.conv2(block_output)

        block_output = self.se(block_output)

        return block_output + identity


class ECGResNet(nn.Module):

    def __init__(self, num_leads: int, num_classes: int,
                 base_filters: int = RESNET_BASE_FILTERS,
                 num_blocks: int = RESNET_NUM_BLOCKS,
                 kernel_size: int = RESNET_KERNEL_SIZE,
                 dropout: float = DROPOUT_RATE,
                 use_se: bool = USE_SE_BLOCK,
                 se_reduction: int = SE_REDUCTION):
        super().__init__()


        self.input_conv = nn.Sequential(
            nn.Conv1d(num_leads, base_filters, kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
        )


        residual_layers: list[nn.Module] = []
        input_channels = base_filters
        for block_index in range(num_blocks):
            output_channels = min(base_filters * (2 ** (block_index // 2)), 256)
            stride = 2 if (block_index % 2 == 1) else 1
            residual_layers.append(ResidualBlock1D(
                input_channels, output_channels, kernel_size, stride, dropout,
                use_se=use_se, se_reduction=se_reduction,
            ))
            input_channels = output_channels

        self.res_blocks = nn.Sequential(*residual_layers)


        self.bn_final   = nn.BatchNorm1d(input_channels)
        self.relu_final = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(input_channels, num_classes)


        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.input_conv(features)
        features = self.res_blocks(features)
        features = self.bn_final(features)
        features = self.relu_final(features)
        features = self.global_pool(features)
        features = features.squeeze(-1)
        return self.fc(features)


class ECGCNNLSTM(nn.Module):

    def __init__(self, num_leads: int, num_classes: int,
                 cnn_filters: list[int] | None = None,
                 kernel_size: int = CNN_LSTM_KERNEL,
                 lstm_hidden: int = LSTM_HIDDEN,
                 lstm_layers: int = LSTM_LAYERS,
                 lstm_dropout: float = LSTM_DROPOUT,
                 dropout: float = DROPOUT_RATE):
        super().__init__()

        if cnn_filters is None:
            cnn_filters = list(CNN_LSTM_FILTERS)


        feature_layers: list[nn.Module] = []
        input_channels = num_leads
        for filter_count in cnn_filters:
            feature_layers.extend([
                nn.Conv1d(input_channels, filter_count, kernel_size=kernel_size,
                          padding=(kernel_size - 1) // 2, bias=False),
                nn.BatchNorm1d(filter_count),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ])
            input_channels = filter_count
        self.cnn = nn.Sequential(*feature_layers)


        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )


        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * lstm_hidden, num_classes)


        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.cnn(features)
        features = features.permute(0, 2, 1)
        features, _ = self.lstm(features)
        features = features.mean(dim=1)
        features = self.dropout(features)
        return self.fc(features)


def build_model(arch: str, num_leads: int, num_classes: int,
                **kwargs) -> nn.Module:
    if arch == "resnet":
        return ECGResNet(
            num_leads=num_leads,
            num_classes=num_classes,
            base_filters=kwargs.get("base_filters", RESNET_BASE_FILTERS),
            num_blocks=kwargs.get("num_blocks", RESNET_NUM_BLOCKS),
            kernel_size=kwargs.get("kernel_size", RESNET_KERNEL_SIZE),
            dropout=kwargs.get("dropout", DROPOUT_RATE),
            use_se=kwargs.get("use_se", USE_SE_BLOCK),
            se_reduction=kwargs.get("se_reduction", SE_REDUCTION),
        )
    elif arch == "cnn_lstm":
        return ECGCNNLSTM(
            num_leads=num_leads,
            num_classes=num_classes,
            cnn_filters=kwargs.get("cnn_filters", CNN_LSTM_FILTERS),
            kernel_size=kwargs.get("kernel_size", CNN_LSTM_KERNEL),
            lstm_hidden=kwargs.get("lstm_hidden", LSTM_HIDDEN),
            lstm_layers=kwargs.get("lstm_layers", LSTM_LAYERS),
            lstm_dropout=kwargs.get("lstm_dropout", LSTM_DROPOUT),
            dropout=kwargs.get("dropout", DROPOUT_RATE),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch!r}. "
                         f"Choose 'resnet' or 'cnn_lstm'.")
