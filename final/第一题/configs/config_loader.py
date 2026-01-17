import configargparse


def get_config():
    parser = configargparse.ArgumentParser()

    # 基础参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints')

    # 模型参数
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=2)

    # 训练控制
    parser.add_argument('--save_freq', type=int, default=10)

    return parser.parse_args()
