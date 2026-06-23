import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Tuple


class Client:
    def __init__(self, model: keras.Model, data: Tuple[np.ndarray, np.ndarray], batch_size: int = 32):
        self.model = model
        self.x, self.y = data
        self.batch_size = batch_size

    def train(self, epochs: int, global_weights: List[np.ndarray]) -> Tuple[List[np.ndarray], int]:
        """在本地数据上训练模型，并返回更新后的权重和样本数"""
        self.model.set_weights(global_weights)
        self.model.fit(self.x, self.y, epochs=epochs, batch_size=self.batch_size, verbose=0)
        return self.model.get_weights(), len(self.x)


class Server:
    def __init__(self, model_fn: callable, num_clients: int, clients_data: List[Tuple[np.ndarray, np.ndarray]]):
        """
        初始化联邦学习服务器
        model_fn: 用于创建模型的函数
        """
        self.global_model = model_fn()
        self.clients = [Client(model_fn(), data) for data in clients_data[:num_clients]]
        self.num_clients = num_clients

    def aggregate_weights(self, client_weights: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """聚合客户端权重，使用Federated Averaging算法"""
        total_samples = sum(n for _, n in client_weights)
        # 初始化聚合后的权重为0
        aggregated_weights = [np.zeros_like(w) for w in self.global_model.get_weights()]

        for weights, n in client_weights:
            weight_factor = n / total_samples
            for i in range(len(aggregated_weights)):
                aggregated_weights[i] += weights[i] * weight_factor

        return aggregated_weights

    def train_round(self, selected_clients: List[int], local_epochs: int) -> float:
        """执行一轮联邦学习，包括客户端训练和模型聚合"""
        # 获取当前全局模型权重
        global_weights = self.global_model.get_weights()

        # 选择客户端并训练
        client_updates = []
        for client_id in selected_clients:
            weights, n = self.clients[client_id].train(local_epochs, global_weights)
            client_updates.append((weights, n))

        # 聚合权重
        new_global_weights = self.aggregate_weights(client_updates)
        self.global_model.set_weights(new_global_weights)

        # 评估全局模型（可选）
        # loss, acc = self.global_model.evaluate(test_data, test_labels)
        # return acc
        return 0.0  # 简化版本，实际应用中应返回评估指标


# 示例使用
def create_model():
    """创建一个简单的MNIST分类模型"""
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 模拟数据加载和客户端划分
def load_data_and_create_clients(num_clients: int):
    """加载MNIST数据并模拟划分给多个客户端"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 简单地将数据平均分配给客户端
    client_data = []
    samples_per_client = len(x_train) // num_clients

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        client_data.append((x_train[start_idx:end_idx], y_train[start_idx:end_idx]))

    return client_data, (x_test, y_test)


# 主函数示例
def main():
    num_clients = 10
    clients_data, test_data = load_data_and_create_clients(num_clients)

    # 初始化服务器
    server = Server(create_model, num_clients, clients_data)

    # 训练多个轮次
    num_rounds = 10
    clients_per_round = 3

    for round in range(num_rounds):
        # 随机选择客户端
        selected = np.random.choice(num_clients, clients_per_round, replace=False)
        # 执行一轮训练
        accuracy = server.train_round(selected, local_epochs=2)
        print(f"Round {round + 1}, Selected clients: {selected}, Accuracy: {accuracy:.4f}")

    # 最终评估
    test_loss, test_acc = server.global_model.evaluate(*test_data)
    print(f"Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()