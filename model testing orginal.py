import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import random
from tqdm import tqdm
import copy


# 设置随机种子确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed()


# 数据加载与预处理
def create_data_loaders(data_dir, batch_size=64):
    # 减少数据增强操作
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        validation_split=0.1
    )

    # 验证集和测试集使用较小的预处理
    test_datagen = ImageDataGenerator(rescale=1./255)

    # 加载训练集
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # 加载验证集
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # 加载测试集
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    class_names = list(train_generator.class_indices.keys())

    return train_generator, validation_generator, test_generator, class_names


# 模型构建
def create_model(num_classes):
    # 使用预训练的ResNet50作为基础模型
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True

    # 冻结部分层
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),  # 添加Dropout减少过拟合
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# 训练函数
def train_model(model, train_generator, validation_generator, num_epochs=50):
    # 定义损失函数和优化器
    criterion = losses.CategoricalCrossentropy()
    optimizer = optimizers.Adam(learning_rate=0.0001, weight_decay=1e-5)

    # 学习率调度器
    scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    model.compile(optimizer=optimizer, loss=criterion, metrics=['accuracy'])

    best_acc = 0.0
    best_epoch = 0
    best_model_weights = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        history = model.fit(
            train_generator,
            epochs=1,
            validation_data=validation_generator,
            callbacks=[scheduler],
            verbose=0
        )

        train_loss = history.history['loss'][0]
        train_acc = history.history['accuracy'][0]
        val_loss = history.history['val_loss'][0]
        val_acc = history.history['val_accuracy'][0]

        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_model_weights = model.get_weights()
            print(f'Best model saved at epoch {best_epoch} with acc: {best_acc:.4f}')

        print()

    print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')

    # 加载最佳模型权重
    model.set_weights(best_model_weights)
    return model


# 测试函数
def test_model(model, test_generator):
    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    print(f'Test Acc: {test_acc:.4f}')

    return test_acc


# 主函数
def main():
    # 数据目录
    data_dir = r"C:\Users\15236\PycharmProjects\PythonProject2\cifar10_clients_data\before_erased"  # 替换为你的数据目录

    # 创建数据加载器
    print("Loading data...")
    train_generator, validation_generator, test_generator, class_names = create_data_loaders(data_dir, batch_size=32)
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_generator.filenames) * 0.9}")
    print(f"Validation samples: {len(train_generator.filenames) * 0.1}")
    print(f"Test samples: {len(test_generator.filenames)}")

    # 创建模型
    print("Creating model...")
    model = create_model(len(class_names))

    # 训练模型
    print("Training model...")
    best_model = train_model(model, train_generator, validation_generator, num_epochs=5)

    # 测试模型
    print("Testing model...")
    test_acc = test_model(best_model, test_generator)

    # 保存模型
    model_save_path = "best_model.h5"
    best_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # 保存类别信息
    class_info_path = "class_names.txt"
    with open(class_info_path, 'w') as f:
        f.write('\n'.join(class_names))
    print(f"Class names saved to {class_info_path}")


if __name__ == "__main__":
    main()