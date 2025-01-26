import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Определение модели
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3, 32)  # 3 входных пикселя
        self.fc2 = nn.Linear(32, 16)  # Скрытый слой
        self.fc3 = nn.Linear(16, 1)  # Выходной слой

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU
        x = torch.relu(self.fc2(x))  # ReLU
        x = torch.sigmoid(self.fc3(x))  # Сигмоида на выходе
        return x

# Функция для создания набора данных из изображения
def create_dataset(image_array):
    X = []
    y = []
    rows, cols = image_array.shape

    for i in range(1, rows):
        for j in range(1, cols):
            # Собираем три известных пикселя
            left = image_array[i, j-1]  # Левый пиксель
            top = image_array[i-1, j]  # Верхний пиксель
            diag = image_array[i-1, j-1]  # Диагональный пиксель

            # Целевой пиксель
            target = image_array[i, j]

            X.append([left, top, diag])
            y.append(target)

    X = np.array(X, dtype=np.float32).reshape(-1, 3)  # Формат: (количество примеров, 3 пикселя)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)  # Формат: (количество примеров, 1 целевой пиксель)
    return X, y

# Основной скрипт
def main():
    # Путь к папке с изображениями
    samples_dir = os.path.join('.', 'samples')

    # Поиск всех PNG-изображений в папке
    image_paths = [os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f.endswith('.png')]

    if not image_paths:
        print("В папке 'samples' не найдено PNG-изображений.")
        return

    # Создание модели
    model = SimpleModel()

    # Подготовка данных для обучения
    X_all = []
    y_all = []

    for image_path in image_paths:
        # Загрузка изображения и преобразование в grayscale
        image = Image.open(image_path).convert('L')
        image_array = np.array(image) / 255.0  # Нормализация

        # Создание набора данных для текущего изображения
        X, y = create_dataset(image_array)
        X_all.append(X)
        y_all.append(y)

    # Объединение данных из всех изображений
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Преобразование данных в тензоры PyTorch
    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.float32)

    # Компиляция модели
    criterion = nn.MSELoss()  # Функция потерь
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Оптимизатор

    # Обучение модели
    print("Начало обучения...")
    for epoch in range(500):  # 10 эпох
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Эпоха {epoch + 1}, Loss: {loss.item()}")
    print("Обучение завершено.")

    # Экспорт модели в ONNX
    dummy_input = torch.randn(1, 3)  # Пример входных данных (3 пикселя)
    onnx_model_path = "pixel_predictor.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Модель экспортирована в ONNX: {onnx_model_path}")

    # Посттренировочное квантование (PTQ) с использованием ONNX Runtime (Int8)
    quantized_model_path = "pixel_predictor_quantized.onnx"
    quantize_dynamic(
        onnx_model_path,
        quantized_model_path,
        weight_type=QuantType.QInt8  # 8-битное квантование для весов (Int8)
    )
    print(f"Квантованная модель сохранена в формате ONNX: {quantized_model_path}")

if __name__ == "__main__":
    main()