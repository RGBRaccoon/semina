import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb

# 1. LOGIN (로그인)
# 터미널에서: wandb login
# 또는 코드에서:
# import os
# os.environ["WANDB_API_KEY"] = "your_api_key_here"

# 2. INIT (초기화)
wandb.init(
    project="CNN_SEMINA",
    name="FCNN",
    config={
        "model_type": "FCNN",
        "input_size": 64 * 64 * 3,
        "hidden_size": 128,
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 16
    }
)

# 3. LOG (로깅) - 다양한 예제들

# 3.1 기본 메트릭 로깅
for epoch in range(5):
    # 가상의 학습 결과
    train_loss = 1.0 - epoch * 0.15 + np.random.normal(0, 0.05)
    val_accuracy = 50 + epoch * 8 + np.random.normal(0, 2)
    
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": 0.001
    })
    
    print(f"Epoch {epoch}: Loss={train_loss:.3f}, Acc={val_accuracy:.1f}%")

# 3.2 이미지 로깅
# 가상의 이미지 데이터 생성
fake_image = np.random.rand(64, 64, 3)
wandb.log({"sample_image": wandb.Image(fake_image)})

# 3.3 모델 정보 로깅
class SimpleFCNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 평탄화
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleFCNN(64*64*3, 128)
total_params = sum(p.numel() for p in model.parameters())

wandb.log({
    "total_parameters": total_params,
    "model_size_mb": total_params * 4 / (1024 * 1024)
})

# 3.4 예측 결과 테이블 로깅
predictions = ["Cat", "Dog", "Cat", "Dog", "Cat"]
targets = ["Cat", "Dog", "Dog", "Dog", "Cat"]
correct = [True, True, False, True, True]

wandb.log({
    "predictions": wandb.Table(
        columns=["prediction", "target", "correct"],
        data=list(zip(predictions, targets, correct))
    )
})

# 3.5 그래프 로깅
fig, ax = plt.subplots()
x = np.linspace(0, 4, 100)
y = np.sin(x)
ax.plot(x, y)
ax.set_title("Sample Plot")
wandb.log({"sample_plot": wandb.Image(fig)})

# 3.6 설정값 로깅
wandb.log({
    "hyperparameters": wandb.Table(
        columns=["parameter", "value"],
        data=[
            ["learning_rate", 0.001],
            ["batch_size", 16],
            ["epochs", 5],
            ["model_type", "FCNN"]
        ]
    )
})

print("WandB 로깅 완료!")
print("wandb.ai에서 실험 결과를 확인하세요.")

# 실험 종료
wandb.finish() 