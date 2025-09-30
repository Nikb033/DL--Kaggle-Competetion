# CIFAR-10 Image Classification with Modified ResNet

This project implements a **computationally efficient ResNet-based architecture** for image classification on the **CIFAR-10 dataset**.  
The model is designed to stay under **5 million parameters** while achieving **state-of-the-art accuracy**.  

Final performance:  
- ✅ **94.3% validation accuracy**  
- ✅ **83.5% Kaggle private score**  
- ✅ **Parameter budget: ~4.9M**  

---

## 🚀 Key Highlights
- **Lightweight ResNet variant** tailored for 32×32 CIFAR-10 images.  
- **Parameter-efficient**: only 4.89M trainable params (vs. 11M in ResNet-18).  
- **High performance**: 94.3% test accuracy with just 1.03 GFLOPs.  
- **Optimized training**: label smoothing, weight decay, warmup + linear decay scheduler.  
- **Robust generalization** via advanced augmentation (Cutout, Mixup, CutMix).  

---

## 📂 Repository Structure
cifar10-resnet/
├── model.py # Modified ResNet implementation
├── train.py # Training script
├── utils.py # Helper functions (augmentation, metrics, etc.)
├── requirements.txt # Dependencies
├── DL_Report.pdf # Full project report
└── README.md # Documentation

yaml
Copy code

---

## 🛠️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nikb033/cifar10-resnet.git
   cd cifar10-resnet
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run training:

bash
Copy code
python train.py
⚙️ Model Architecture
Initial Layer: Conv2D (3→36), BN, ReLU

Residual Layers:

Layer 1: 36 channels × 3 blocks

Layer 2: 72 channels × 3 blocks

Layer 3: 144 channels × 3 blocks

Layer 4: 256 channels × 3 blocks

Layer 5: 64 channels × 1 block

Classifier: AdaptiveAvgPool2d(1) → FC(64→10)

Parameter count: 4.89M
Memory footprint: 19.4 MB

🧪 Training Configuration
Optimizer: Adam (lr=0.001, weight decay=1e-4)

Loss: Label-smoothed cross-entropy (α=0.1)

Scheduler: 5-epoch warmup + 75-epoch linear decay

Epochs: 80

Batch size: 128

Framework: PyTorch 2.5.1 + CUDA 12.1

📊 Results
Metric	Score
Training Accuracy	98.56%
Validation Accuracy	94.3%
Kaggle Public Score	83.44%
Kaggle Private Score	83.5%

Trainable Params: 4.89M

GFLOPs: 1.03

Memory Footprint: 19.4 MB

🔍 Example Inference
python
Copy code
import torch
from torchvision import transforms
from PIL import Image
from model import ModifiedResNet

# Load trained model
model = ModifiedResNet(num_classes=10)
model.load_state_dict(torch.load("checkpoint.pth"))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

img = Image.open("sample.png")
x = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(x)
    pred = output.argmax(dim=1).item()

print("Predicted class:", pred)
📖 References
He et al. (2016) – Deep Residual Learning for Image Recognition

Howard et al. (2017) – MobileNets: Efficient Models for Mobile Vision

Tan & Le (2019) – EfficientNet

Krizhevsky (2009) – CIFAR-10 Dataset

👨‍💻 Authors
Nikhil Bhise 

Vishwas Karale
