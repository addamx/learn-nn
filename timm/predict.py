import datasets
import torch
from torchvision import transforms
from PIL import Image
import timm
from datasets import load_dataset

proxy = {
    "http": "http://127.0.0.1:10796",
    "https": "http://127.0.0.1:10796",
}
config = datasets.DownloadConfig(proxies=proxy)
# 加载数据集
dataset = load_dataset("huggingface/cats-image", download_config=config)
image = dataset["test"]["image"][0]

# 定义预处理操作
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 预处理图像
image = preprocess(image).unsqueeze(0)  # 添加批次维度

# 加载预训练的EfficientNet模型
model = timm.create_model('efficientnet_b0', pretrained=True)
model.eval()

# 使用模型进行预测
with torch.no_grad():
    logits = model(image)

# 获取预测标签
predicted_label = logits.argmax(-1).item()

# EfficientNet使用的是ImageNet的1000个类别
# 从timm的模型配置中获取类别名称
class_map = model.default_cfg['label']
predicted_class = class_map[predicted_label]
print(predicted_class)