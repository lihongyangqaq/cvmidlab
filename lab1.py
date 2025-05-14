import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy
import os
import random
import shutil

# --------------------------- 配置 ---------------------------
data_dir = './caltech-101'  # 你的数据文件夹，里面有101_ObjectCategories
batch_size = 32
num_epochs = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --------------------------- 划分数据 ---------------------------
def prepare_data():
    categories_path = os.path.join(data_dir, '101_ObjectCategories')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    for category in os.listdir(categories_path):
        if category == 'BACKGROUND_Google':
            continue  # 跳过 BACKGROUND_Google 类

        category_path = os.path.join(categories_path, category)
        if os.path.isdir(category_path):
            images = [img for img in os.listdir(category_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
            random.shuffle(images)

            train_images = images[:30]
            val_images = images[30:]

            # 训练集
            train_category_path = os.path.join(train_dir, category)
            os.makedirs(train_category_path, exist_ok=True)
            for img in train_images:
                shutil.copy(os.path.join(category_path, img), os.path.join(train_category_path, img))

            # 验证集
            val_category_path = os.path.join(val_dir, category)
            os.makedirs(val_category_path, exist_ok=True)
            for img in val_images:
                shutil.copy(os.path.join(category_path, img), os.path.join(val_category_path, img))

prepare_data()

# --------------------------- 数据预处理 ---------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'))
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# --------------------------- 加载模型 ---------------------------
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet":
        model = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        if feature_extract:
            set_parameter_requires_grad(model, feature_extract)
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        if feature_extract:
            set_parameter_requires_grad(model, feature_extract)
    else:
        raise ValueError("Invalid model name.")
    return model

# --------------------------- 训练与验证（加了每步进度显示） ---------------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for step, (inputs, labels) in enumerate(dataloaders[phase], 1):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 每10步显示一次进度
                if step % 10 == 0 or step == len(dataloaders[phase]):
                    print(f'{phase.capitalize()} Step {step}/{len(dataloaders[phase])}')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

# --------------------------- 主程序 ---------------------------
def run_experiment(model_name, pretrained=True):
    model = initialize_model(model_name, num_classes=101, feature_extract=False, use_pretrained=pretrained)
    model = model.to(device)

    params_to_update = model.parameters()
    optimizer = optim.SGD(params_to_update, lr=0.001 if pretrained else 0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model, train_loss, val_loss, train_acc, val_acc = train_model(model, dataloaders, criterion, optimizer, num_epochs)

    # 保存模型
    torch.save(model.state_dict(), f'{model_name}_pretrained_{pretrained}.pth')

    # 绘制曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title(f'{model_name} Loss Curve (pretrained={pretrained})')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title(f'{model_name} Accuracy Curve (pretrained={pretrained})')
    plt.legend()

    plt.savefig(f'{model_name}_pretrained_{pretrained}_curves.png')
    plt.show()

# --------------------------- 开始实验 ---------------------------
# 1. ResNet18 微调（预训练）
run_experiment("resnet", pretrained=True)

# 2. ResNet18 随机初始化
run_experiment("resnet", pretrained=False)

# 3. AlexNet 微调（预训练）
run_experiment("alexnet", pretrained=True)

# 4. AlexNet 随机初始化
run_experiment("alexnet", pretrained=False)
