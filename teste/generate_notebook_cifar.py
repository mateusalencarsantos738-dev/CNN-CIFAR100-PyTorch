import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Treinamento Profissional PyTorch - CIFAR-100\n",
                "## Otimizado para GPUs H100 / A100\n",
                "\n",
                "**Objetivo**: Atingir a maior acurácia possível no CIFAR-100 treinando do zero, sem pesos pré-treinados, utilizando WideResNet-28-10, CutMix, RandAugment, SGD+Momentum, Cosine Annealing, Label Smoothing e Mixed Precision (AMP)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### FASE 1: Preparação do Ambiente"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import platform\n",
                "\n",
                "print(f\"Python version: {sys.version}\")\n",
                "print(f\"OS: {platform.platform()}\")\n",
                "print(f\"CPU Cores: {os.cpu_count()}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### FASE 2: Instalação das dependências\n",
                "Instalando `torchmetrics` para métricas mais confiáveis e matriz de confusão elegante."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q torchmetrics seaborn pandas matplotlib"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### FASE 3: Configuração da GPU\n",
                "Verificando a presença de A100 ou H100."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f\"Using device: {device}\")\n",
                "\n",
                "if device.type == 'cuda':\n",
                "    print(f\"GPU Name: {torch.cuda.get_device_name(0)}\")\n",
                "    print(f\"CUDA Capability: {torch.cuda.get_device_capability(0)}\")\n",
                "    print(f\"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")\n",
                "    \n",
                "    # Habilitar o modo de alta performance (TF32) nas arquiteturas Ampere/Hopper\n",
                "    torch.backends.cuda.matmul.allow_tf32 = True\n",
                "    torch.backends.cudnn.allow_tf32 = True\n",
                "    torch.backends.cudnn.benchmark = True # Otimiza convs para tamanhos estáticos"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### FASE 4: Seed Global\n",
                "Garantindo reprodutibilidade (embora `cudnn.benchmark` introduza um leve não-determinismo em prol da velocidade, compensa no nosso cenário)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import random\n",
                "import numpy as np\n",
                "\n",
                "def set_seed(seed=42):\n",
                "    random.seed(seed)\n",
                "    np.random.seed(seed)\n",
                "    torch.manual_seed(seed)\n",
                "    torch.cuda.manual_seed_all(seed)\n",
                "    # torch.backends.cudnn.deterministic = True  # Desativado para não perder performance extrema\n",
                "    \n",
                "set_seed(42)\n",
                "print(\"Global seed set to 42.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### FASE 5, 6 e 7: Transforms e Download do CIFAR-100\n",
                "O uso de RandAugment é a técnica SOTA para augmentation automático sem precisar buscar hiperparâmetros na mão."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torchvision\n",
                "import torchvision.transforms as transforms\n",
                "\n",
                "# Normalização padrão para CIFAR-100\n",
                "mean = (0.5071, 0.4867, 0.4408)\n",
                "std = (0.2675, 0.2565, 0.2761)\n",
                "\n",
                "# Transforms de Treino (RandAugment SOTA)\n",
                "transform_train = transforms.Compose([\n",
                "    transforms.RandomCrop(32, padding=4),\n",
                "    transforms.RandomHorizontalFlip(),\n",
                "    transforms.RandAugment(num_ops=2, magnitude=9), # RandAugment para grande generalização\n",
                "    transforms.ToTensor(),\n",
                "    transforms.Normalize(mean, std),\n",
                "    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0) # Opcional: Cutout em cima do RandAugment\n",
                "])\n",
                "\n",
                "# Transforms de Validação\n",
                "transform_val = transforms.Compose([\n",
                "    transforms.ToTensor(),\n",
                "    transforms.Normalize(mean, std),\n",
                "])\n",
                "\n",
                "train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)\n",
                "test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)\n",
                "\n",
                "print(f\"Train size: {len(train_dataset)}\")\n",
                "print(f\"Test size: {len(test_dataset)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### FASE 8: DataLoaders Otimizados\n",
                "Batch size de 256 para manter a estocasticidade do SGD forte. H100 suportaria 8192, mas degrada a acurácia. Pin memory para IO rápido."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "BATCH_SIZE = 256\n",
                "NUM_WORKERS = os.cpu_count() or 4\n",
                "\n",
                "train_loader = torch.utils.data.DataLoader(\n",
                "    train_dataset, batch_size=BATCH_SIZE, shuffle=True, \n",
                "    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True\n",
                ")\n",
                "\n",
                "test_loader = torch.utils.data.DataLoader(\n",
                "    test_dataset, batch_size=BATCH_SIZE, shuffle=False, \n",
                "    num_workers=NUM_WORKERS, pin_memory=True\n",
                ")\n",
                "\n",
                "print(f\"DataLoaders otimizados criados com Batch Size = {BATCH_SIZE} e Workers = {NUM_WORKERS}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Implementação CutMix (Parte vital para atingir SOTA)\n",
                "Esta função é aplicada aos batches direto no loop de treino."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def rand_bbox(size, lam):\n",
                "    W = size[2]\n",
                "    H = size[3]\n",
                "    cut_rat = np.sqrt(1. - lam)\n",
                "    cut_w = int(W * cut_rat)\n",
                "    cut_h = int(H * cut_rat)\n",
                "\n",
                "    # uniform\n",
                "    cx = np.random.randint(W)\n",
                "    cy = np.random.randint(H)\n",
                "\n",
                "    bbx1 = np.clip(cx - cut_w // 2, 0, W)\n",
                "    bby1 = np.clip(cy - cut_h // 2, 0, H)\n",
                "    bbx2 = np.clip(cx + cut_w // 2, 0, W)\n",
                "    bby2 = np.clip(cy + cut_h // 2, 0, H)\n",
                "\n",
                "    return bbx1, bby1, bbx2, bby2"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### FASE 9: Definição da Arquitetura (WideResNet-28-10)\n",
                "ResNet-50 nativa no PyTorch tem kernel 7x7 e stride 2 inicial, focada no ImageNet. Reduz de cara 32x32 para 16x16, e depois MaxPool para 8x8. Isso estraga o CIFAR.\n",
                "Para CIFAR-100, a **WideResNet-28-10** é padrão-ouro para altíssima acurácia do zero."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch.nn as nn\n",
                "import torch.nn.functional as F\n",
                "\n",
                "class BasicBlock(nn.Module):\n",
                "    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):\n",
                "        super(BasicBlock, self).__init__()\n",
                "        self.bn1 = nn.BatchNorm2d(in_planes)\n",
                "        self.relu1 = nn.ReLU(inplace=True)\n",
                "        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,\n",
                "                               padding=1, bias=False)\n",
                "        self.bn2 = nn.BatchNorm2d(out_planes)\n",
                "        self.relu2 = nn.ReLU(inplace=True)\n",
                "        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,\n",
                "                               padding=1, bias=False)\n",
                "        self.droprate = dropRate\n",
                "        self.equalInOut = (in_planes == out_planes)\n",
                "        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,\n",
                "                               padding=0, bias=False) or None\n",
                "\n",
                "    def forward(self, x):\n",
                "        if not self.equalInOut:\n",
                "            x = self.relu1(self.bn1(x))\n",
                "        else:\n",
                "            out = self.relu1(self.bn1(x))\n",
                "        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))\n",
                "        if self.droprate > 0:\n",
                "            out = F.dropout(out, p=self.droprate, training=self.training)\n",
                "        out = self.conv2(out)\n",
                "        return torch.add(x if self.equalInOut else self.convShortcut(x), out)\n",
                "\n",
                "class NetworkBlock(nn.Module):\n",
                "    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):\n",
                "        super(NetworkBlock, self).__init__()\n",
                "        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)\n",
                "    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):\n",
                "        layers = []\n",
                "        for i in range(nb_layers):\n",
                "            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))\n",
                "        return nn.Sequential(*layers)\n",
                "    def forward(self, x):\n",
                "        return self.layer(x)\n",
                "\n",
                "class WideResNet(nn.Module):\n",
                "    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):\n",
                "        super(WideResNet, self).__init__()\n",
                "        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]\n",
                "        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'\n",
                "        n = (depth - 4) // 6\n",
                "        block = BasicBlock\n",
                "        # 1st conv before any network block\n",
                "        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,\n",
                "                               padding=1, bias=False)\n",
                "        # 1st block\n",
                "        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)\n",
                "        # 2nd block\n",
                "        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)\n",
                "        # 3rd block\n",
                "        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)\n",
                "        # global average pooling and classifier\n",
                "        self.bn1 = nn.BatchNorm2d(nChannels[3])\n",
                "        self.relu = nn.ReLU(inplace=True)\n",
                "        self.fc = nn.Linear(nChannels[3], num_classes)\n",
                "        self.nChannels = nChannels[3]\n",
                "\n",
                "        for m in self.modules():\n",
                "            if isinstance(m, nn.Conv2d):\n",
                "                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')\n",
                "            elif isinstance(m, nn.BatchNorm2d):\n",
                "                m.weight.data.fill_(1)\n",
                "                m.bias.data.zero_()\n",
                "            elif isinstance(m, nn.Linear):\n",
                "                m.bias.data.zero_()\n",
                "                \n",
                "    def forward(self, x):\n",
                "        out = self.conv1(x)\n",
                "        out = self.block1(out)\n",
                "        out = self.block2(out)\n",
                "        out = self.block3(out)\n",
                "        out = self.relu(self.bn1(out))\n",
                "        out = F.avg_pool2d(out, 8)\n",
                "        out = out.view(-1, self.nChannels)\n",
                "        return self.fc(out)\n",
                "\n",
                "model = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)\n",
                "model = model.to(device)\n",
                "\n",
                "# Total de parâmetros deve ficar em torno de 36.5M\n",
                "print(f\"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### FASE 10: Loss Function (CrossEntropy com Label Smoothing)\n",
                "O Label smoothing de 0.1 evita 'overconfidence', dando margem para as predições. Combinado com CutMix, é SOTA."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "criterion = nn.CrossEntropyLoss(label_smoothing=0.1)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### FASE 11 e 12: Optimizer (SGD + Momentum) e Scheduler (Cosine Annealing)\n",
                "Para CIFAR e imagens, `SGD` puro com `momentum` e alto `weight_decay` generaliza notoriamente melhor do que `AdamW` que tem tendência de overfittar cedo. `CosineAnnealingLR` garante uma descida harmoniosa para zero."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "EPOCHS = 200 # Tempo ideal para convergência total em SGD com Cosine Annealing\n",
                "LEARNING_RATE = 0.1\n",
                "\n",
                "optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4, nesterov=True)\n",
                "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### FASE 13 e 14: Mixed Precision (AMP) e Loop de Treinamento Modularizado\n",
                "O Treinamento inclui AMP via `GradScaler`, CutMix e salvamento do melhor modelo baseado na acurácia."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import time\n",
                "from torch.amp import autocast, GradScaler\n",
                "from torchmetrics import Accuracy\n",
                "\n",
                "scaler = GradScaler('cuda')\n",
                "accuracy_metric = Accuracy(task=\"multiclass\", num_classes=100).to(device)\n",
                "top5_accuracy_metric = Accuracy(task=\"multiclass\", num_classes=100, top_k=5).to(device)\n",
                "\n",
                "best_acc = 0.0\n",
                "cutmix_prob = 0.5\n",
                "\n",
                "print(\"Iniciando Treinamento...\")\n",
                "for epoch in range(EPOCHS):\n",
                "    model.train()\n",
                "    train_loss = 0\n",
                "    correct = 0\n",
                "    total = 0\n",
                "    \n",
                "    start_time = time.time()\n",
                "    for batch_idx, (inputs, targets) in enumerate(train_loader):\n",
                "        inputs, targets = inputs.to(device), targets.to(device)\n",
                "        \n",
                "        # CutMix implementation\n",
                "        r = np.random.rand(1)\n",
                "        if r < cutmix_prob:\n",
                "            # generate mixed sample\n",
                "            lam = np.random.beta(1.0, 1.0)\n",
                "            rand_index = torch.randperm(inputs.size()[0]).cuda()\n",
                "            target_a = targets\n",
                "            target_b = targets[rand_index]\n",
                "            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)\n",
                "            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]\n",
                "            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))\n",
                "            \n",
                "            optimizer.zero_grad()\n",
                "            with autocast('cuda'):\n",
                "                outputs = model(inputs)\n",
                "                loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)\n",
                "        else:\n",
                "            optimizer.zero_grad()\n",
                "            with autocast('cuda'):\n",
                "                outputs = model(inputs)\n",
                "                loss = criterion(outputs, targets)\n",
                "                \n",
                "        scaler.scale(loss).backward()\n",
                "        scaler.step(optimizer)\n",
                "        scaler.update()\n",
                "        \n",
                "        train_loss += loss.item()\n",
                "        _, predicted = outputs.max(1)\n",
                "        total += targets.size(0)\n",
                "        correct += predicted.eq(targets).sum().item()\n",
                "        \n",
                "    # --- Validação ---\n",
                "    model.eval()\n",
                "    val_loss = 0\n",
                "    accuracy_metric.reset()\n",
                "    top5_accuracy_metric.reset()\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        for inputs, targets in test_loader:\n",
                "            inputs, targets = inputs.to(device), targets.to(device)\n",
                "            with autocast('cuda'):\n",
                "                outputs = model(inputs)\n",
                "                loss = criterion(outputs, targets)\n",
                "            \n",
                "            val_loss += loss.item()\n",
                "            accuracy_metric.update(outputs, targets)\n",
                "            top5_accuracy_metric.update(outputs, targets)\n",
                "            \n",
                "    val_acc = accuracy_metric.compute().item() * 100\n",
                "    val_top5 = top5_accuracy_metric.compute().item() * 100\n",
                "    scheduler.step()\n",
                "    \n",
                "    epoch_time = time.time() - start_time\n",
                "    \n",
                "    print(f'Epoch [{epoch+1}/{EPOCHS}] | Time: {epoch_time:.0f}s | '\n",
                "          f'Loss: {train_loss/len(train_loader):.4f} | '\n",
                "          f'Val Loss: {val_loss/len(test_loader):.4f} | '\n",
                "          f'Val Acc: {val_acc:.2f}% | Val Top5: {val_top5:.2f}% | '\n",
                "          f'LR: {scheduler.get_last_lr()[0]:.5f}')\n",
                "    \n",
                "    if val_acc > best_acc:\n",
                "        best_acc = val_acc\n",
                "        torch.save(model.state_dict(), 'best_wrn28_10_cifar100.pth')\n",
                "        print(f'>>> Melhor modelo salvo! Acurácia: {best_acc:.2f}%')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### FASE 15: Avaliação Final (Confusion Matrix)\n",
                "Usamos `torchmetrics` e `seaborn` para exibir os resultados ricos."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import seaborn as sns\n",
                "import matplotlib.pyplot as plt\n",
                "from torchmetrics import ConfusionMatrix\n",
                "\n",
                "# Carregar o melhor modelo\n",
                "model.load_state_dict(torch.load('best_wrn28_10_cifar100.pth'))\n",
                "model.eval()\n",
                "\n",
                "confmat = ConfusionMatrix(task=\"multiclass\", num_classes=100).to(device)\n",
                "accuracy_metric.reset()\n",
                "top5_accuracy_metric.reset()\n",
                "\n",
                "with torch.no_grad():\n",
                "    for inputs, targets in test_loader:\n",
                "        inputs, targets = inputs.to(device), targets.to(device)\n",
                "        outputs = model(inputs)\n",
                "        confmat.update(outputs, targets)\n",
                "        accuracy_metric.update(outputs, targets)\n",
                "        top5_accuracy_metric.update(outputs, targets)\n",
                "\n",
                "cm = confmat.compute().cpu().numpy()\n",
                "final_acc = accuracy_metric.compute().item() * 100\n",
                "final_top5 = top5_accuracy_metric.compute().item() * 100\n",
                "\n",
                "print(f\"\\n--- RESULTADOS FINAIS ---\")\n",
                "print(f\"Melhor Top-1 Accuracy: {final_acc:.2f}%\")\n",
                "print(f\"Melhor Top-5 Accuracy: {final_top5:.2f}%\")\n",
                "\n",
                "plt.figure(figsize=(20, 20))\n",
                "sns.heatmap(cm, cmap='Blues', annot=False, xticklabels=False, yticklabels=False)\n",
                "plt.title('Confusion Matrix - CIFAR-100', fontsize=16)\n",
                "plt.ylabel('True label')\n",
                "plt.xlabel('Predicted label')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### FASE 16: Inferência em Novas Imagens"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import urllib.request\n",
                "from PIL import Image\n",
                "\n",
                "classes = test_dataset.classes\n",
                "\n",
                "def predict_image(img_url, model, transform):\n",
                "    urllib.request.urlretrieve(img_url, \"sample.jpg\")\n",
                "    img = Image.open(\"sample.jpg\").convert('RGB')\n",
                "    img_t = transform(img).unsqueeze(0).to(device)\n",
                "    \n",
                "    model.eval()\n",
                "    with torch.no_grad():\n",
                "        outputs = model(img_t)\n",
                "        probs = torch.nn.functional.softmax(outputs, dim=1)\n",
                "        top_prob, top_class = probs.topk(1, dim=1)\n",
                "        \n",
                "    pred_class_name = classes[top_class.item()]\n",
                "    print(f\"Predição: {pred_class_name} | Confiança: {top_prob.item()*100:.2f}%\")\n",
                "    \n",
                "    plt.imshow(img)\n",
                "    plt.title(f\"{pred_class_name} ({top_prob.item()*100:.2f}%)\")\n",
                "    plt.axis('off')\n",
                "    plt.show()\n",
                "\n",
                "# Exemplo com a imagem de um trator, que é uma classe do CIFAR100 (tractor)\n",
                "sample_url = \"https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Red_Tractor.jpg/320px-Red_Tractor.jpg\"\n",
                "\n",
                "# Notar que usamos um resize primeiro porque imagens aleatórias da web não são 32x32 nativamente\n",
                "inference_transform = transforms.Compose([\n",
                "    transforms.Resize((32, 32)),\n",
                "    transforms.ToTensor(),\n",
                "    transforms.Normalize(mean, std),\n",
                "])\n",
                "\n",
                "predict_image(sample_url, model, inference_transform)"
            ]
        }
    ]
}

# Salvar como JSON/IPYNB
with open('/home/teus/CNN/CIFAR100_SOTA_Colab.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook gerado com sucesso!")
