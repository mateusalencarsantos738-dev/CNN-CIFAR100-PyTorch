# Requisitos de Hardware e Consumo de Recursos

**Modelo:** WideResNet-28-10 (CIFAR-100 SOTA)  
**Configuração atual:** Batch Size 256, RandAugment, CutMix, AMP (Mixed Precision)

---

## 1. O que mais consome recursos na máquina e por quê?

O nosso código foi feito para extrair a acurácia máxima, e isso exige bastante do hardware. Aqui estão os três grandes "devoradores" de recursos do nosso notebook:

### A. Processamento Matemático (GPU Cores / FLOPS)
- **O Vilão:** A arquitetura **WideResNet-28-10**.
- **Por quê:** O número "10" na arquitetura significa *Widen Factor = 10*. Isso quer dizer que as camadas convolucionais são 10 vezes mais "largas" que uma ResNet normal, chegando a 640 canais muito cedo na rede. Fazer convoluções de 640x640 canais exige uma quantidade absurda de multiplicações de matrizes simultâneas. É isso que frita a placa de vídeo e faz o uso dela ir a 100%.

### B. Memória de Vídeo (VRAM da GPU)
- **O Vilão:** O `BATCH_SIZE = 256` combinado com o tamanho da rede.
- **Por quê:** Durante o treinamento, a GPU não guarda apenas a imagem. Ela precisa guardar as 256 imagens ao mesmo tempo, **mais** o mapa de características gerado por cada uma das 28 camadas, **mais** os gradientes matemáticos usados para atualizar os pesos na volta (backward pass). Tudo isso fica armazenado na memória da placa de vídeo.

### C. Processador (CPU) e Memória RAM do Sistema
- **O Vilão:** O `RandAugment` e o `DataLoader`.
- **Por quê:** A placa de vídeo não sabe ler imagens do disco rígido nem aplicar filtros nelas. Quem faz o download, recorta, vira, muda a cor e distorce (RandAugment) as 256 imagens por fração de segundo é o processador (CPU). Se a CPU for fraca, a placa de vídeo super potente fica parada esperando a CPU terminar de distorcer as imagens. Chamamos isso de *Gargalo de CPU* (CPU Bottleneck).

---

## 2. Recomendações Mínimas para Rodar o Modelo

Para rodar este código exato (com Batch Size 256 e Mixed Precision ativado), a máquina (seja física ou nuvem) precisa de no mínimo:

### 🖥️ Placa de Vídeo (GPU) - O mais crítico
- **Mínimo:** NVIDIA com **8GB de VRAM** (ex: RTX 2060 Super, RTX 3060 Ti, RTX 4060, GTX 1080).
- **Ideal:** NVIDIA T4 (G4) ou Google Colab Free (geralmente T4 com 16GB).
- **Por quê 8GB?** Graças ao nosso código usar `autocast` (Mixed Precision), o consumo de VRAM cai pela metade. Se rodássemos sem AMP, precisaríamos de no mínimo 12GB ou 16GB de VRAM para segurar o batch size de 256 sem dar erro de falta de memória (OOM).

### 🧠 Processador (CPU)
- **Mínimo:** CPU com 4 Núcleos (Cores) / 8 Threads.
- **Ideal:** 8 Núcleos ou mais.
- **Por quê:** Como o `RandAugment` é extremamente pesado, CPUs muito antigas ou com apenas 2 núcleos vão atrasar o treinamento inteiro. O nosso código usa `num_workers = os.cpu_count()` para tentar usar o processador inteiro da máquina.

### 💾 Memória RAM (Sistema)
- **Mínimo:** 12 GB.
- **Ideal:** 16 GB.
- **Por quê:** O dataset CIFAR-100 inteiro cabe na memória, e usamos `pin_memory=True` no DataLoader. Isso reserva um espaço gigante na RAM do computador para transferir os dados para a placa de vídeo de forma instantânea. Se tiver menos de 12GB, o Colab ou o PC pode travar.

---

## Resumo: Onde eu posso rodar isso?

1. **Google Colab (Gratuito):** Roda perfeitamente! Ele te dá 12GB de RAM, uma CPU decente e uma placa de vídeo T4 (16GB VRAM). Vai levar umas 4 horas para 100 épocas.
2. **Máquinas Locais (PC Gamer):** Qualquer PC razoável montado nos últimos 5 anos com uma placa de vídeo da NVIDIA de 8GB vai treinar isso tranquilamente.
3. **AWS/GCP (Nuvem):** Instâncias **G4dn** (na AWS) são a opção de melhor custo-benefício, entregando exatamente o que é pedido acima.
