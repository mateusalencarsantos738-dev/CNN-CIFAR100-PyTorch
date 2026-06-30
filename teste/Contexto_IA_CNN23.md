# RELATÓRIO TÉCNICO E CONTEXTO DE IA: Projeto CNN23 (CIFAR-100)

> **[INSTRUÇÃO PARA QUALQUER IA LENDO ESTE ARQUIVO]**
> Este documento serve como a **Base de Conhecimento Absoluta** sobre o projeto `CNN23` do usuário. Ele documenta o escopo, as regras rigorosas acadêmicas impostas ao projeto, a arquitetura escolhida e o histórico de desempenho. Se você for ajudar o usuário a modificar código, analisar erros ou escrever o trabalho acadêmico final, **siga rigidamente as restrições arquiteturais detalhadas abaixo.**

---

## 1. Escopo e Regras do Projeto (Trabalho Acadêmico)
O objetivo do projeto é classificar a base de dados **CIFAR-100** (60.000 imagens, 100 classes, 32x32 pixels coloridas) construindo uma Rede Neural Convolucional (CNN) do absoluto zero.
*   **A Regra de Ouro:** É **ESTRITAMENTE PROIBIDO** utilizar modelos pré-treinados (ResNet, VGG, etc.) ou pesos pré-treinados (ImageNet). Tudo deve ser aprendido do zero.
*   **Permitido/Obrigatório:** TensorFlow/Keras, Conv2D, MaxPooling2D, Dense, ReLU, Softmax, Dropout, BatchNormalization, Data Augmentation e Callbacks (EarlyStopping, ReduceLROnPlateau).

---

## 2. Arquitetura da Rede (Modelo V3 Consolidado)
O usuário optou pela arquitetura V3 consolidada no arquivo `CNN23.ipynb`. A rede é profunda e utiliza blocos duplos de convolução seguidos por Pooling e Dropout agressivo para conter o overfitting nas 100 classes.

*   **Bloco 1:** 2x Conv2D (64 filtros, 3x3, padding='same') + BatchNorm + ReLU + MaxPooling2D (2x2) + Dropout(0.25)
*   **Bloco 2:** 2x Conv2D (128 filtros, 3x3, padding='same') + BatchNorm + ReLU + MaxPooling2D (2x2) + Dropout(0.25)
*   **Bloco 3:** 2x Conv2D (256 filtros, 3x3, padding='same') + BatchNorm + ReLU + MaxPooling2D (2x2) + Dropout(0.35)
*   **Bloco 4:** 2x Conv2D (512 filtros, 3x3, padding='same') + BatchNorm + ReLU + MaxPooling2D (2x2) + Dropout(0.35)
*   **Classificador Final:**
    *   **GlobalAveragePooling2D:** (Utilizado estrategicamente no lugar do Flatten para reduzir os parâmetros a apenas 512, prevenindo overfitting massivo).
    *   Dense (512) + BatchNorm + ReLU + Dropout(0.50)
    *   Dense (100) + Softmax (Saída)

---

## 3. Estratégias de Treinamento e Hiperparâmetros
*   **Pré-processamento:** Escala de pixels para [0, 1] (`X / 255.0`) e labels em *One-Hot Encoding*. Separação de 5.000 amostras para Validação.
*   **Data Augmentation:** Horizontal Flip, Rotation (15 graus), Zoom (0.1), Width/Height Shift (0.1).
*   **Otimizador e Loss:** `Adam` com *Learning Rate* inicial de `0.001`. Função de perda: `categorical_crossentropy`.
*   **Tamanho do Lote (Batch Size):** `32`. (Atenção IAs: O usuário já testou 64 e o 32 performou muito melhor devido ao ruído estocástico que ajuda na generalização).
*   **Callbacks Ativos:**
    *   `ModelCheckpoint`: Salva apenas o melhor modelo de validação em `modelo_v3.keras`.
    *   `EarlyStopping`: Paciência de 15 épocas monitorando `val_accuracy`.
    *   `ReduceLROnPlateau`: Monitora `val_loss`.

---

## 4. Resultados e Desempenho Alcançado
O modelo V3 provou ser a arquitetura mais sólida do projeto. O histórico de estabilidade aponta para as seguintes métricas (baseado na run campeã registrada):

*   **Acurácia de Treinamento (Train Acc):** ~75% a 76%
*   **Acurácia de Validação (Val Acc):** ~67.6% (Pico oficial registrado antes do Early Stopping).
*   **Diagnóstico:** O modelo atingiu a convergência por volta da época 60-70. Há uma leve presença de *overfitting* (gap de 8% entre treino e validação), o que é perfeitamente normal e saudável para a complexidade do CIFAR-100 sem uso de Transfer Learning. O `GlobalAveragePooling2D` combinado com os pesados `Dropouts` (0.35 e 0.50) seguraram a rede de decorar a base de treino.

---

## 5. Diretrizes Futuras para IAs
1.  **Não sugira `Flatten()`:** Foi testado e o aumento de dimensionalidade e parâmetros prejudicou a generalização da rede. O usuário já estabeleceu o `GlobalAveragePooling2D` como o design padrão.
2.  **Mantenha o escopo:** Se o usuário pedir para melhorar o código, sugira apenas otimizações nos hiperparâmetros (como `factor` ou `patience` do ReduceLROnPlateau) ou técnicas simples de Regularização. Nunca modifique a arquitetura do modelo para incluir quebras de regras da faculdade.
