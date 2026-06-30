# %% [markdown]
# # GUIA DEFINITIVO E CÓDIGO - CNN 23 (Modelo V3)
# 
# Olá! Este é o seu notebook organizado com as instruções exatas. 
# Leia o texto acima de cada bloco de código para saber se você deve rodá-lo ou pulá-lo.
# 
# ---
# ## 🔴 CENÁRIO A: Treinar do Zero (O que você vai fazer AGORA)
# Se você quer treinar o modelo para aplicar a **variação de brilho** e a **nova taxa de aprendizado**, você não precisa rodar de um em um! 
# Basta ir no menu do Colab em **Ambiente de execução -> Reiniciar sessão e executar tudo**. Ele vai rodar tudo certinho.
# 
# ## 🟡 CENÁRIO B: Apenas Usar o Modelo (Amanhã / Dia a Dia)
# Se você já treinou, já fez o download do arquivo `modelo_v3.keras` hoje, e quer apenas testá-lo amanhã, siga as instruções escritas **acima de cada célula**.

# %% [markdown]
# ---
# ### BLOCO 1: Imports e Bibliotecas
# **Status:** 🟢 **SEMPRE RODE ESTE BLOCO** (Seja para treinar ou para testar depois).

# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import platform
import os

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    GlobalAveragePooling2D
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from sklearn.metrics import confusion_matrix, classification_report

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# %% [markdown]
# ---
# ### BLOCO 2: Carregamento dos Dados (CIFAR-100)
# **Status:** 🟢 **SEMPRE RODE ESTE BLOCO** (Para poder avaliar as imagens depois).

# %%
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print("X_train:", x_train.shape)
print("y_train:", y_train.shape)
print("X_test :", x_test.shape)
print("y_test :", y_test.shape)
print("\nNúmero de classes:", len(np.unique(y_train)))

# %% [markdown]
# ---
# ### BLOCO 3: Pré-processamento e Validação
# **Status:** 🟢 **SEMPRE RODE ESTE BLOCO**.

# %%
# Normalização dos pixels (escala de 0 a 1)
X_train = x_train.astype("float32") / 255.0
X_test = x_test.astype("float32") / 255.0

# One-Hot Encoding para as 100 classes
y_train_cat = to_categorical(y_train, 100)
y_test_cat = to_categorical(y_test, 100)

# Separação de validação (reservando 5000 imagens)
X_val = X_train[:5000]
y_val = y_train_cat[:5000]

X_train_f = X_train[5000:]
y_train_f = y_train_cat[5000:]

print("Treino:", X_train_f.shape, y_train_f.shape)
print("Validação:", X_val.shape, y_val.shape)
print("Teste:", X_test.shape, y_test_cat.shape)

# %% [markdown]
# ---
# ### BLOCO 4: Data Augmentation
# **Status:** 🔴 **PULE ESTE BLOCO** se você estiver apenas carregando o modelo pronto amanhã.
# *(Note que a regra do brilho já foi adicionada aqui!)*

# %%
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2]  # Variação de brilho adicionada!
)

datagen.fit(X_train_f)
print("Data Augmentation configurado com sucesso!")

# %% [markdown]
# ---
# ### BLOCO 5: Arquitetura da Rede (V3)
# **Status:** 🔴 **PULE ESTE BLOCO** se você estiver apenas carregando o modelo pronto amanhã.
# *(O GlobalAveragePooling2D foi mantido conforme você pediu)*

# %%
model = Sequential()

# ===== BLOCO 1 =====
model.add(Conv2D(64, (3,3), padding='same', input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# ===== BLOCO 2 =====
model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# ===== BLOCO 3 =====
model.add(Conv2D(256, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.35))

# ===== BLOCO 4 =====
model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.35))

# ===== CLASSIFICADOR =====
model.add(GlobalAveragePooling2D())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

model.summary()

# %% [markdown]
# ---
# ### BLOCO 6: Compilação e Callbacks (V3)
# **Status:** 🔴 **PULE ESTE BLOCO** se você estiver apenas carregando o modelo pronto amanhã.
# *(Note que os valores 0.7 e 8 já foram atualizados aqui!)*

# %%
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.7,       # Atualizado de 0.5 para 0.7
    patience=8,       # Atualizado de 5 para 8
    min_lr=1e-6,
    verbose=1
)

checkpoint_v3 = ModelCheckpoint(
    'modelo_v3.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("Modelo compilado e callbacks configurados (Específico V3).")

# %% [markdown]
# ---
# ### BLOCO 7: O Treinamento
# **Status:** 🔴 **PULE ESTE BLOCO** se você estiver apenas testando. Só rode se for treinar!

# %%
# A linha steps_per_epoch foi removida para não dar erro no Keras 3!
history_v3 = model.fit(
    datagen.flow(
        X_train_f,
        y_train_f,
        batch_size=32
    ),
    validation_data=(X_val, y_val),
    epochs=150,
    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint_v3
    ],
    verbose=1
)

print("Treinamento finalizado!")

# %% [markdown]
# ---
# ### BLOCO 8: Carregar o Modelo Pronto (USO DIÁRIO)
# **Status:** 🟡 **RODE ESTE BLOCO APENAS** se você abriu o Colab hoje e fez o upload do arquivo `modelo_v3.keras` na barra lateral esquerda. 
# Se você acabou de rodar o bloco 7 (Treinamento), não precisa rodar este.

# %%
from tensorflow.keras.models import load_model

# Descomente as linhas abaixo apagando o '#' quando for usar o modelo pronto
# modelo_pronto = load_model('modelo_v3.keras')
# print("Modelo carregado com sucesso e pronto para previsões!")
