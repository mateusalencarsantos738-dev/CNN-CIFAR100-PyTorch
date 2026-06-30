# O Que Fazer ao Abrir o Notebook no Colab?

Sempre que você fechar o navegador e voltar a abrir o seu projeto (CNN23.ipynb) no Google Colab, a máquina virtual do Google terá sido "zerada". Siga este roteiro dependendo do que você quer fazer:

---

## 🟢 O que você precisa fazer AGORA (Treinar com as Novas Mudanças)

Como você acabou de alterar o `ImageDataGenerator` (adicionando o brilho) e o `ReduceLROnPlateau` (mudando os valores para 0.7 e 8), você precisa garantir que o Colab apague o modelo antigo da memória e rode com as regras novas.

**Siga este passo a passo exato no menu do Colab:**
1. Vá no menu superior e clique em **Ambiente de execução** (ou *Runtime*).
2. Selecione a opção **Reiniciar sessão e executar tudo** (ou *Restart session and run all*).
3. Confirme se aparecer um aviso.

**O que isso vai fazer?**
Ele vai rodar automaticamente, na ordem certa, o download dos dados, o Data Augmentation atualizado, recriar a arquitetura com o `GlobalAveragePooling2D` zerada, aplicar o novo otimizador e começar a treinar sozinho! É só sentar e assistir a barra de progresso.

---

## 🟡 O que fazer no dia a dia (Quando for só testar o modelo pronto)

Se amanhã você abrir o Colab só para ver o modelo ou testar imagens, **você não precisa treinar tudo de novo por horas**.

**Passo a passo:**
1. Faça o **Upload** do arquivo `modelo_v3.keras` que você baixou no dia anterior (arraste para a aba de arquivos na lateral esquerda do Colab).
2. Rode as células **1** (Imports), **2** e **3** (Pré-processamento dos dados).
3. **Pule todas as outras células** (Arquitetura, Augmentation e Fit).
4. Vá para o final do notebook e rode o código de carregamento:
```python
from tensorflow.keras.models import load_model
model = load_model('modelo_v3.keras')
print("Modelo pronto para uso!")
```

---

> **Aviso Crítico Permanente:** O Google Colab apaga TUDO quando você sai. Sempre, sem exceção, quando o seu treinamento (model.fit) acabar e der um resultado bom, **faça o download do arquivo `modelo_v3.keras` para o seu computador**. Se a aba fechar antes de você baixar, você perde as horas de treinamento.
