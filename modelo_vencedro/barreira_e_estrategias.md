# Barreira Identificada e Plano de Ataque — CIFAR-100

**Data:** 18 de Junho de 2026  
**Contexto:** Treinamento limitado a 100 épocas máximo (restrição acadêmica)

---

## 1. A Barreira

### O Problema
O primeiro treinamento completo (200 épocas) atingiu **84.52% de Top-1 Accuracy**.  
Porém, com o limite de **100 épocas**, o mesmo modelo atingiu apenas **~74.16%**.

### Por que acontece?
O scheduler atual é o `CosineAnnealingLR` configurado para **T_max = 200 épocas**.  
Isso significa que o Learning Rate foi projetado para descer suavemente de `0.1` até `1e-5` ao longo de 200 épocas.

Ao cortar o treino na época 100, o LR ainda estava em **~0.05** — metade do caminho.  
O modelo foi "desligado" no meio do processo de aprendizado, sem ter completado o ciclo de refinamento.

### Evidência do Log Real

| Época | Val Acc | LR (Learning Rate) | Status |
|---|---|---|---|
| 70 | 72.44% | 0.072 | Ainda explorando |
| 92 | 73.26% | 0.056 | Ainda explorando |
| **99** | **74.16%** | **0.050** | **Corte forçado aqui** |
| 148 | 80.38% | 0.016 | Onde o modelo decolaria |
| 197 | 84.52% | 0.000 | Pico real do modelo |

**Conclusão:** Na época 100, o modelo ainda estava na fase de exploração. O refinamento real só começa após a época 130, quando o LR fica abaixo de 0.03.

---

## 2. Meta do Próximo Experimento

> **Objetivo:** Superar 80% de Val Accuracy dentro de 100 épocas.

---

## 3. Estratégias para Ultrapassar a Barreira

### Estratégia 1 — Trocar o Scheduler: `OneCycleLR` ⭐ (Maior Impacto)

**O que muda:**  
Substituir o `CosineAnnealingLR(T_max=200)` pelo `OneCycleLR`.

**Como funciona o OneCycleLR:**  
- Sobe o LR rapidamente nas primeiras 5–10 épocas (warmup)
- Desce agressivamente até próximo de zero ao longo das 90 épocas restantes
- Completa o ciclo INTEIRO dentro das 100 épocas disponíveis

**Comparação de comportamento do LR:**

| Época | Cosine (200) — Atual | OneCycleLR (100) — Proposto |
|---|---|---|
| 5 | 0.099 | ~0.08 (já descendo) |
| 30 | 0.094 | ~0.05 (refinando) |
| 60 | 0.085 | ~0.01 (afinando) |
| 100 | 0.050 ❌ (cortado) | ~0.000 ✅ (convergido) |

**Ganho esperado:** +4% a +6% → de 74% para ~78–80%

---

### Estratégia 2 — Ajustar o `weight_decay`: `5e-4` → `3e-4`

**O que muda:**  
Reduzir levemente a penalização dos pesos.

**Por que funciona:**  
O `weight_decay` alto (5e-4) foi calibrado para 200 épocas de treinamento. Com 100 épocas, ele "aperta" os pesos antes que a rede tenha aprendido tudo o que precisa, freando o aprendizado justamente no momento em que o modelo mais precisa de liberdade.

**Ganho esperado:** +0.5% a +1%

---

### Estratégia 3 — Warmup Linear de 5 Épocas

**O que muda:**  
Nas primeiras 5 épocas, o LR sobe linearmente de `0.01` até `0.1` antes de começar a descer.

**Por que funciona:**  
No início do treino, a rede parte de pesos aleatórios. Sair com LR = 0.1 direto pode causar instabilidade nos gradientes nas primeiras épocas, desperdiçando um aprendizado valioso no nosso orçamento curto de 100 épocas. O warmup estabiliza a rede antes de acelerar.

**Ganho esperado:** +0.5% a +1% (quando combinado com OneCycleLR, já está embutido)

---

## 4. Estimativa de Resultado com as Estratégias

| Configuração | Acurácia Esperada em 100 Épocas |
|---|---|
| Atual (Cosine 200 cortado) | ~74% |
| + OneCycleLR | ~78–80% |
| + OneCycleLR + weight_decay 3e-4 | **~80–82%** |

---

## 5. O que NÃO mudar

Para que o experimento seja científico e isolado, o restante da configuração permanece **idêntico** ao primeiro treinamento:

- ✅ Arquitetura: WideResNet-28-10
- ✅ CutMix (probabilidade 0.5)
- ✅ RandAugment (num_ops=2, magnitude=9)
- ✅ Label Smoothing (0.1)
- ✅ Batch Size (256)
- ✅ AMP (Mixed Precision)
- ✅ Seed Global (42)
