# Guia de Alterações — Como Superar 80% em 100 Épocas

Baseado em: `barreira_e_estrategias.md`

Abra o Colab e procure o título: **"FASE 11 e 12: Optimizer e Scheduler"**

---

# CÓDIGO ATUAL (o que está no notebook hoje)

```
-----------------------------------------------------------------------
EPOCHS = 200
LEARNING_RATE = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
-----------------------------------------------------------------------
```

---

# ESTRATÉGIA 1 — OneCycleLR (Maior Impacto) | Ganho: ~78% a 80%

Mexe em 2 lugares.

## PARTE A — FASE 11 e 12

Apague o código atual e cole este:

```
-----------------------------------------------------------------------
EPOCHS = 100
LEARNING_RATE = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4, nesterov=True)

steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1e4
)
-----------------------------------------------------------------------
```

## PARTE B — FASE 13 e 14 (loop de treino)

Encontre esta linha (está depois do bloco de validação):

```
-----------------------------------------------------------------------
    scheduler.step()   ← REMOVA DAQUI
-----------------------------------------------------------------------
```

E cole ela aqui (dentro do loop de batches, logo após scaler.update):

```
-----------------------------------------------------------------------
        scaler.update()
        scheduler.step()  ← COLE AQUI
-----------------------------------------------------------------------
```

---

# ESTRATÉGIA 2 — Reduzir weight_decay | Ganho adicional: ~+0.5% a +1%

Mexe só em 1 lugar: **FASE 11 e 12**

Na linha do optimizer, troque `weight_decay=5e-4` por `weight_decay=3e-4`:

```
-----------------------------------------------------------------------
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=3e-4, nesterov=True)
-----------------------------------------------------------------------
```

Esta estratégia é um complemento da Estratégia 1, não substituta.

---

# ESTRATÉGIA 3 — Warmup Linear | Ganho adicional: ~+0.5% a +1%

**Nota:** Quando se usa o OneCycleLR (Estratégia 1), o warmup já está embutido automaticamente via `pct_start=0.1` (10% das épocas = 10 épocas de warmup). Não é necessário fazer nada adicional.

Esta estratégia só seria relevante se você usar CosineAnnealing em vez do OneCycleLR.

---

# RESUMO — Ordem de Implementação Recomendada

| Passo | Estratégia | Fases alteradas | Ganho esperado |
|---|---|---|---|
| 1º | OneCycleLR | FASE 11 e 12 + FASE 13 e 14 | ~78–80% |
| 2º | weight_decay 3e-4 (junto com passo 1) | FASE 11 e 12 | +~1% extra |
| 3º | Warmup (já incluído no OneCycleLR) | Nenhuma | Automático |
| **Total** | **Combinação dos 3** | — | **~80–82%** |
