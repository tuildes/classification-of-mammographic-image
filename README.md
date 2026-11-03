# Classificador de mamografias

Pipeline de machine learning em Python para classificar massas mamográficas em **benigno** ou **maligno**. O estudo compara seis modelos: SVM e Random Forest treinados sobre características radiômicas, características profundas de uma ResNet-50 e a fusão das duas, no split oficial de treino e teste do CBIS-DDSM.

**Python · PyTorch · scikit-learn · Hugging Face · OpenCV**

No teste (378 imagens), a SVM treinada na fusão chegou a **70,9% de acurácia**. As radiômicas isoladas ficaram em torno de **59%**.

O relatório escrito do trabalho está em [`docs/artigo.pdf`](docs/artigo.pdf).

## Pipeline

```text
CBIS-DDSM (massas, split oficial)
        │
        ▼
OpenCV: filtro bilateral + CLAHE
        │
        ├──────────────────────────────┐
        ▼                              ▼
PyRadiomics                      ResNet-50 (PyTorch)
93 características               2048 características
na máscara da lesão              na ROI recortada
        │                              │
        └──────────┬───────────────────┘
                   ▼
        três representações
        radiômicas · ResNet-50 · fusão
                   │
                   ▼
        StandardScaler + SVM (RBF) e Random Forest
                   │
                   ▼
        teste: acurácia, F1, sensibilidade,
        especificidade e matriz de confusão
```

1. O dataset CBIS-DDSM é baixado com `kagglehub` e os caminhos do CSV oficial de massas são resolvidos para os JPEGs locais.
2. Cada imagem passa por filtro bilateral e CLAHE (OpenCV).
3. PyRadiomics descreve a lesão dentro da máscara. A ResNet-50 pré-treinada, carregada com PyTorch, descreve a ROI recortada. A rede não é retreinada: a cabeça de classificação sai e o embedding de 2048 dimensões entra como feature.
4. SVM e Random Forest (scikit-learn) são treinados em cada representação e na fusão, só com o conjunto de treino.
5. Os seis modelos são avaliados no conjunto de teste.

`BENIGN` e `BENIGN_WITHOUT_CALLBACK` formam a classe 0. `MALIGNANT` é a classe 1, a classe positiva da sensibilidade e do F1.

## O que entra em cada modelo

| Representação | Origem | Características usadas no classificador |
| --- | --- | --- |
| Radiômicas | PyRadiomics na mamografia completa + máscara | 93 |
| ResNet-50 | ROI recortada, extrator pré-treinado | 2048 |
| Fusão | junção das duas pelo identificador da lesão | 2141 |

As 93 radiômicas são as famílias padrão do PyRadiomics: first-order (18), GLCM (24), GLRLM (16), GLSZM (16), GLDM (14) e NGTDM (5). O extrator usa `binWidth` 25, normalização, interpolação B-spline e `force2D`.

A ResNet-50 é a rede pré-treinada no ImageNet exposta pelo PyTorch (`torchvision.models.resnet50`, `ResNet50_Weights.DEFAULT`). A mesma arquitetura está publicada no Hugging Face Hub como [`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50). Neste repositório o modelo é carregado com PyTorch e usado só como extrator de features: `fc` vira `nn.Identity()`. A ROI é convertida para RGB, redimensionada para 240×240 e normalizada com a média e o desvio do ImageNet. O artigo cita 224×224 nessa etapa; o valor executado pelo código está em `src/extract_resnet50.py`.

Os classificadores, ambos com `random_state=42`:

- SVM com kernel RBF, `C=2.0` e `gamma='scale'`
- Random Forest com 100 árvores

As features passam por `StandardScaler` ajustado no treino. O teste só transforma.

## Dados

Fonte: [CBIS-DDSM no Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) (`awsaf49/cbis-ddsm-breast-cancer-image-dataset`).

O pipeline usa o split oficial de **massas** (`mass_case_description_train_set.csv` e `mass_case_description_test_set.csv`). Calcificações ficam de fora.

| Conjunto | Imagens | Benigno (classe 0) | Maligno (classe 1) |
| --- | ---: | ---: | ---: |
| Treino | 1318 | 681 | 637 |
| Teste | 378 | 231 | 147 |

O benigno de treino reúne 577 `BENIGN` e 104 `BENIGN_WITHOUT_CALLBACK`. No teste são 194 e 37.

## Resultados no teste

Métricas do pipeline final (filtro bilateral + CLAHE). Sensibilidade é o recall da classe maligna. Especificidade é o recall da classe benigna.

| Características | Modelo | Acurácia | F1 | Sensibilidade | Especificidade |
| --- | --- | ---: | ---: | ---: | ---: |
| Fusão | SVM | **0,7090** | **0,6259** | 0,6259 | 0,7619 |
| ResNet-50 | SVM | 0,7037 | 0,6242 | 0,6327 | 0,7489 |
| Fusão | Random Forest | 0,6852 | 0,5966 | 0,5986 | 0,7403 |
| ResNet-50 | Random Forest | 0,6852 | 0,5795 | 0,5578 | 0,7662 |
| Radiômicas | SVM | 0,5952 | 0,4632 | 0,4490 | 0,6883 |
| Radiômicas | Random Forest | 0,5873 | 0,4834 | 0,4966 | 0,6450 |

A fusão melhora pouco a ResNet-50 sozinha: a acurácia da SVM sobe de 0,7037 para 0,7090 e a especificidade de 0,7489 para 0,7619, com uma queda pequena de sensibilidade (0,6327 para 0,6259). Radiômicas isoladas ficam perto de 0,59.

### Exemplo: relatório da SVM na fusão

Arquivo [`results/report_merged_SVM.txt`](results/report_merged_SVM.txt):

```text
Results of testing: merged with model: SVM
Acurácia:       0.7090
F1-Score:       0.6259
Sensibilidade:  0.6259
Especificidade: 0.7619

              precision    recall  f1-score   support

      BENIGN       0.76      0.76      0.76       231
   MALIGNANT       0.63      0.63      0.63       147

    accuracy                           0.71       378
```

A matriz correspondente está em [`results/confusion_matrix_merged_SVM.png`](results/confusion_matrix_merged_SVM.png).

![Matriz de confusão da SVM treinada na fusão de radiômicas e ResNet-50](results/confusion_matrix_merged_SVM.png)

Leitura dessa matriz, no conjunto de teste:

|  | Predito benigno | Predito maligno |
| --- | ---: | ---: |
| Benigno | 176 | 55 |
| Maligno | 55 | 92 |

176 + 92 acertos em 378 casos dão a acurácia 0,7090. Os 92 malignos encontrados em 147 dão a sensibilidade 0,6259. Os 176 benignos encontrados em 231 dão a especificidade 0,7619.

Os seis relatórios e as seis matrizes da execução final estão em [`results/`](results/).

## Pré-processamentos comparados

O pré-processamento escolhido para o resultado final é filtro bilateral (`d=9`, `sigmaColor=75`, `sigmaSpace=75`) seguido de CLAHE (`clipLimit=2.0`, grade 8×8), em [`src/preprocessing.py`](src/preprocessing.py).

Outras variantes ficam em [`results/ablation/`](results/ablation/). Abaixo, a acurácia da SVM na fusão, que foi o critério do resultado principal:

| Pré-processamento | Pasta | Acurácia da SVM na fusão |
| --- | --- | ---: |
| Bilateral + CLAHE (final) | `results/` | **0,7090** |
| Equalização de histograma | `results/ablation/histogram_equalization/` | 0,6852 |
| Sem pré-processamento | `results/ablation/no_preprocessing/` | 0,6825 |
| CLAHE | `results/ablation/clahe/` | 0,6799 |
| Mediana | `results/ablation/median/` | 0,6772 |
| Unsharp | `results/ablation/unsharp/` | 0,6667 |
| Bilateral | `results/ablation/bilateral/` | 0,6640 |

Nas pastas de ablação, as métricas da ResNet-50 se repetem (SVM com acurácia 0,6958). A execução final, em `results/`, reextraiu a ResNet-50 com bilateral + CLAHE e chegou a 0,7037.

## Artigo

[`docs/artigo.pdf`](docs/artigo.pdf) — *Classificação de Imagens de Tecido de Mama*, de Aurélio Gabriel Ortiz Vieira e Gustavo Benitez Frehse (UFPR).

O texto descreve este pipeline: preparação do CBIS-DDSM, radiômica com PyRadiomics, ResNet-50 como extratora, SVM, Random Forest e a comparação dos seis modelos no teste. Os números da discussão (SVM na fusão com acurácia 0,7090 e F1 0,6259) são os desta pasta `results/`.

## Como executar

O ambiente precisa de Python 3, credencial da API do Kaggle (`~/.kaggle/kaggle.json`) e espaço para o CBIS-DDSM. A primeira execução baixa o dataset e extrai as features. PyRadiomics no conjunto inteiro é a etapa lenta. Se os CSV de features já existirem em `results/`, a extração é pulada.

```bash
bash scripts/setup.sh
source .venv/bin/activate
python src/main.py
```

`scripts/main.sh` faz o mesmo `python src/main.py` a partir da raiz do repositório. Rodar o script de novo regrava os relatórios e as matrizes em `results/`. Com os CSV atuais e `random_state=42`, as métricas da tabela principal se repetem.

Saídas da execução final:

```text
results/radiomics_features_{train,test}.csv
results/resnet_features_{train,test}.csv
results/merged_features_{train,test}.csv
results/report_{radiomics,resnet50,merged}_{SVM,random_forest}.txt
results/confusion_matrix_{radiomics,resnet50,merged}_{SVM,random_forest}.png
```

## Estrutura

```text
src/dataset.py              download e split oficial de massas do CBIS-DDSM
src/preprocessing.py        filtro bilateral + CLAHE
src/extract_radiomics.py    PyRadiomics
src/extract_resnet50.py     ResNet-50 (PyTorch) como extratora
src/main.py                 fusão, treino e métricas dos seis modelos
scripts/setup.sh            ambiente virtual e dependências
scripts/main.sh             atalho para src/main.py
requirements.txt
results/                    métricas e features da execução final
results/ablation/           variantes de pré-processamento
docs/artigo.pdf             relatório escrito
```

## Limitações

O teste tem 231 benignos e 147 malignos, e o treino tem 1318 imagens. A especificidade fica acima da sensibilidade em todos os seis modelos: o melhor ainda deixa 55 das 147 lesões malignas como benignas. O ganho da fusão sobre a ResNet-50 sozinha é pequeno. Estes números servem para comparar representações e classificadores neste split. O estudo não é um sistema de laudo.
