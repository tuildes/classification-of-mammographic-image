# image-processing-work
someday i'll change this

---

# Description

O projeto é um pipeline completo de classificação de imagens médicas (raios-X) focado em nódulos mamários. O fluxo de trabalho está dividido nas seguintes etapas:
1. Preparação de Dados: Carregamento das bases de treino e teste do CBIS-DDSM, aplicação de técnicas de pré-processamento e criação de máscaras de segmentação para isolar os nódulos.
2. Extração de Características: Geração de três conjuntos de dados (vetores) para cada imagem:
    - Atributos profundos extraídos por uma CNN (ResNet50).
    - Atributos de forma e textura extraídos via técnica radiômica (pyradiomics), que exige a máscara de segmentação.
    - Fusão (concatenação) das características da ResNet50 e da radiômica.
3. Modelagem: Treinamento de dois algoritmos clássicos (Support Vector Machine e Random Forest) sobre os três conjuntos de características, resultando em 6 cenários de teste distintos.
4. Avaliação de Desempenho: Aplicação dos modelos treinados na base de teste isolada para calcular Acurácia, Sensibilidade, Especificidade e F1-score.
5. Visualização e Resultados: Geração de matrizes de confusão, gráficos e tabelas para documentar o desempenho e identificar o melhor classificador geral.
