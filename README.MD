# Detecção de Fake News com Machine Learning e Transformers em Português

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=yellow)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--learn%20%7C%20PyTorch%20%7C%20Transformers-orange)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

Este repositório contém o código-fonte e os modelos desenvolvidos para o Trabalho de Conclusão de Curso (TCC) em Engenharia de Software, focado na detecção de notícias falsas (`fake news`) em português do Brasil.

## 📝 Sobre o Projeto

O objetivo deste projeto foi desenvolver e realizar uma análise comparativa da eficácia de diferentes algoritmos de aprendizado de máquina na tarefa de classificar textos noticiosos como **verdadeiros** ou **falsos**. Para isso, foram implementadas e avaliadas quatro abordagens distintas:

1.  **Naive Bayes:** Um modelo probabilístico clássico, utilizado como linha de base (*baseline*).
2.  **Random Forest:** Um modelo *ensemble* baseado em árvores de decisão.
3.  **Support Vector Machine (SVM):** Um classificador de margem máxima, robusto para espaços de alta dimensionalidade.
4.  **BERT (Bidirectional Encoder Representations from Transformers):** Um modelo de linguagem de última geração, utilizando a arquitetura pré-treinada `BERTimbau` para português.

O dataset utilizado foi o **Fake.br-Corpus**, um corpus de notícias em português brasileiro, garantindo a relevância contextual dos resultados para a realidade nacional.

## 📊 Resultados Finais

Após treinamento e otimização, os modelos foram avaliados em um conjunto de teste, com o **SVM** apresentando o melhor desempenho geral:

| Modelo                  | Acurácia | Precisão (falsa) | Recall (falsa) | F1-Score (falsa) |
| :---------------------- | :------- | :--------------- | :------------- | :--------------- |
| **SVM Otimizado**           | **97.0%**    | **96.9%**            | **97.1%**          | **97.0%**            |
| Random Forest Otimizado | 96.1%    | 95.7%            | 96.5%          | 96.1%            |
| BERT                    | 93.3%    | 90.6%            | 96.7%          | 93.6%            |
| Naive Bayes Otimizado   | 88.7%    | 90.8%            | 86.2%          | 88.5%            |

## 📂 Estrutura do Repositório

O projeto está organizado da seguinte forma para garantir clareza e reprodutibilidade:

```
.
├── notebooks/
│   └── tcc_treinamento_e_analise.ipynb   # Notebook com toda a pesquisa e treinamento.
├── scripts/
│   └── verificar_noticia.py            # Interface CLI para previsões rápidas.
├── modelos_salvos/
│   ├── modelo_svm.pkl                    # Modelos clássicos e vetorizador salvos.
│   ├── modelo_nb.pkl                     
│   ├── modelo_rf.pkl  
│   ├── vetorizador_tfidf.pkl
│   └── modelo_bert.bin                   # Modelo BERT (não foi possível incluir no repositório por causa de seu tamanho, mas pode ser gerado em notebooks/tcc_treinamento_e_analise.ipynb).
├── .gitignore                            # Ficheiro para ignorar ficheiros irrelevantes.
├── README.md                             # Este manual de instruções.
└── requirements.txt                      # Lista de dependências para instalação.
```

## 🚀 Como Usar

Siga os passos abaixo para configurar o ambiente e executar o projeto.

### Pré-requisitos

-   Python 3.8 ou superior
-   Git

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/robertossoares/tcc-deteccao-fake-news.git
    cd tcc-deteccao-fake-news
    ```

2.  **Crie e ative um ambiente virtual (Recomendado):**
    ```bash
    # Criar o ambiente
    python -m venv venv

    # Ativar no Windows (Git Bash)
    source venv/Scripts/activate

    # Ativar no macOS/Linux
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

### Execução

Existem duas maneiras de interagir com o projeto:

#### 1. Análise Completa (Jupyter Notebook)

Para visualizar todo o processo de pesquisa, treinamento e geração de gráficos, inicie o Jupyter Lab:
```bash
jupyter lab
```
Em seguida, navegue até a pasta `notebooks/` e abra o ficheiro `tcc_treinamento_e_analise.ipynb`.

#### 2. Previsão Rápida (Interface CLI)

Para classificar uma nova notícia de forma rápida, execute o script a partir da raiz do projeto:
```bash
python scripts/verificar_noticia.py
```
O programa carregará os modelos e solicitará que você insira um texto para classificação. Para sair, digite `sair` ou `exit`.

## 🎓 Autoria

-   **Autor:** Roberto Santana Soares
-   **Instituição:** Unicesumar
-   **Curso:** Engenharia de Software
-   **Ano:** 2025
