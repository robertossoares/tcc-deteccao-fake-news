# Deteção de Fake News com Machine Learning e BERT

Este repositório contém o código e os modelos desenvolvidos para o Trabalho de Conclusão de Curso (TCC) em Engenharia de Software, focado na deteção de notícias falsas em português do Brasil.

## Sobre o Projeto

O objetivo deste projeto foi desenvolver e comparar a eficácia de diferentes algoritmos de aprendizado de máquina na tarefa de classificar textos noticiosos como verdadeiros ou falsos. Foram utilizados modelos clássicos (Naive Bayes, Random Forest, SVM) e um modelo de deep learning baseado em Transformers (BERT).

O dataset utilizado foi o **Fake.br-Corpus**, um corpus de notícias em português brasileiro.

## Estrutura do Repositório

- **/notebooks**: Contém o Jupyter Notebook `tcc_treinamento_e_analise.ipynb`, que detalha todo o processo de pesquisa.
- **/scripts**: Contém a interface de linha de comando `verificar_noticia.py` para realizar previsões.
- **/modelos_salvos**: Contém os ficheiros dos modelos finais e do vetorizador TF-IDF.
- **requirements.txt**: Lista das bibliotecas Python necessárias para executar o projeto.

## Como Usar

### Pré-requisitos

- Python 3.8+
- Git

### Instalação

1. Clone o repositório:
   ```bash
   git clone [https://github.com/](https://github.com/)[seu-utilizador]/tcc-deteccao-fake-news.git
   cd tcc-deteccao-fake-news