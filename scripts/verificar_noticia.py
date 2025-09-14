import os
import re
import joblib
import torch
import unidecode
import nltk
from transformers import BertTokenizer, BertForSequenceClassification

# ETAPA 1: CONFIGURAÇÃO INICIAL E PRÉ-PROCESSAMENTO

# Função de pré-processamento
def preprocess_text(text):
    """Limpa e normaliza um texto em português para os modelos clássicos."""
    try:
        stopwords_pt = nltk.corpus.stopwords.words("portuguese")
        stemmer_pt = nltk.stem.RSLPStemmer()
    except LookupError:
        print("Baixando pacotes NLTK necessários...")
        nltk.download("stopwords", quiet=True)
        nltk.download("rslp", quiet=True)
        stopwords_pt = nltk.corpus.stopwords.words("portuguese")
        stemmer_pt = nltk.stem.RSLPStemmer()

    text = str(text).lower()
    text = unidecode.unidecode(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [stemmer_pt.stem(word) for word in tokens if word not in stopwords_pt]
    return " ".join(tokens)

# ETAPA 2: CARREGAMENTO DOS MODELOS

def carregar_modelos():
    """Carrega todos os modelos e artefatos salvos do disco."""
    print("Carregando modelos e artefatos... Por favor, aguarde.")
    
    # Verifica se o diretório de modelos existe
    if not os.path.exists('modelos_salvos'):
        print("\nERRO: O diretório 'modelos_salvos' não foi encontrado.")
        print("Certifique-se de que este script está no mesmo local que a pasta com os modelos salvos.")
        return None

    try:
        # Carrega o vetorizador
        vectorizer = joblib.load('modelos_salvos/vetorizador_tfidf.pkl')

        # Carrega os modelos clássicos
        svm_model = joblib.load('modelos_salvos/modelo_svm.pkl')
        rf_model = joblib.load('modelos_salvos/modelo_rf.pkl')
        nb_model = joblib.load('modelos_salvos/modelo_nb.pkl')

        # Configura o dispositivo (CPU, pois é mais comum em ambientes de usuário final)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo de inferência do BERT: {device}")

        # Carrega o tokenizador e o modelo BERT
        bert_tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
        bert_model = BertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=2)
        bert_model.load_state_dict(torch.load('modelos_salvos/modelo_bert.bin', map_location=device))
        bert_model.to(device)
        bert_model.eval() # Coloca o BERT em modo de avaliação

        modelos = {
            "vectorizer": vectorizer,
            "svm": svm_model,
            "rf": rf_model,
            "nb": nb_model,
            "bert_tokenizer": bert_tokenizer,
            "bert_model": bert_model,
            "device": device
        }
        
        print("Modelos carregados com sucesso!")
        return modelos

    except FileNotFoundError as e:
        print(f"\nERRO: Arquivo de modelo não encontrado: {e.fileName}")
        print("Verifique se todos os arquivos .pkl e .bin estão na pasta 'modelos_salvos'.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao carregar os modelos: {e}")
        return None


# ETAPA 3: FUNÇÃO DE PREVISÃO

def prever_noticia(texto, modelos):
    """Realiza a previsão de um texto usando todos os modelos carregados."""
    if not texto or texto.isspace():
        print("Texto inválido. Por favor, insira uma notícia.")
        return

    print("\n" + "="*60)
    print(f"Analisando: '{texto[:100]}...'")
    print("-"*60)
    
    # Previsão com modelos clássicos
    texto_limpo = preprocess_text(texto)
    texto_vetorizado = modelos["vectorizer"].transform([texto_limpo])
    
    pred_svm = modelos["svm"].predict(texto_vetorizado)[0]
    pred_rf = modelos["rf"].predict(texto_vetorizado)[0]
    pred_nb = modelos["nb"].predict(texto_vetorizado)[0]
    
    print(f"Previsão (SVM): {pred_svm.upper()}")
    print(f"Previsão (Random Forest): {pred_rf.upper()}")
    print(f"Previsão (Naive Bayes): {pred_nb.upper()}")

    # Previsão com BERT
    tokenizer = modelos["bert_tokenizer"]
    model = modelos["bert_model"]
    device = modelos["device"]
    
    encoded_text = tokenizer.encode_plus(
        texto, max_length=128, add_special_tokens=True, return_token_type_ids=False,
        padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    prediction = torch.argmax(outputs.logits, dim=1).item()
    rev_labels_map = {0: 'verdadeira', 1: 'falsa'}
    
    print(f"Previsão (BERT): {rev_labels_map[prediction].upper()}")
    print("="*60)

# ETAPA 4: INTERFACE PRINCIPAL

if __name__ == "__main__":
    modelos_carregados = carregar_modelos()

    if modelos_carregados:
        print("\n--- Sistema de Detecção de Fake News ---")
        print("Digite o texto de uma notícia e pressione Enter para analisar.")
        print("Digite 'sair' ou 'exit' para fechar o programa.")

        while True:
            try:
                texto_usuario = input("\nDigite a notícia: ")
                if texto_usuario.lower() in ['sair', 'exit']:
                    print("Encerrando o programa. Até logo!")
                    break
                prever_noticia(texto_usuario, modelos_carregados)
            except KeyboardInterrupt:
                print("\nEncerrando o programa. Até logo!")
                break