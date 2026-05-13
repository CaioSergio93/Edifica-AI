import streamlit as st
import os
import gc
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_cloudflare import CloudflareWorkersAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader

# Carrega variáveis do .env
load_dotenv()

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Consultor de Engenharia JP", page_icon="🏗️")
st.title("🏗️ Consultoria Técnica de Engenharia")
st.caption("Baseado estritamente nas normas e leis municipais de João Pessoa")

# --- CREDENCIAIS ---
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
DATA_PATH = os.getenv("DATA_PATH", "./dados")
DB_PATH = os.getenv("DB_PATH", "faiss_db")

# --- INICIALIZAÇÃO DE EMBEDDINGS ---
embeddings = CloudflareWorkersAIEmbeddings(
    account_id=CLOUDFLARE_ACCOUNT_ID,
    api_token=CLOUDFLARE_API_TOKEN,
    model_name="@cf/baai/bge-small-en-v1.5"
)

# --- FUNÇÕES DE NÚCLEO ---
@st.cache_resource
def carregar_ou_criar_db():
    if os.path.exists(os.path.join(DB_PATH, "index.faiss")):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        st.error(f"Erro: Documentos base não encontrados em '{DATA_PATH}'.")
        return None

    with st.status("Indexando legislação técnica...", expanded=True) as status:
        all_documents = []
        for file in os.listdir(DATA_PATH):
            if file.endswith(".pdf"):
                st.write(f"📖 Processando lei: {file}")
                loader = PyMuPDFLoader(os.path.join(DATA_PATH, file))
                all_documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
        texts = text_splitter.split_documents(all_documents)
        
        db = FAISS.from_documents(texts[:100], embeddings)
        for i in range(100, len(texts), 100):
            db.add_documents(texts[i : i + 100])
            st.write(f"✅ {min(i + 100, len(texts))}/{len(texts)} trechos processados")
        
        db.save_local(DB_PATH)
        status.update(label="Base Legal Carregada!", state="complete")
    return db

# --- SETUP DO RAG ---
vectorstore = carregar_ou_criar_db()

if vectorstore:
    # Temperatura 0 é vital para fidelidade à lei
    llm = ChatOpenAI(
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
        api_key=CLOUDFLARE_API_TOKEN,
        model="@cf/google/gemma-7b-it-lora",
        temperature=0 
    )

    # Aumentamos o K para pegar mais artigos relacionados
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    template = """Você é um Consultor Jurídico-Técnico de Engenharia Civil em João Pessoa, especializado em legislação municipal.

    Sua resposta deve ser estritamente fundamentada no contexto legal fornecido abaixo.
    REGRAS DE OURO:
    1. Se a informação não constar explicitamente no contexto, responda: "Esta informação não consta na base de dados oficial fornecida".
    2. Cite sempre o documento fonte e, se disponível no texto, o número do Artigo ou Decreto.
    3. Mantenha um tom jurídico-técnico formal e objetivo.
    4. Nunca utilize conhecimentos externos que contradigam os documentos base fornecidos.

    Contexto Legal:
    {context}

    Pergunta do Usuário: {question}

    Resposta Fundamentada:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([f"FONTE: {d.metadata.get('source', 'Documento Oficial')}\nCONTINGENTE: {d.page_content}" for d in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- INTERFACE DE CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_input := st.chat_input("Consulte um artigo ou norma técnica..."):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            with st.spinner("Consultando legislação..."):
                try:
                    response = rag_chain.invoke(prompt_input)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Erro na consulta: {e}")