import streamlit as st
import subprocess
import time
import requests
import sys

# Import moduli interni
from modules.storage.qdrant_ops import VectorDB
from modules.llm.retrieve_info import InfoRetriever
from modules.llm.llm_interface import LlamaProcessor 

# --- AUTO-START OLLAMA FUNCTION ---
# --- AUTO-START OLLAMA FUNCTION (CORRETTA) ---
def ensure_ollama_started():
    """
    Controlla se Ollama è attivo sulla porta 11434.
    Se non risponde, prova ad avviarlo in background.
    """
    ollama_url = "http://localhost:11434"
    
    # 1. Controllo rapido se è già attivo
    try:
        requests.get(ollama_url, timeout=0.2)
        return True
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        pass # È spento o non risponde

    # 2. Se spento, avviamo il processo
    print("🔄 Ollama non rilevato. Avvio del server in background...")
    
    try:
        # Popen lancia il processo senza bloccare lo script Python
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Diamo 2 secondi di respiro al server per inizializzarsi
        time.sleep(2)
        
        # 3. Polling (attesa attiva) finché non risponde
        max_retries = 10 
        for i in range(max_retries):
            try:
                print(f"⏳ Waiting for Ollama... ({i+1}/{max_retries})")
                requests.get(ollama_url, timeout=1.0) # Timeout alzato a 1s
                print("✅ Ollama è pronto!")
                return True # Partito!
            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
                time.sleep(1)
        
        return False # Timeout dopo i tentativi
        
    except FileNotFoundError:
        st.error("❌ Comando 'ollama' non trovato. Assicurati di averlo installato.")
        return False

# ----------------------------------

# 1. THE ENGINE ROOM: Loaded once and shared across all sessions
@st.cache_resource
def load_rag_system():
    # Inizializziamo il DB
    db = VectorDB(collection_name="abb_video_collection")
    
    # Inizializziamo il Retriever
    retriever = InfoRetriever(db_wrapper=db)
    
    # Inizializziamo il LLM (Mistral o Llama3)
    # Nota: Questo passaggio richiede che Ollama sia GIA' attivo,
    # ecco perché facciamo il controllo nel main prima di chiamare questa funzione.
    llm = LlamaProcessor(model_name="mistral")
    
    return retriever, llm

class UniBotGUI:
    def __init__(self, retriever, llm, bot_name="ABBot"):
        self.bot_name = bot_name
        self.retriever = retriever
        self.llm = llm
        # Nota: setup_page rimosso da qui perché set_page_config va nel main
        self._initialize_session_state()

    def _initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def response_generator(self, user_input):
        """
        Ponte tra il retriever e il processore Mistral.
        """
        with st.spinner("🧠 Searching memory..."):
            # Recuperiamo i payload (che ora contengono 'content_to_use')
            context_payloads = self.retriever.get_answer(user_input, top_k=4)
            
            # Salviamo nel session_state per l'expander delle fonti
            st.session_state["current_context"] = context_payloads
            
        # Passiamo i payload al llm per la generazione streaming
        return self.llm.chat_with_context(user_input, context_payloads)

    def display_chat(self):
        # Mostra la cronologia
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Se il messaggio ha delle fonti salvate nella history, mostra l'expander
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Fonti storiche"):
                        for i, p in enumerate(message["sources"]):
                            st.caption(f"**Fonte {i+1}:** {p.get('source')} @ {p.get('timestamp')}s")

    def run(self):
        # Header GUI
        st.title(f"🤖 {self.bot_name}")
        st.markdown("---")

        # Sidebar info
        with st.sidebar:
            st.header("System Status")
            st.success("✅ Ollama Engine: Online")
            st.success("✅ Qdrant DB: Connected")
            st.markdown("---")
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

        self.display_chat()

        # Input Utente
        if prompt := st.chat_input("Hi, I'm ABBot, what can I help you with?"):
            # 1. Mostra messaggio utente
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Genera risposta assistente
            with st.chat_message("assistant"):
                # write_stream consuma il generatore
                full_response = st.write_stream(self.response_generator(prompt))
                
                # Recuperiamo i payload salvati in session_state durante la generazione
                payloads = st.session_state.get("current_context", [])
                
                if payloads:
                    with st.expander("🔍 Analisi Fonti (RAG)"):
                        for i, p in enumerate(payloads):
                            score_val = p.get('score', 0.0) 
                            m_type = p.get('modality', 'unknown').upper()
                            
                            st.markdown(f"**#{i+1} [{m_type}]** (Score: `{score_val:.4f}`)")
                            st.caption(f"📁 `{p.get('source')}` ⏱️ `{p.get('timestamp')}s`")
                            
                            # Logica di visualizzazione contenuto
                            content = p.get('content_to_use') or p.get('text') or "Nessun testo."
                            st.info(content)
                            
                            # Se c'è un'immagine, potremmo mostrarla (opzionale)
                            if p.get('image_path') and m_type == "VISUAL":
                                try:
                                    st.image(p.get('image_path'), width=200)
                                except:
                                    pass

            # 3. Salva tutto nella history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": payloads 
            })

if __name__ == "__main__":
    # 1. CONFIGURAZIONE PAGINA (Deve essere la PRIMA istruzione Streamlit)
    st.set_page_config(page_title="ABBot", layout="wide")

    # 2. CONTROLLO OLLAMA (Prima di caricare qualsiasi modello)
    # Se Ollama è spento, proviamo ad accenderlo e mostriamo uno spinner
    if not ensure_ollama_started():
        with st.spinner("🔌 Wake up Neo... (Starting Ollama Server in background)"):
             if not ensure_ollama_started(): # Doppio check con attesa
                st.error("❌ ERRORE CRITICO: Ollama non risponde sulla porta 11434.")
                st.warning("Apri un terminale e scrivi: 'ollama serve'")
                st.stop()
    
    # 3. CARICAMENTO SISTEMA (Ora sicuro perché Ollama è attivo)
    try:
        retriever_engine, llm_engine = load_rag_system()
    except Exception as e:
        st.error(f"Errore nel caricamento del RAG System: {e}")
        st.stop()
    
    # 4. AVVIO GUI
    bot = UniBotGUI(retriever=retriever_engine, llm=llm_engine)
    bot.run()