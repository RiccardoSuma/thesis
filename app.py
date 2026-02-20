import streamlit as st
import time
from modules.storage.qdrant_ops import VectorDB
from modules.llm.retrieve_info import InfoRetriever
from modules.llm.llm_interface import LlamaProcessor 

# 1. THE ENGINE ROOM
@st.cache_resource
def load_rag_system():
    db = VectorDB(collection_name="video_collection")
    retriever = InfoRetriever(db_wrapper=db)
    llm = LlamaProcessor(model_name="mistral")
    return retriever, llm

class UniBotGUI:
    def __init__(self, retriever, llm, bot_name="UniBOT 🎓"):
        self.bot_name = bot_name
        self.retriever = retriever
        self.llm = llm
        self._setup_page()
        self._initialize_session_state()

    def _setup_page(self):
        st.set_page_config(page_title="UniBOT", page_icon="🎓", layout="wide")
        st.title(self.bot_name)
        st.markdown("---")

    def _initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def response_generator(self, user_input):
        """
        Bridges the retriever script with the Mistral processor.
        """
        with st.spinner("Searching through lectures..."):
            context_payloads = self.retriever.get_answer(user_input, top_k=4)
            
            # Salvo nel session_state per l'expander delle fonti
            st.session_state["current_context"] = context_payloads
            
        # Passo i payload al llm
        return self.llm.chat_with_context(user_input, context_payloads)

    def display_chat(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Se il messaggio ha delle fonti salvate, mostra l'expander
                if "sources" in message and message["sources"]:
                    with st.expander("🔬 Fonti recuperate"):
                        for i, p in enumerate(message["sources"]):
                            st.write(f"**Fonte {i+1}:** {p.get('source')} @ {p.get('timestamp')}s")

    def run(self):
        # Sidebar info
        with st.sidebar:
            st.header("Session Info")
            st.info("System: Mistral + CLIP + Qdrant (Hybrid)")
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

        self.display_chat()

        if prompt := st.chat_input("Ask about Deep Learning..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                full_response = st.write_stream(self.response_generator(prompt))
                

                payloads = st.session_state.get("current_context", [])
                
                if payloads:
                    with st.expander("🔬 Analisi Multimodale del Recupero"):
                        for i, p in enumerate(payloads):
                            score_val = p.get('score', 0.0) 
                            m_type = p.get('modality', 'unknown').upper()
                            
                            st.write(f"**Fonte {i+1}** (Score: {score_val:.4f}) - 🔊 {m_type}")
                            st.markdown(f"**File:** `{p.get('source')}` @ {p.get('timestamp')}s")

                            content = p.get('content_to_use') or p.get('text') or "Nessun testo estratto."
                            st.info(content)
                            # ----------------------------
            
            # Salvataggio completo nel messaggio
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": payloads 
            })

if __name__ == "__main__":
    # Initialize engines
    retriever_engine, llm_engine = load_rag_system()
    
    # Start GUI
    bot = UniBotGUI(retriever=retriever_engine, llm=llm_engine)
    bot.run()