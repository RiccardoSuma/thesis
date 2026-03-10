import ollama
import re

class LlamaProcessor:
    def __init__(self, model_name="qwen2.5"):
        self.model_name = model_name

    def clean_context(self, text):
        """Pulizia del testo per rimuovere rumore tecnico inutile."""
        if not text: return ""
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+\.ipynb', '', text)
        text = re.sub(r'In \[\d+\]:', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,;:\-\(\)\[\]\'\"èéàùò]{4,}', '', text)
        return " ".join(text.split())

    def format_context(self, payloads):
        context_str = ""
        # Ordino per timestamp per coerenza
        sorted_payloads = sorted(payloads, key=lambda x: (x.get('source'), x.get('timestamp', 0)))

        for i, p in enumerate(sorted_payloads):
            content = p.get('content_to_use') or p.get('text', '')
            clean_text = self.clean_context(content)
            
            if len(clean_text) < 15: continue

            modality = p.get('modality', 'UNK').upper()
            src_ref = f"File: {p.get('source')}, Time: {p.get('timestamp')}s"
            
            context_str += f"""
            <chunk id="{i}" type="{modality}">
                <source>{src_ref}</source>
                <content>{clean_text}</content>
            </chunk>
            """
        return context_str

    def chat_with_context(self, query_text, context_payloads):
        xml_context = self.format_context(context_payloads)

        # --- Prompt anti-ripetizione ---
        prompt = f"""
        Sei UniBot, un assistente accademico di livello esperto in Deep Learning, progettato per supportare uno studente magistrale in Ingegneria Elettronica estremamente analitico.
        Ti verranno forniti dei frammenti (chunk) di appunti universitari in formato Markdown (Contesto). Questi chunk possono contenere testo teorico, snippet di codice e descrizioni testuali di diagrammi/grafici generate da un modello visivo.

        REGOLA D'ORO:
        Il tuo compito è l'analisi tecnica rigorosa. Non sei qui per fare conversazione. Rispondi in modo diretto, denso e matematicamente ineccepibile.

        DATI (XML):
        {xml_context}

        ---
        DOMANDA: "{query_text}"
        ---

        REGOLE DI SINTESI E GENERAZIONE (CRITICHE):
        1. STRICT GROUNDING E ANTI-ALLUCINAZIONE: Basati ESCLUSIVAMENTE sulle informazioni presenti nei chunk forniti. 
           - NON usare conoscenze pre-addestrate per inventare risposte. 
           - Se il contesto è insufficiente o irrilevante per la domanda, DEVI rispondere ESATTAMENTE con: "Le informazioni recuperate dal contesto non sono sufficienti per rispondere in modo rigoroso a questa domanda." Nessuna ipotesi.

        2. FORMATTAZIONE MATEMATICA (TASSATIVO):
           - Usa RIGOROSAMENTE la sintassi LaTeX per qualsiasi variabile o formula.
           - Usa il dollaro singolo per le formule inline (es. $w_{{ij}}$, $\sigma(x)$).
           - Usa il doppio dollaro per le equazioni a blocco (es. $$ \Delta w_{{ij}} = -\eta \delta_j x_{{ij}} $$).
           - È SEVERAMENTE VIETATO usare parentesi quadre come \[ o \] o parentesi tonde come \( o \) per delimitare la matematica.

        3. STILE INGEGNERISTICO:
           - Niente frasi introduttive inutili come "Nel contesto fornito viene detto che...". Inizia subito con l'analisi.
           - Usa elenchi puntati per snocciolare parametri, vincoli o passaggi algoritmici.
           - Integra in modo fluido le descrizioni dei grafici/immagini con la teoria (es. "Come evidenziato dall'andamento della curva di loss...").

        4. DEDUPLICAZIONE E CITAZIONI:
           - Sintetizza le ripetizioni presenti in chunk diversi in un'unica argomentazione logica.
           - Ogni affermazione tecnica, equazione o definizione deve essere seguita dalla sua fonte alla fine della frase, usando il formato [NomeFile/Chunk]. Esempio: "Il learning rate decresce esponenzialmente [Chunk 4]."

        Rispondi nella stessa lingua della domanda
                """

        stream = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
            options={
                "temperature": 0.1,  # Bassa per rigore, ma non 0 assoluto per permettere un minimo di riformulazione linguistica
                "num_ctx": 8192,
                "repeat_penalty": 1.1 # PENALITÀ ALTA PER LE RIPETIZIONI
            }
        )
        
        for chunk in stream:
            yield chunk['message']['content']