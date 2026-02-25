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
        Sei un Senior Engineer di ABB che assiste altri ingegneri.
        Ti verranno forniti dei frammenti di trascrizioni audio e slide (Contesto).
        I frammenti audio possono contenere linguaggio parlato frammentato o rumore.

        REGOLA D'ORO:
        Analizza attentamente TUTTO il contesto fornito. Non limitarti a una risposta secca. 
        Costruisci una spiegazione tecnica dettagliata, articolata e discorsiva. Se ci sono passaggi tecnici o parametri, elencali in modo strutturato.
        Se il contesto include esempi o motivazioni, includili nella tua risposta.

        DATI (XML):
        {xml_context}

        ---
        DOMANDA: "{query_text}"
        ---

        REGOLE DI SINTESI (CRITICHE):
        1.  **DEDUPLICAZIONE:** Spesso l'audio contiene ripetizioni. Se più chunk dicono la stessa cosa, scrivi UNA sola frase ben strutturata e raggruppa le citazioni alla fine.
            * ERRATO: "La batch norm serve a X [Fonte A]. Inoltre la batch norm aiuta X [Fonte B]."
            * CORRETTO: "La batch normalization è tecnica fondamentale per stabilizzare i gradienti e velocizzare il training [Fonte A][Fonte B]."
            **NO ALLUCINAZIONI:** Se i chunk recuperati sono frammentari o non pertinenti alla domanda (es. parlano di "software" generico invece che di SGD), RISPONDI: "Il materiale recuperato non è sufficiente per rispondere." NON RISPONDERE A MEMORIA. Non usare conoscenze esterne o generalizzazioni per rispondere alle domande
              se non supportate dai chunk.

            

        2.  **STILE:** Usa un tono professionale. Non usare frasi come "Nel chunk X viene detto...", ma esponi direttamente il concetto. Per casi numerici specifici, usa "ad esempio" o "come descritto in [Fonte]".

        
        3.  **GERARCHIA:**
            * Usa il testo [VISUAL] per definire i termini esatti.
            * Usa il testo [AUDIO] per spiegare il "perché" e il funzionamento.

        4.  **CITAZIONI:** Ogni affermazione deve avere la sua fonte [File: ..., Time: mm:ss]. Se una frase riassume più chunk, metti tutte le fonti pertinenti alla fine della frase.

        Genera la risposta seguendo queste regole. Non iniziare MAI la tua risposta dicendo 'Il materiale non è sufficiente' o frasi simili, dillo solo se il contesto ti permetterebbe di rispondere solo allucinando. 
        Se il contesto non contiene la risposta esatta al 100%, usa le informazioni disponibili per formulare l'ipotesi più tecnica e pertinente possibile, iniziando direttamente con l'analisi tecnica. 
        Se il contesto parla di argomenti correlati, usali per spiegare i meccanismi sottostanti. La risposta deve essere nella stessa lingua della domanda:
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