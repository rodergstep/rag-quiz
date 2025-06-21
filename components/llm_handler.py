from llama_cpp import Llama
import json
from typing import Dict, List, Optional
import logging
import os  # Import os for os.cpu_count()

# Configure logging for the LLM Handler
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalLLMHandler:
    """
    Handles interactions with a locally loaded Large Language Model (LLM)
    using the llama-cpp-python library, specifically for GGUF models.
    """

    def __init__(self, model_path: str, n_ctx: int = 8192):
        """
        Initializes the LocalLLMHandler.

        Args:
            model_path (str): The file path to the GGUF model (e.g., '.gguf' file).
            n_ctx (int): The context window size for the LLM. This determines how many
                         tokens the model can process at once. Adjust based on your
                         hardware (RAM/VRAM) and the model's capabilities.
                         A larger context uses more memory.
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.llm: Optional[Llama] = None  # Explicitly type as Optional Llama
        self.load_model()

    def load_model(self):
        """
        Loads the local LLM model using llama-cpp-python.
        Configures for Apple Silicon (MPS) usage.
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"LLM model file not found at: {self.model_path}")

            # Configure for Apple Silicon (MPS)
            # n_gpu_layers=-1 attempts to offload all layers to the GPU (MPS).
            # n_threads: Adjust based on your M4 core count. Leaving it or setting to 0 lets llama.cpp
            # usually determine an optimal number, e.g., 8 or 10.
            # n_batch: Batch size for prompt processing. Larger batches can be faster but use more VRAM.
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_batch=512,  # A reasonable default for batch processing
                n_gpu_layers=-1,  # Use Metal Performance Schedulers for GPU offloading
                verbose=False,  # Suppress llama.cpp verbose output
                # Use all logical cores, or default to 4 if detection fails
                n_threads=os.cpu_count() or 4
            )
            logger.info(
                f"LLM model loaded successfully from {self.model_path} with n_ctx={self.n_ctx} on GPU (MPS).")
        except FileNotFoundError as e:
            logger.critical(
                f"Failed to load LLM: Model file not found. {e}", exc_info=True)
            raise
        except Exception as e:
            logger.critical(
                f"Failed to load LLM from {self.model_path}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to load LLM: {str(e)}")

    def generate_response(self, prompt: str, max_tokens: int = 1000,
                          temperature: float = 0.3, stop_sequences: Optional[List[str]] = None) -> str:
        """
        Generates a text response from the loaded LLM.

        Args:
            prompt (str): The input prompt for the LLM.
            max_tokens (int): The maximum number of tokens to generate in the response.
            temperature (float): Controls the randomness of the output. Lower values (e.g., 0.1-0.3)
                                 make the output more deterministic, higher values (e.g., 0.7-1.0) make it more creative.
            stop_sequences (Optional[List[str]]): A list of sequences that, if encountered, will
                                                  stop the generation. Defaults to common chat delimiters.

        Returns:
            str: The generated text response, with additional cleanup.

        Raises:
            Exception: If the LLM is not loaded or an error occurs during generation.
        """
        if not self.llm:
            logger.error("LLM not loaded. Cannot generate response.")
            raise Exception(
                "LLM not loaded. Please ensure load_model() was successful.")

        # Default stop sequences for common chat formats and model specific tokens
        if stop_sequences is None:
            stop_sequences = ["Human:", "Assistant:",
                              "\n\n---", "<s>", "</s>", "[/INST]", "<|eot_id|>"]

        try:
            logger.debug(
                f"Generating response for prompt (first 100 chars): {prompt[:100]}...")
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,          # Nucleus sampling parameter
                repeat_penalty=1.1,  # Penalty for repeating token sequences
                stop=stop_sequences,
                echo=False          # Do not echo the prompt back in the response
            )

            # Extract generated text from the response structure
            generated_text = response['choices'][0]['text'].strip()

            # Additional cleanup for partial stop sequences or unwanted trailings
            # Iteratively remove lines that match common trailing patterns
            lines = generated_text.split('\n')
            cleaned_lines = []
            for line in reversed(lines):
                stripped_line = line.strip()
                if not any(stripped_line.endswith(s.strip()) for s in stop_sequences if s.strip()):
                    # Add to front to reconstruct in original order
                    cleaned_lines.insert(0, line)
                else:
                    logger.debug(
                        f"Removed trailing line due to stop sequence match: '{stripped_line}'")
                    # If a line matches a stop sequence, stop adding previous lines as well
                    break
            generated_text = '\n'.join(cleaned_lines).strip()

            logger.debug(
                f"Generated response (first 100 chars): {generated_text[:100]}...")
            return generated_text

        except Exception as e:
            logger.error(
                f"Error generating response from LLM: {str(e)}", exc_info=True)
            raise Exception(f"Error generating response from LLM: {str(e)}")

    def create_quiz_prompt(self, context: str, topic: str, num_questions: int = 5) -> str:
        """
        Creates a structured prompt for the LLM to generate multiple-choice quiz questions.
        Includes critical requirements and a mandatory output format.

        Args:
            context (str): The medical guideline content to base the questions on.
            topic (str): The specific topic for the quiz questions.
            num_questions (int): The exact number of questions to generate.

        Returns:
            str: The formatted prompt string.
        """
        prompt = f"""<s>[INST] You are a medical education expert. Based on the following medical guideline content, create EXACTLY {num_questions} multiple-choice quiz questions about {topic}.

CRITICAL REQUIREMENTS:
1. Generate EXACTLY {num_questions} questions - no more, no less
2. Each question must have 4 options (A, B, C, D)
3. Only one option should be correct
4. Use information ONLY from the provided context. Do NOT make up information.
5. Each question must cover a DIFFERENT aspect, fact, or subtopic of {topic}.
6. Vary the clinical scenario, diagnosis, treatment, epidemiology, and context within the questions.
7. Do NOT repeat question types, subtopics, or phrasings. Ensure maximum diversity among questions.
8. Do NOT generate questions that are too similar to each other.
9. Format each question EXACTLY as shown below.
10. Separate each question with "---QUESTION_SEPARATOR---"

Context:
{context}

MANDATORY FORMAT for each question:
Question: [Your question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A, B, C, or D]
---QUESTION_SEPARATOR---

Example:
Question: What is the first-line treatment for mild asthma?
A) Oral corticosteroids
B) Short-acting beta-agonists as needed
C) Long-acting beta-agonists
D) Leukotriene modifiers
Correct Answer: B
---QUESTION_SEPARATOR---

Now generate EXACTLY {num_questions} questions about {topic} using this exact format: [/INST]
"""
        return prompt

    def create_learning_objective_prompt(self, context: str, target_group: Optional[str] = None) -> str:
        """
        Creates a structured prompt for the LLM to extract clinically relevant learning objectives
        for medical continuing education in a specific JSON format, including few-shot examples.

        Args:
            context (str): The professional medical text (in German or English) to analyze.
            target_group (Optional[str]): The medical target audience for the learning objectives.
                                           E.g., "General Medicine", "Internal Medicine".
                                           If None, defaults to "Internal Medicine".

        Returns:
            str: The formatted prompt string, requesting JSON output.
        """
        # Default target group if not specified
        if target_group is None:
            target_group = "Internal Medicine"

        # Keyword list as provided by the user
        keyword_list = [
            "History", "Clinical Examination Finding", "Disease Symptom", "Leading Symptom", "Patient Assessment",
            "Symptom Definition", "Auscultation", "Comorbidity", "Antibody Diagnostics", "Biomarker",
            "Tumor Marker", "Genetic Examination", "Synovial Analysis", "Cost-Benefit",
            "Mechanism of Action of Medications", "Drug Properties", "Pharmacokinetics", "Antidote",
            "Drug Class", "Etiology", "Antibody", "Histology", "Immunological Basis", "Cause of Disease",
            "Cell Function", "Heredity", "Cytology", "Pathophysiological Basis", "Pathogen",
            "Disease Activity", "Disease Consequence", "Disease Complication", "Disease Course",
            "Remission Chance", "Survival Rate", "Cause of Death", "Drug Concentration", "Prevention",
            "Invasive Therapy", "Medication Administration", "Medication Dosage", "Medication Effect",
            "Medication Approval", "Prophylaxis", "Rehabilitation", "Pain Therapy", "Therapy Compliance",
            "Therapy Duration", "Initial Therapy", "Therapy Indication", "Therapy Interval",
            "Therapy Guideline", "Therapy Goal", "Response Rate", "Remission Criteria", "Vaccination",
            "Side Effect", "Perioperative Management", "Premedication", "Screening for Pre-existing Conditions",
            "Therapy Risk", "Therapy and Pregnancy", "Therapy Complication", "Therapy Contraindication",
            "Therapy Control", "Interaction"
        ]

        # Few-shot example 1 (Existing - without links)
        few_shot_example_input_1 = """
        Bei Patienten mit neu diagnostiziertem Vorhofflimmern und einem CHA2DS2-VASc-Score von 2 oder mehr wird eine Antikoagulation mit NOACs (z.B. Apixaban, Rivaroxaban) gegenüber Warfarin bevorzugt, sofern keine Kontraindikationen vorliegen. Dies basiert auf Studien, die eine vergleichbare Wirksamkeit bei geringerem Blutungsrisiko zeigten. Bei Niereninsuffizienz ist eine Dosisanpassung einiger NOACs erforderlich.
        """
        few_shot_example_output_1 = """
[
  {
    "thema": "Vorhofflimmern-Therapie",
    "kapitel": "Antikoagulation",
    "nummer": "001-01",
    "zielgruppe": "Internal Medicine",
    "lernziel": "Die Indikation zur oralen Antikoagulation bei neu diagnostiziertem Vorhofflimmern anhand des CHA2DS2-VASc-Scores stellen und die bevorzugte Substanzklasse auswählen.",
    "zielkompetenz": "Deciding",
    "kurzfassung": "Bei Patienten mit neu diagnostiziertem Vorhofflimmern und einem CHA2DS2-VASc-Score von 2 oder mehr wird eine Antikoagulation mit NOACs (z.B. Apixaban, Rivaroxaban) gegenüber Warfarin bevorzugt.",
    "primäre_literaturquelle": "",
    "leitlinien_link": "",
    "hinweis_fragefokus": "Choice for initial therapy – vignette with one-step decision",
    "specification": "Medication",
    "sub_specification": "Initial Therapy"
  }
]
"""

        # Few-shot example 2 (With DOI and PubMed links)
        few_shot_example_input_2 = """
        Eine neue Studie (doi:10.1056/NEJMoa1910243) zeigt, dass bei Patienten mit chronischer Herzinsuffizienz und reduzierter Ejektionsfraktion (HFrEF) SGLT2-Inhibitoren das Risiko für kardiovaskulären Tod oder Hospitalisierung aufgrund von Herzinsuffizienz signifikant reduzieren. Die Leitlinie der ESC von 2021 (https://www.escardio.org/guidelines) empfiehlt nun den Einsatz dieser Medikamente zusätzlich zur Standardtherapie. Eine Metaanalyse (PMID:34212345) bestätigt diese Ergebnisse.
        """
        few_shot_example_output_2 = """
[
  {
    "thema": "Herzinsuffizienz-Therapie",
    "kapitel": "Medikamentöse Therapie",
    "nummer": "001-02",
    "zielgruppe": "Internal Medicine",
    "lernziel": "Die Rolle von SGLT2-Inhibitoren in der Therapie der Herzinsuffizienz mit reduzierter Ejektionsfraktion basierend auf aktueller Evidenz bewerten.",
    "zielkompetenz": "Evaluating",
    "kurzfassung": "SGLT2-Inhibitoren reduzieren das Risiko für kardiovaskulären Tod oder Hospitalisierung aufgrund von Herzinsuffizienz bei Patienten mit HFrEF.",
    "primäre_literaturquelle": "https://doi.org/10.1056/NEJMoa1910243",
    "leitlinien_link": "https://www.escardio.org/guidelines",
    "hinweis_fragefokus": "Evaluation of a therapeutic approach based on current study data – not yet integrated into guidelines",
    "specification": "Medication",
    "sub_specification": "Medication Effect"
  },
  {
    "thema": "Herzinsuffizienz-Therapie",
    "kapitel": "Leitlinien",
    "nummer": "001-03",
    "zielgruppe": "Internal Medicine",
    "lernziel": "Die Empfehlungen der aktuellen ESC-Leitlinie zum Einsatz von SGLT2-Inhibitoren bei HFrEF anwenden.",
    "zielkompetenz": "Applying",
    "kurzfassung": "Die Leitlinie der ESC von 2021 empfiehlt nun den Einsatz dieser Medikamente zusätzlich zur Standardtherapie.",
    "primäre_literaturquelle": "https://pubmed.ncbi.nlm.nih.gov/34212345",
    "leitlinien_link": "https://www.escardio.org/guidelines",
    "hinweis_fragefokus": "Vergleichende Bewertung therapeutischer Alternativen – direkte Fachfrage",
    "specification": "Therapy",
    "sub_specification": "Therapy Guideline"
  }
]
"""
        prompt = f"""<s>[INST] Du bist Facharzt mit langjähriger klinischer Erfahrung und arbeitest als medizinischer Redakteur. Analysiere den folgenden professionellen medizinischen Fachtext (in deutscher oder englischer Sprache). Deine Aufgabe ist es, daraus ein oder mehrere strukturierte, klinisch relevante Lernziele für ärztliche Fortbildungen auf Facharztniveau (z. B. Rheumatolog:innen oder internistisch tätige Fachärzt:innen) in deutscher Sprache zu extrahieren.

"zielgruppe": "{target_group}"
Berücksichtige dabei die ärztliche Zielgruppe, sofern diese zu Beginn des Prompts angegeben ist. Richte die klinische Relevanz, die Komplexität und die Formulierung des Lernziels explizit an den Anforderungen dieser Zielgruppe aus – z. B. praxisrelevante Differenzialdiagnosen für Hausärzt:innen statt subspezifische Spezialfragen für ophthalmologische Fachärzt:innen.
Wenn keine Zielgruppe angegeben ist, formuliere das Lernziel standardmäßig für internistisch tätige Fachärzt:innen mit klinischer Erfahrung in der Patientenversorgung.

Ziel:
Erstelle ausschließlich prüfbare, evidenzbasierte Lernziele für Fachärzt:innen bzw. Ärzt:innen in Weiterbildung. Die Lernziele müssen sich methodisch eindeutig als Multiple-Choice-Frage (MCQ) umsetzen lassen und eine ärztliche Entscheidungssituation zwischen mindestens zwei medizinisch sinnvollen Handlungsoptionen abbilden (z. B. Substanz A vs. B, Indikation vs. Kontraindikation, Monitoring vs. Therapieanpassung, Diagnosestellun).

Wenn die Quelle eine Studie ist (z. B. DOI-Link oder PubMed-Link), extrahieren Sie nur dann ein Lernziel, wenn
– die Studie eine potenziell praxisrelevante Aussage enthält (z. B. klinischer Nutzen, Therapiewirksamkeit), und
– diese Aussage nicht auf konkretem Detailwissen der Studie basiert (z. B. kein Studienaufbau, keine exakten Zahlen, keine Autor:innen oder Studiendesigns).

Ziel ist die Formulierung eines Lernziels, das eine klinisch interpretierbare Evidenzbewertung ermöglicht – nicht das Abfragen von Studienwissen.

Verwenden Sie in diesem Fall die Markierung:
"lernziel_typ": "Evidenzbasiert ohne Handlungsempfehlung"

Im Fokus stehen:
– Optimale Therapie nach Evidenz
– Therapieentscheidungen
– Therapiesicherheit
– Medikamentöse Maßnahmen
– Differenzialdiagnostische Abgrenzung
– Diagnostisches Vorgehen
– Diagnosestellung
- Verdachtsdiagnose stellen

Wenn keine geeigneten Therapieaussagen oder zur diagnostischen Entscheidungsfindung enthalten sind, können differenzierende Aussagen zu Klassifikation, Prognose oder Differenzialdiagnose verwendet werden.

Ein Lernziel ist nur dann zulässig, wenn:
– es isoliert prüfbar ist (also ohne die Antwortoptionen gelöst werden kann)
– mindestens zwei medizinisch plausible, homogene Handlungsoptionen erkennbar sind
– eine medizinische Konsequenz oder Differenz erkennbar ist
– es auf einer klar quantifizierten Empfehlung oder Studiendaten beruht
– es sich eindeutig in ein methodisch klares MCQ-Format (z. B. Vignette, 2-Step, direkte Frage) überführen lässt
– die zugrundeliegende Quelle entweder eine Leitlinie, eine randomisierte kontrollierte Studie (RCT), eine systematische Metaanalyse oder ein Expertenreview mit systematischer Literaturauswertung und kritischer Bewertung ist
– bei Expertenreviews ausschließlich dann ein Lernziel erstellt wird, wenn darin eine oder mehrere evidenzbasierte Primärquellen zitiert und bewertet werden, aus denen eine klinische Konsequenz hervorgeht

Verwerfe ein Lernziel konsequent, wenn:
– Das Lernziel nur dadurch lösbar ist, dass die Antwortoptionen gelesen werden – nicht aber aus dem Lernziel selbst
– die Quelle ausschließlich auf einer nicht belegten Expertenmeinung oder klinischen Intuition beruht, ohne Bezug zu systematisch bewerteter Literatur
– keine explizite Einschätzung oder Gewichtung der Evidenzlage erfolgt (z. B. nur narrativ, ohne Bezug auf Studiendaten, p-Werte, Effektstärken)
– **(Gelockert)** Es keine *ausdrückliche* Über- oder Unterlegenheit beschrieben ist. Stattdessen fokussiere auf klare klinische Empfehlungen oder Entscheidungen, auch wenn keine direkten, quantifizierten Vergleiche von zwei Optionen im Text stehen. Es muss aber ein Unterschied oder eine Präferenz klar werden.
– **(Gelockert)** Der Effekt nur als allgemeine Tendenz oder „kann“, „könnte“ beschrieben wird. Stattdessen erwarte eine klare Empfehlung oder einen Fakt mit klinischer Relevanz, selbst wenn keine exakte Effektgröße oder p-Wert im *aktuell vorliegenden Kontext* genannt wird, solange die Aussage evidenzbasiert ist (z.B. aus einer Leitlinie).
– **(Gelockert)** Zwei Verfahren, Medikamente oder Parameter *gleichwertig* beschrieben werden. Wenn die Gleichwertigkeit klinisch relevant ist, kann das Lernziel die **Entscheidungskriterien für die Wahl zwischen diesen gleichwertigen Optionen** beleuchten (z.B. basierend auf Nebenwirkungen, Patientenfaktoren, Kosten), wenn der Kontext diese Informationen bietet.
– **(Gelockert)** Ein Vergleich nur den Status quo beschreibt, ohne klinische Konsequenz (z. B. „PPV und SVV vergleichbar besser als CVP“, aber nicht: „PPV ist überlegen gegenüber SVV“). Finde stattdessen eine ableitbare klinische Konsequenz.

Literaturquelle (primäre_literaturquelle):
Literaturquelle:
Übernimm nur solche Quellen in das Feld primäre_literaturquelle, die im Text vollständig und eindeutig referenziert sind – entweder direkt als DOI-Link (z. B. https://doi.org/...) oder als funktionierender PubMed-Link (https://pubmed.ncbi.nlm.nih.gov/...).

Wenn im Text nur eine bibliografische Angabe enthalten ist (z. B. „Azari & Barney, JAMA 2013“) ohne Link, darf kein Link ergänzt werden – auch nicht bei vermeintlich eindeutiger Zuordnung.

Verwende niemals generierte, automatisch recherchierte oder erfundene Links.
Wenn kein funktionierender Link aus dem Text hervorgeht, bleibt das Feld primäre_literaturquelle leer.
Keine Platzhalter, keine Ergänzungen, keine URL-Erzeugung.

Leitlinienlink (leitlinien_link):
Wenn im Text explizit auf eine Leitlinie verwiesen wird (z. B. durch S3-Klassifikation, AWMF-Registernummer, RKI-Ratgeber, NICE- oder AAO-Leitlinie), übernimm den genannten Link exakt so, wie er im Text steht – z. B. register.awmf.org/... oder rki.de/....
Wenn kein konkreter Link zur Leitlinie genannt ist, aber eine Empfehlung erkennbar leitlinienbasiert ist (z. B. „gemäß S3-Leitlinie 2022“), dann:

Feld leitlinien_link: leer lassen

Feld free_tag_5: trage "S3-Leitlinie 2022" (oder entsprechendes Jahr und Klassifikation) ein

Wenn keine Leitlinie erwähnt wird, bleiben sowohl leitlinien_link als auch free_tag_5 leer.

Wenn kein Leitlinienbezug besteht oder im Text ersichtlich ist, lass das Feld ebenfalls leer.
Hinweis zur Studienlage:
Falls ein Lernziel ausschließlich auf aktuellen Studien basiert, die noch nicht in Leitlinien integriert sind, kennzeichne dies im "hinweis_fragefokus" mit:
"Bewertung eines Therapieansatzes auf Basis aktueller Studienlage – noch nicht leitlinienintegriert".
Lernziele, die ausschließlich auf Expertenmeinung ohne evidenzbasierte Quellenbezüge beruhen, dürfen nicht generiert werden.

Zielkompetenz:
Gib exakt eine Zielkompetenz pro Lernziel an. Die Kompetenz darf nicht im Lernziel selbst wiederholt werden. Wähle aus:
– Wissen (Fakten, Definitionen, feste Werte)
– Verstehen (Pathomechanismen, Klassifikation)
– Anwenden (Scores, Grenzwerte, Dosierungen)
– Entscheiden (therapeutische oder diagnostische Auswahl)
– Bewerten (Risiko-Nutzen, Studiendaten, klinische Aussagekraft)

Vermeide Doppelformulierungen wie „kennen und bewerten“. Wenn mehrere Aspekte im Text enthalten sind, formuliere zwei getrennte Lernziele.

Use `specification` from the following fixed terms: Diagnosis, Classification, Clinical Examination, Comorbidity, Laboratory Diagnostics, Medication, Pathophysiology, Prognosis, Therapy, Therapy Safety.

Use `sub_specification` from the following keyword list. If a new term is needed, it must be clear and präzise (z. B. „Therapieindikation“, nicht „Indikation“):
{", ".join(keyword_list)}

---
Here is an example to follow:
Input Context:
{few_shot_example_input_1}
Output JSON:
{few_shot_example_output_1}
---
Here is another example to follow for primary literature and guideline links:
Input Context:
{few_shot_example_input_2}
Output JSON:
{few_shot_example_output_2}
---

Format:
Return the output exclusively in the following JSON array format – ohne erklärenden Text, ohne Markdown, nur reines JSON. If no suitable learning objective can be extracted, return an empty JSON array:
[
{{
"thema": "...",
"kapitel": "...",
"nummer": "...",
"zielgruppe":"...",
"lernziel": "...",
"zielkompetenz": "...",
"kurzfassung": "...",
"primäre_literaturquelle": "...",
"leitlinien_link":"...",
"hinweis_fragefokus": "...",
"specification": "...",
"sub_specification": "..."
}}
]

Felddefinitionen:
thema: Übergeordneter Block mit Krankheitsbezug, z. B. „Therapie bei COPD“
kapitel: Optionaler Abschnittstitel oder Gliederungspunkt
nummer: Fortlaufende ID im Format nnn-xx (z. B. 001-01). Die Nummerierung erfolgt projektweit fortlaufend. Kein Zurücksetzen pro Thema.
zielgruppe: Primäre Zielgruppe der ärztlichen Lerninhalte sind Ärzte in einer bestimmten Fachrichtung, z. B. "Allgemeinmedizin", "Innere Medizin", "Notfallmedizin". Muss mit den Inhalten des Lernziels übereinstimmen. Kein Freitext, keine mehrfachen Nennungen.

lernziel: Klinisch prüfbares Ziel aus Sicht eines Facharztes. Die Zielkompetenz darf nicht im Wortlaut wiederholt werden. Die korrekte Antwort darf nicht wörtlich genannt sein – beschreibe stattdessen das klinische Dilemma.
zielkompetenz: Wissen, Verstehen, Anwenden, Entscheiden, Bewerten
kurzfassung: 1–2 Sätze direkt aus dem Quelltext – exakt übernommen, keine Umformulierungen
primäre_literaturquelle: DOI-Link der zitierten Studie oder leer
hinweis_fragefokus: Konkreter Umsetzungsvorschlag für MCQ. Beispiele:
– „Auswahl bei Ersttherapie – Vignette mit One-Step-Entscheidung“
– „2-Step: Interpretation + Auswahl bei Kontraindikation“
– „Vergleichende Bewertung therapeutischer Alternativen – direkte Fachfrage“
– „Bewertung klinischer Aussage – Entscheidung anhand Studiendaten“
– „Bewertung eines Therapieansatzes auf Basis aktueller Studienlage – noch nicht leitlinienintegriert“
specification: Einer der festen Begriffe: Diagnose, Klassifikation, Klinische Untersuchung, Komorbidität, Labordiagnostik, Medikament, Pathophysiologie, Prognose, Therapie, Therapiesicherheit
sub_specification: Begriffe aus Schlagwortliste verwenden. Wenn neu, dann klar und präzise (z. B. „Therapieindikation“, nicht „Indikation“)

Context:
{context}

Jetzt, basierend NUR auf dem bereitgestellten Kontext und den oben genannten Regeln, extrahiere Lernziele im angegebenen JSON-Format. [/INST]
"""
        return prompt
