from llama_cpp import Llama
import json
from typing import Dict, List


class LocalLLMHandler:
    def __init__(self, model_path: str, n_ctx: int = 4096):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.llm = None
        self.load_model()

    def load_model(self):
        """Load the local LLM model"""
        try:
            # Configure for Apple Silicon M4
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_batch=512,
                n_gpu_layers=-1,  # Use Metal Performance Shaders
                verbose=False,
                n_threads=10  # Adjust based on your M4 cores
            )
            print("LLM model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load LLM: {str(e)}")

    def generate_response(self, prompt: str, max_tokens: int = 1000,
                          temperature: float = 0.3, stop_sequences: List[str] = None) -> str:
        """Generate response from the LLM"""
        if not self.llm:
            raise Exception("LLM not loaded")

        if stop_sequences is None:
            stop_sequences = ["Human:", "Assistant:", "\n\n---", "<s>", "</s>"]

        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=stop_sequences
            )

            generated_text = response['choices'][0]['text'].strip()

            # Additional cleanup
            if generated_text.endswith(('Human:', 'Assistant:', '---')):
                lines = generated_text.split('\n')
                generated_text = '\n'.join(lines[:-1])

            return generated_text

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    def create_quiz_prompt(self, context: str, topic: str, num_questions: int = 5) -> str:
        """Create a prompt for quiz generation"""
        prompt = f"""<s>[INST] You are a medical education expert. Based on the following medical guideline content, create EXACTLY {num_questions} multiple-choice quiz questions about {topic}.

CRITICAL REQUIREMENTS:
1. Generate EXACTLY {num_questions} questions - no more, no less
2. Each question must have 4 options (A, B, C, D)  
3. Only one option should be correct
4. Use information ONLY from the provided context
5. Format each question EXACTLY as shown below
6. Separate each question with "---QUESTION_SEPARATOR---"

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
