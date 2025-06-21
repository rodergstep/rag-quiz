import pandas as pd
import re
from typing import List, Dict
import csv
from io import StringIO
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    def levenshtein_distance(a, b):
        # Simple fallback if python-Levenshtein is not installed
        if a == b:
            return 0
        if len(a) < len(b):
            return levenshtein_distance(b, a)
        if len(b) == 0:
            return len(a)
        previous_row = range(len(b) + 1)
        for i, c1 in enumerate(a):
            current_row = [i + 1]
            for j, c2 in enumerate(b):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]


class QuizGenerator:
    def __init__(self, llm_handler, vector_store, embedding_handler):
        self.llm_handler = llm_handler
        self.vector_store = vector_store
        self.embedding_handler = embedding_handler

    def generate_quiz(self, topic: str, num_questions: int = 10, max_context_chunks: int = 20) -> pd.DataFrame:
        """Generate quiz questions based on topic"""
        # Increase context chunks for better coverage
        query_embedding = self.embedding_handler.encode_query(topic)
        search_results = self.vector_store.similarity_search(
            query_embedding.tolist(),
            k=max_context_chunks
        )

        # Combine retrieved context
        context = "\n\n".join(search_results['documents'])
        if len(context) > 8000:
            context = context[:8000] + "..."

        all_questions = []
        batch_size = 2  # Generate questions in slightly larger batches
        remaining_questions = num_questions

        while remaining_questions > 0 and len(all_questions) < num_questions:
            current_batch = min(batch_size, remaining_questions)

            # Generate batch of questions with higher temperature
            prompt = self.llm_handler.create_quiz_prompt(
                context, topic, current_batch)

            try:
                quiz_text = self.llm_handler.generate_response(
                    prompt,
                    max_tokens=2000 + (current_batch * 200),
                    temperature=0.7  # Increased temperature for more variety
                )

                # Parse the generated quiz
                questions = self.parse_quiz_text(quiz_text)

                if questions:
                    all_questions.extend(questions)
                    remaining_questions -= len(questions)

                    # If we need more questions, create a follow-up prompt
                    if remaining_questions > 0:
                        follow_up_prompt = self.create_follow_up_prompt(
                            context, topic, remaining_questions, all_questions)
                        additional_quiz = self.llm_handler.generate_response(
                            follow_up_prompt,
                            max_tokens=1500 + (remaining_questions * 150),
                            temperature=0.8  # Even higher temperature for follow-up
                        )
                        additional_questions = self.parse_quiz_text(
                            additional_quiz)
                        if additional_questions:
                            all_questions.extend(additional_questions)
                            remaining_questions -= len(additional_questions)

            except Exception as e:
                print(f"Error generating batch of questions: {str(e)}")
                if len(all_questions) == 0:
                    raise Exception(
                        f"Failed to generate any questions: {str(e)}")
                break

        # Deduplicate questions by their text (case-insensitive, strip whitespace), using fuzzy matching
        unique_questions = []
        seen = []
        for q in all_questions:
            q_text = q["Question"].strip().lower()
            is_duplicate = False
            for seen_text in seen:
                if levenshtein_distance(q_text, seen_text) < 10:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_questions.append(q)
                seen.append(q_text)
            if len(unique_questions) >= num_questions:
                break

        if unique_questions:
            df = pd.DataFrame(unique_questions[:num_questions])
            return df
        else:
            raise Exception("No valid questions could be generated")

    def create_follow_up_prompt(self, context: str, topic: str, num_remaining: int, existing_questions: List[Dict]) -> str:
        """Create a follow-up prompt for remaining questions"""
        existing_topics = [q.get('Question', '') for q in existing_questions]
        existing_summary = "\n".join([f"- {q}" for q in existing_topics])

        prompt = f"""<s>[INST] You already generated some questions about {topic}. Here are the previous questions:

{existing_summary}

Now generate {num_remaining} MORE questions about {topic} using this medical content. 
**IMPORTANT:** Each new question must be about a DIFFERENT aspect, fact, or subtopic than the previous questions. 
Do NOT repeat the same question type, phrasing, or focus. 
Vary the clinical scenario, diagnosis, treatment, and context.

Context:
{context[:6000]}

Generate EXACTLY {num_remaining} questions using this format:
Question: [Your question here]
A) [Option A]
B) [Option B]
C) [Option C] 
D) [Option D]
Correct Answer: [A, B, C, or D]
---QUESTION_SEPARATOR---

Generate {num_remaining} questions now: [/INST]
"""
        return prompt

    def parse_quiz_text(self, quiz_text: str) -> List[Dict[str, str]]:
        """Parse the generated quiz text into structured format"""
        questions = []

        # First try separator-based parsing
        if "---QUESTION_SEPARATOR---" in quiz_text:
            question_blocks = quiz_text.split("---QUESTION_SEPARATOR---")
        else:
            # Fallback: split by multiple patterns
            patterns = [
                r'\n(?=Question \d+:)',
                r'\n(?=Question:)',
                r'\n(?=Q\d+:)',
                r'\n(?=\d+\.)',
                r'\n\n(?=\w+.*\?)'
            ]

            question_blocks = [quiz_text]  # Start with full text

            for pattern in patterns:
                new_blocks = []
                for block in question_blocks:
                    splits = re.split(pattern, block)
                    new_blocks.extend(splits)
                question_blocks = new_blocks
                if len(question_blocks) > 1:
                    break

        for block in question_blocks:
            if not block.strip():
                continue

            question_data = self.extract_question_data(block)
            if question_data:
                questions.append(question_data)

        return questions

    def extract_question_data(self, block: str) -> Dict[str, str]:
        """Extract question data from a text block"""
        try:
            # Clean the block
            block = block.strip()
            if not block:
                return None

            lines = [line.strip()
                     for line in block.split('\n') if line.strip()]

            question = ""
            options = {"A": "", "B": "", "C": "", "D": ""}
            correct_answer = ""

            for line in lines:
                # Remove any leading "Question:", "Q1:", "Q2:", etc.
                if re.match(r'^(Question:|Q\d+:|\d+\.)\s*', line):
                    # Remove all such prefixes
                    question = re.sub(r'^(Question:|Q\d+:|\d+\.)\s*', '', line)
                elif not question and line.endswith('?') and not any(prefix in line for prefix in ['A)', 'B)', 'C)', 'D)', 'Correct']):
                    question = line

                # Option patterns
                elif re.match(r'^A[).]\s*', line):
                    options["A"] = re.sub(r'^A[).]\s*', '', line)
                elif re.match(r'^B[).]\s*', line):
                    options["B"] = re.sub(r'^B[).]\s*', '', line)
                elif re.match(r'^C[).]\s*', line):
                    options["C"] = re.sub(r'^C[).]\s*', '', line)
                elif re.match(r'^D[).]\s*', line):
                    options["D"] = re.sub(r'^D[).]\s*', '', line)

                # Correct answer patterns
                elif 'Correct Answer:' in line or 'Answer:' in line:
                    answer_part = line.split('Answer:')[1].strip()
                    for letter in ['A', 'B', 'C', 'D']:
                        if letter in answer_part:
                            correct_answer = letter
                            break

            # Final cleanup: Remove any remaining "Question:" prefix just in case
            question = re.sub(r'^(Question:|Q\d+:|\d+\.)\s*',
                              '', question).strip()

            # Validation
            if question and all(options.values()) and correct_answer in ['A', 'B', 'C', 'D']:
                return {
                    "Question": question,
                    "Option A": options["A"],
                    "Option B": options["B"],
                    "Option C": options["C"],
                    "Option D": options["D"],
                    "Correct Answer": correct_answer
                }
            else:
                if question:
                    print(
                        f"Debug - Incomplete question: Q='{question}', A='{options['A']}', B='{options['B']}', C='{options['C']}', D='{options['D']}', Correct='{correct_answer}'")

        except Exception as e:
            print(f"Error parsing question block: {e}")
            print(f"Block content: {block[:200]}...")

        return None

    def save_quiz_to_csv(self, df: pd.DataFrame, filename: str) -> str:
        """Save quiz DataFrame to CSV file"""
        try:
            df.to_csv(filename, index=False)
            return filename
        except Exception as e:
            raise Exception(f"Error saving CSV: {str(e)}")
