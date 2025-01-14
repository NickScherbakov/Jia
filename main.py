import os
import re
import datetime
import sqlite3
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest
import requests
from dotenv import load_dotenv
from openai import OpenAI
from gigachat import GigaChat

# Load environment variables
load_dotenv()

# Get configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")
GIGACHAT_API_URL = os.getenv("GIGACHAT_API_URL")
GIGACHAT_MODEL = os.getenv("MYGIGACHAT_MODEL")

# Database configuration
DATABASE_PATH = "saphire.db"


@contextmanager
def get_db_connection():
    """Context manager for database operations."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize tables for storing dialogues."""
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_dialogues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                message_type TEXT NOT NULL,
                message_content TEXT NOT NULL,
                aspect TEXT,
                sequence_number INTEGER NOT NULL
            )
            """
        )
        conn.commit()


class TestCooperativeBehavior:
    @pytest.fixture(autouse=True)
    def setup(self):
        """General fixture for setting up necessary configurations."""
        # Initialize the database
        init_db()

        # Initialize clients
        self.openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_URL,
            timeout=60.0,
        )

        self.gigachat_client = GigaChat(
            credentials=GIGACHAT_API_KEY,
            base_url=GIGACHAT_API_URL,
            verify_ssl_certs=False,
            model=GIGACHAT_MODEL,
        )

        yield

        # Cleanup after tests
        if hasattr(self, "gigachat_client"):
            self.gigachat_client.close()

    def get_model_response(self, client_type, prompt):
        """Get a response from a specific model."""
        try:
            if client_type == "openai":
                response = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content

            elif client_type == "ollama":
                response = requests.post(
                    f"{OLLAMA_API_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                return response.json()["response"]

            elif client_type == "gigachat":
                response = self.gigachat_client.chat(prompt)
                return response.choices[0].message.content

        except Exception as e:
            return f"Error getting response from {client_type}: {str(e)}"

    def save_dialogue_to_db(self, test_name, dialogue_entries):
        """Save dialogue to the database."""
        with get_db_connection() as conn:
            for seq_num, entry in enumerate(dialogue_entries):
                conn.execute(
                    """
                    INSERT INTO model_dialogues 
                    (test_name, model_name, message_type, message_content, aspect, sequence_number)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        test_name,
                        entry.get("model", "system"),
                        entry.get("type", "message"),
                        entry["content"],
                        entry.get("aspect"),
                        seq_num,
                    ),
                )
            conn.commit()

    def get_dialogue_from_db(self, test_name):
        """Retrieve dialogue from the database."""
        with get_db_connection() as conn:
            return conn.execute(
                """
                SELECT * FROM model_dialogues 
                WHERE test_name = ? 
                ORDER BY sequence_number
                """,
                (test_name,),
            ).fetchall()

    def print_dialogue_section(self, title, content):
        """Print a dialogue section in a formatted way."""
        print("\n" + "=" * 80)
        print(f" {title} ".center(80, "="))
        print("=" * 80)
        print(content)
        print("-" * 80)

    def test_russian_dialogue(self):
        """Test dialogue between models in Russian."""
        dialogue_entries = []
        test_name = f"russian_dialogue_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initial topic
        topic = """
        Let's discuss the importance of collaboration between different language models 
        for solving complex tasks. How can we best utilize the strengths of each model?
        """

        dialogue_entries.append(
            {
                "model": "system",
                "type": "topic",
                "content": topic,
            }
        )

        models = ["openai", "ollama", "gigachat"]

        for i in range(3):
            for model in models:
                context = "\n".join([entry["content"] for entry in dialogue_entries])

                prompt = f"""
                Context of the previous discussion:
                {context}

                Please continue the dialogue, considering the following requirements:
                1. The response must be in Russian
                2. The response must be related to previous messages
                3. Provide a constructive suggestion or idea
                4. Response length - no more than 2-3 sentences
                """

                response = self.get_model_response(model, prompt)

                # Validate the response
                assert isinstance(response, str), f"Response from {model} must be a string"
                assert len(response) > 0, f"Response from {model} must not be empty"
                assert any(
                    char.isalpha() for char in response
                ), f"Response from {model} must contain letters"
                has_russian = bool(re.search("[а-яА-Я]", response))
                assert has_russian, f"Response from {model} must be in Russian"

                dialogue_entries.append(
                    {
                        "model": model,
                        "type": "response",
                        "content": response,
                    }
                )

                self.print_dialogue_section(f"Response from model {model}", response)

        # Save the dialogue to the database
        self.save_dialogue_to_db(test_name, dialogue_entries)

    def test_task_solving_dialogue(self):
        """Test collaborative task-solving by models."""
        dialogue_entries = []
        test_name = f"task_solving_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        task = """
        Task: Develop a concept for an educational platform for children.
        Each model should propose a solution for the following aspects:
        1. Technical aspect
        2. Pedagogical aspect
        3. User experience aspect
        """

        dialogue_entries.append(
            {
                "model": "system",
                "type": "task",
                "content": task,
            }
        )

        models = ["openai", "ollama", "gigachat"]
        aspects = {
            "openai": "technical aspect of the platform",
            "ollama": "pedagogical aspect of the platform",
            "gigachat": "user experience aspect",
        }

        responses = {}

        for model, aspect in aspects.items():
            prompt = f"""
            {task}

            Please propose a solution for the following aspect: {aspect}

            Requirements for the response:
            1. The response must be in Russian
            2. Provide a specific solution
            3. Explain how it will help in children's education
            4. Response length - 2-3 sentences
            """

            response = self.get_model_response(model, prompt)
            responses[model] = response

            dialogue_entries.append(
                {
                    "model": model,
                    "type": "aspect_response",
                    "content": response,
                    "aspect": aspect,
                }
            )

            # Validate the response
            assert isinstance(response, str), f"Response from {model} must be a string"
            assert len(response) > 0, f"Response from {model} must not be empty"
            has_russian = bool(re.search("[а-яА-Я]", response))
            assert has_russian, f"Response from {model} must be in Russian"

            self.print_dialogue_section(f"Response from model {model} for aspect '{aspect}'", response)

        final_prompt = f"""
        Analyze the proposed solutions and suggest how they can be combined:
        Technical solution: {responses['openai']}
        Pedagogical solution: {responses['ollama']}
        UX solution: {responses['gigachat']}

        Requirements for the response:
        1. The response must be in Russian
        2. Propose a concrete plan for combining the solutions
        3. Indicate the advantages of such a combination
        4. Response length - 3-4 sentences
        """

        print("\nFinal discussion of solutions:")
        for model in models:
            final_response = self.get_model_response(model, final_prompt)

            dialogue_entries.append(
                {
                    "model": model,
                    "type": "final_response",
                    "content": final_response,
                }
            )

            # Validate the final response
            assert isinstance(final_response, str), f"Final response from {model} must be a string"
            assert len(final_response) > 0, f"Final response from {model} must not be empty"
            has_russian = bool(re.search("[а-яА-Я]", final_response))
            assert has_russian, f"Final response from {model} must be in Russian"

            # Check for meaningful content
            min_words = 20
            word_count = len(final_response.split())
            assert word_count >= min_words, f"Final response from {model} is too short (less than {min_words} words)"

            self.print_dialogue_section(f"Final response from model {model}", final_response)
            responses[f"{model}_final"] = final_response

        # Save the entire dialogue to the database
        self.save_dialogue_to_db(test_name, dialogue_entries)

        # Ensure all models provided unique responses
        final_responses = [responses[f"{model}_final"] for model in models]
        assert len(set(final_responses)) == len(models), "Final responses from models must be unique"

        print("\nTest completed successfully: all models provided unique and meaningful responses in Russian")


def view_latest_test_results():
    """View the latest test results from the database."""
    with get_db_connection() as conn:
        # Retrieve the latest tests
        latest_tests = conn.execute(
            """
            SELECT DISTINCT test_name, timestamp 
            FROM model_dialogues 
            ORDER BY timestamp DESC 
            LIMIT 5
            """
        ).fetchall()

        for test in latest_tests:
            print(f"\n=== Test: {test['test_name']} ===")
            print(f"Time: {test['timestamp']}")

            # Retrieve messages for the test
            messages = conn.execute(
                """
                SELECT model_name, message_type, message_content, aspect
                FROM model_dialogues
                WHERE test_name = ?
                ORDER BY sequence_number
                """,
                (test["test_name"],),
            ).fetchall()

            for msg in messages:
                print(f"\nModel: {msg['model_name']}")
                if msg["aspect"]:
                    print(f"Aspect: {msg['aspect']}")
                print(f"Type: {msg['message_type']}")
                print(f"Message: {msg['message_content']}")
                print("-" * 40)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
