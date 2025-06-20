import unittest
from unittest.mock import patch, MagicMock, call
import os
from typing import Generator

# Assuming the tests will be run from the root of the project
# If not, sys.path adjustments might be needed here or in the test runner
from termite.shared.call_llm import (
    call_llm,
    call_ollama,
    call_gemini,
    get_llm_provider,
)

class TestCallLLM(unittest.TestCase):
    def setUp(self):
        """Clear relevant environment variables before each test."""
        self.original_environ = os.environ.copy()
        vars_to_clear = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OLLAMA_MODEL"]
        for var in vars_to_clear:
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self.original_environ)

    # Test methods will be added here

    @patch('termite.shared.call_llm.chat')
    def test_call_ollama_non_streaming(self, mock_ollama_chat):
        # Configure mock
        mock_response = MagicMock()
        mock_response.message.content = "ollama_test_response"
        mock_ollama_chat.return_value = mock_response

        # Set environment variable
        os.environ["OLLAMA_MODEL"] = "test_ollama_model"

        # Call the function
        system_prompt = "System prompt"
        messages = [{"role": "user", "content": "Hello"}]
        response = call_ollama(system_prompt, messages)

        # Assertions
        self.assertEqual(response, "ollama_test_response")
        mock_ollama_chat.assert_called_once_with(
            model="test_ollama_model",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Hello"}
            ],
            # stream=False is default, so not explicitly passed in current implementation
        )

    @patch('termite.shared.call_llm.chat')
    def test_call_ollama_streaming(self, mock_ollama_chat):
        # Configure mock for streaming
        stream_chunks = [
            {"message": {"content": "chunk1"}},
            {"message": {"content": " chunk2"}},
            {"message": {"content": "chunk3"}},
            # Simulate a chunk that might be empty or without content
            {"message": {}},
            {},
        ]
        mock_ollama_chat.return_value = iter(stream_chunks)

        # Set environment variable
        os.environ["OLLAMA_MODEL"] = "test_ollama_model_stream"

        # Call the function
        system_prompt = "System stream prompt"
        messages = [{"role": "user", "content": "Stream hello"}]
        response_generator = call_ollama(system_prompt, messages, stream=True)

        # Assertions
        self.assertIsInstance(response_generator, Generator)
        responses = list(response_generator)
        self.assertEqual(responses, ["chunk1", " chunk2", "chunk3"])

        mock_ollama_chat.assert_called_once_with(
            model="test_ollama_model_stream",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Stream hello"}
            ],
            stream=True
        )

    @patch('termite.shared.call_llm.genai')
    def test_call_gemini_non_streaming(self, mock_genai):
        # Configure mock
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = MagicMock(text="gemini_test_response")
        mock_genai.GenerativeModel.return_value = mock_model_instance

        # Set environment variable
        os.environ["GEMINI_API_KEY"] = "test_gemini_api_key"

        # Call the function
        system_prompt = "Gemini system prompt"
        messages = [{"role": "user", "content": "Hello Gemini"}]
        response = call_gemini(system_prompt, messages, model="gemini-test-model", temperature=0.5)

        # Assertions
        self.assertEqual(response, "gemini_test_response")
        mock_genai.configure.assert_called_once_with(api_key="test_gemini_api_key")

        # Check if GenerationConfig was called correctly - may need to adjust if using genai.types.GenerationConfig
        try:
            mock_genai.GenerationConfig.assert_called_once_with(temperature=0.5)
        except AttributeError: # Fallback for older library versions if genai.types used
             mock_genai.types.GenerationConfig.assert_called_once_with(temperature=0.5)


        mock_genai.GenerativeModel.assert_called_once_with(
            model_name="gemini-test-model",
            generation_config=mock_genai.GenerationConfig.return_value if hasattr(mock_genai, 'GenerationConfig') else mock_genai.types.GenerationConfig.return_value,
            system_instruction=system_prompt
        )

        expected_mapped_messages = [{"role": "user", "parts": ["Hello Gemini"]}]
        mock_model_instance.generate_content.assert_called_once_with(expected_mapped_messages) # stream=False is default

    @patch('termite.shared.call_llm.genai')
    def test_call_gemini_streaming(self, mock_genai):
        # Configure mock for streaming
        stream_chunks = [
            MagicMock(text="gemini_chunk1 "),
            MagicMock(text="gemini_chunk2"),
            MagicMock(text=None), # Simulate empty text part
            MagicMock(text=" gemini_chunk3"),
        ]
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = iter(stream_chunks)
        mock_genai.GenerativeModel.return_value = mock_model_instance

        # Set environment variable
        os.environ["GEMINI_API_KEY"] = "test_gemini_api_key_stream"

        # Call the function
        system_prompt = "Gemini system stream prompt"
        messages = [{"role": "user", "content": "Stream Hello Gemini"}]
        response_generator = call_gemini(system_prompt, messages, stream=True, model="gemini-test-stream-model")

        # Assertions
        self.assertIsInstance(response_generator, Generator)
        responses = list(response_generator)
        self.assertEqual(responses, ["gemini_chunk1 ", "gemini_chunk2", " gemini_chunk3"])

        mock_genai.configure.assert_called_once_with(api_key="test_gemini_api_key_stream")
        mock_genai.GenerativeModel.assert_called_once_with(
            model_name="gemini-test-stream-model",
            generation_config=mock_genai.GenerationConfig.return_value if hasattr(mock_genai, 'GenerationConfig') else mock_genai.types.GenerationConfig.return_value,
            system_instruction=system_prompt
        )
        expected_mapped_messages = [{"role": "user", "parts": ["Stream Hello Gemini"]}]
        mock_model_instance.generate_content.assert_called_once_with(expected_mapped_messages, stream=True)

    def test_get_llm_provider_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
            self.assertEqual(get_llm_provider(), "openai")

    def test_get_llm_provider_anthropic(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True):
            self.assertEqual(get_llm_provider(), "anthropic")

    def test_get_llm_provider_gemini(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True):
            self.assertEqual(get_llm_provider(), "gemini")

    def test_get_llm_provider_ollama(self):
        with patch.dict(os.environ, {"OLLAMA_MODEL": "test_model"}, clear=True):
            self.assertEqual(get_llm_provider(), "ollama")

    def test_get_llm_provider_priority(self):
        # Test priority: OpenAI > Anthropic > Gemini > Ollama
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "openai_key",
            "ANTHROPIC_API_KEY": "anthropic_key",
            "GEMINI_API_KEY": "gemini_key",
            "OLLAMA_MODEL": "ollama_model"
        }, clear=True):
            self.assertEqual(get_llm_provider(), "openai")

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "anthropic_key",
            "GEMINI_API_KEY": "gemini_key",
            "OLLAMA_MODEL": "ollama_model"
        }, clear=True):
            self.assertEqual(get_llm_provider(), "anthropic")

        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "gemini_key",
            "OLLAMA_MODEL": "ollama_model"
        }, clear=True):
            self.assertEqual(get_llm_provider(), "gemini")

        with patch.dict(os.environ, {"OLLAMA_MODEL": "ollama_model"}, clear=True):
            self.assertEqual(get_llm_provider(), "ollama")

    def test_get_llm_provider_none_set(self):
        # No keys set, should raise ValueError
        # Clearing all relevant keys happens in setUp, but good to be explicit for this test's intent
        with patch.dict(os.environ, {}, clear=True):
             with self.assertRaisesRegex(ValueError, "No API key found for OpenAI, Anthropic, or Gemini. No Ollama model found either."):
                get_llm_provider()

if __name__ == '__main__':
    unittest.main()
