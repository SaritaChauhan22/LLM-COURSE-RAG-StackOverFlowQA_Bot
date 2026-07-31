import unittest
from unittest.mock import patch

import rag


class RAGChatbotTests(unittest.TestCase):
    @patch("rag.AutoTokenizer")
    @patch("rag.AutoModelForSeq2SeqLM")
    def test_load_seq2seq_model_uses_supported_transformers_api(self, mock_model_cls, mock_tokenizer_cls):
        mock_tokenizer = mock_tokenizer_cls.from_pretrained.return_value
        mock_model = mock_model_cls.from_pretrained.return_value

        tokenizer, model = rag.load_seq2seq_model("google/flan-t5-small")

        self.assertIs(tokenizer, mock_tokenizer)
        self.assertIs(model, mock_model)
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("google/flan-t5-small")
        mock_model_cls.from_pretrained.assert_called_once_with("google/flan-t5-small")


if __name__ == "__main__":
    unittest.main()
