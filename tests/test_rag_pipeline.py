from unittest.mock import patch, MagicMock

from src.rag_pipeline import RAGPipeline


@patch("src.rag_pipeline.generate")
@patch("src.rag_pipeline.Retriever")
def test_ask_returns_answer_and_sources(mock_retriever_cls, mock_generate):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"text": "card was charged twice", "metadata": {"complaint_id": "1"}, "distance": 0.1},
    ]
    mock_retriever_cls.return_value = mock_retriever
    mock_generate.return_value = "Several complaints mention duplicate charges."

    pipeline = RAGPipeline()
    result = pipeline.ask("Why are people unhappy with credit cards?")

    assert result["answer"] == "Several complaints mention duplicate charges."
    assert len(result["sources"]) == 1
    mock_generate.assert_called_once()


@patch("src.rag_pipeline.Retriever")
def test_ask_handles_no_retrieved_chunks(mock_retriever_cls):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_retriever_cls.return_value = mock_retriever

    pipeline = RAGPipeline()
    result = pipeline.ask("Some question with no matches")

    assert "don't have enough information" in result["answer"]
    assert result["sources"] == []
