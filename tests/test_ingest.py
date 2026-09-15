"""Validate ingestion preflight and reset semantics without embedding downloads."""

import unittest
from unittest.mock import Mock, patch

import pandas as pd
from chromadb.errors import NotFoundError

from support_ticket_rag import ingest

ROW = {
    "ticket_id": "SR001",
    "product": "Identity",
    "issue": "Cannot sign in",
    "resolution": "Clear session",
}


class IngestionTests(unittest.TestCase):
    def test_rejects_invalid_rows(self):
        invalid = [
            pd.DataFrame(columns=ROW),
            pd.DataFrame([ROW]).drop(columns="issue"),
            pd.DataFrame([ROW, ROW]),
            pd.DataFrame([ROW | {"ticket_id": "bad"}]),
        ]
        for field in ROW:
            for value in ["", "   ", None, float("nan"), 123]:
                invalid.append(pd.DataFrame([ROW | {field: value}]))
        for frame in invalid:
            with self.subTest(frame=frame.to_dict()):
                with self.assertRaises(ValueError):
                    ingest.validate_tickets(frame)

    def test_valid_text_and_metadata_preserve_embedding_input(self):
        frame = pd.DataFrame([ROW])
        ingest.validate_tickets(frame)
        document = ingest.ticket_to_document(frame.iloc[0])
        self.assertEqual(document.text, "Product: Identity\nIssue: Cannot sign in")
        self.assertEqual(document.metadata["resolution"], "Clear session")

    def test_invalid_data_never_opens_storage(self):
        with (
            patch.object(ingest.pd, "read_csv", return_value=pd.DataFrame([ROW | {"issue": ""}])),
            patch.object(ingest.chromadb, "PersistentClient") as client,
        ):
            with self.assertRaises(ValueError):
                ingest.build_index(reset=True)
            client.assert_not_called()

    def test_model_failure_preserves_existing_collection(self):
        with (
            patch.object(ingest.pd, "read_csv", return_value=pd.DataFrame([ROW])),
            patch.object(ingest.chromadb, "PersistentClient") as client,
            patch.object(ingest, "HuggingFaceEmbedding", side_effect=OSError("model unavailable")),
        ):
            with self.assertRaises(OSError):
                ingest.build_index(reset=True)
            client.return_value.delete_collection.assert_not_called()
            client.return_value.get_or_create_collection.assert_not_called()

    def test_reset_handles_existing_and_missing_collection(self):
        for missing in [False, True]:
            with (
                self.subTest(missing=missing),
                patch.object(ingest.pd, "read_csv", return_value=pd.DataFrame([ROW])),
                patch.object(ingest.chromadb, "PersistentClient") as client,
                patch.object(ingest, "HuggingFaceEmbedding") as embed,
                patch.object(ingest, "ChromaVectorStore"),
                patch.object(ingest, "StorageContext"),
                patch.object(ingest, "VectorStoreIndex") as index,
            ):
                events = Mock()
                events.attach_mock(embed, "model")
                events.attach_mock(client.return_value.delete_collection, "delete")
                if missing:
                    client.return_value.delete_collection.side_effect = NotFoundError("missing")
                ingest.build_index(reset=True)
                self.assertEqual([call[0] for call in events.mock_calls][:2], ["model", "delete"])
                index.from_documents.assert_called_once()
                client.return_value.get_or_create_collection.assert_called_once()

    def test_nonempty_index_is_left_intact_without_model_loading(self):
        with (
            patch.object(ingest.pd, "read_csv", return_value=pd.DataFrame([ROW])),
            patch.object(ingest.chromadb, "PersistentClient") as client,
            patch.object(ingest, "HuggingFaceEmbedding") as embed,
        ):
            client.return_value.get_collection.return_value.count.return_value = 30
            ingest.build_index(reset=False)
            client.return_value.delete_collection.assert_not_called()
            embed.assert_not_called()

    def test_unexpected_delete_failure_is_not_swallowed(self):
        with (
            patch.object(ingest.pd, "read_csv", return_value=pd.DataFrame([ROW])),
            patch.object(ingest.chromadb, "PersistentClient") as client,
            patch.object(ingest, "HuggingFaceEmbedding"),
        ):
            client.return_value.delete_collection.side_effect = RuntimeError("storage error")
            with self.assertRaises(RuntimeError):
                ingest.build_index(reset=True)
            client.return_value.get_or_create_collection.assert_not_called()
