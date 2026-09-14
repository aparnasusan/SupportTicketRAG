"""Safe application errors; HTTP and CLI behavior belongs at the boundary."""


class ServiceError(Exception):
    code = "service_unavailable"
    message = "Resolution service is temporarily unavailable."

    def __init__(self):
        super().__init__(self.message)


class IndexUnavailable(ServiceError):
    code = "index_unavailable"
    message = "The ticket index is missing or empty. Run python ingest.py."


class RetrievalUnavailable(ServiceError):
    code = "retrieval_unavailable"
    message = "Ticket retrieval is unavailable. Check index access and the embedding model."


class OllamaUnavailable(ServiceError):
    code = "ollama_unavailable"
    message = "Ollama is unavailable. Check that it is running and the configured model is installed."


class InvalidModelResponse(ServiceError):
    code = "invalid_model_response"
    message = "The local model did not return a valid grounded response."
