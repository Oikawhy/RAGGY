from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    database_url: str = Field(
        default="postgresql://consultant:consultant_dev_pw@localhost:5433/ai_consultant",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    llm_model: str = Field(default="gpt-5.4-mini", alias="LLM_MODEL")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    max_context_tokens: int = Field(default=3000, alias="MAX_CONTEXT_TOKENS", ge=1)
    neighbor_window: int = Field(default=1, alias="NEIGHBOR_WINDOW", ge=0)
    max_chunks: int = Field(default=8, alias="MAX_CHUNKS", ge=1)
    vector_top_k: int = Field(default=20, alias="VECTOR_TOP_K", ge=1)
    lexical_top_k: int = Field(default=20, alias="LEXICAL_TOP_K", ge=1)
    rrf_k: int = Field(default=60, alias="RRF_K", ge=1)
    trace_dir: str = Field(default="traces", alias="TRACE_DIR")
    trace_queue_size: int = Field(default=1000, alias="TRACE_QUEUE_SIZE", ge=1)
    trace_enqueue_timeout_ms: int = Field(default=50, alias="TRACE_ENQUEUE_TIMEOUT_MS", ge=0)
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS", ge=0)
    prompt_version: str = Field(default="p1", alias="PROMPT_VERSION")
    retrieval_config_hash: str = Field(default="default", alias="RETRIEVAL_CONFIG_HASH")
    embedding_config_hash: str = Field(default="default", alias="EMBEDDING_CONFIG_HASH")

    # Live runtime controls
    embedding_backend: str = Field(default="local_bge_m3", alias="EMBEDDING_BACKEND")
    embedding_service_url: str | None = Field(default=None, alias="EMBEDDING_SERVICE_URL")
    embedding_timeout_seconds: int = Field(default=10, alias="EMBEDDING_TIMEOUT_SECONDS", ge=1)
    embedding_circuit_breaker_threshold: int = Field(default=5, alias="EMBEDDING_CIRCUIT_BREAKER_THRESHOLD", ge=1)
    embedding_circuit_breaker_cooldown: int = Field(default=60, alias="EMBEDDING_CIRCUIT_BREAKER_COOLDOWN", ge=1)
    reranker_enabled: bool = Field(default=True, alias="RERANKER_ENABLED")
    reranker_backend: str = Field(default="lexical_overlap", alias="RERANKER_BACKEND")
    reranker_model: str = Field(
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        alias="RERANKER_MODEL",
    )
    llm_timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS", ge=1)
    llm_max_concurrency: int = Field(default=10, alias="LLM_MAX_CONCURRENCY", ge=1)
    embedding_max_concurrency: int = Field(default=5, alias="EMBEDDING_MAX_CONCURRENCY", ge=1)
    redis_control_plane_failure_mode: str = Field(default="fail_open", alias="REDIS_CONTROL_PLANE_FAILURE_MODE")
    trace_retention_days: int = Field(default=30, alias="TRACE_RETENTION_DAYS", ge=1)
