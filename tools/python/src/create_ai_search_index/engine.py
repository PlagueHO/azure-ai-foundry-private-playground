"""
Azure AI Search Index Creation Engine.

This module provides functionality to create and manage Azure AI Search indexes,
including document processing, embedding generation, and search index management.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    CorsOptions,
    HnswAlgorithmConfiguration,
    IndexingParameters,
    IndexingParametersConfiguration,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerSkill,
    SearchIndexerSkillset,
    SplitSkill,
    VectorSearch,
    VectorSearchProfile,
)

if TYPE_CHECKING:
    pass

__all__: list[str] = ["CreateAISearchIndex", "CreateAISearchIndexConfig"]

@dataclass
class CreateAISearchIndexConfig:
    """
    Configuration dataclass for the CreateAISearchIndex pipeline.

    Holds all user-supplied and derived settings required to build and run
    the Azure AI Search indexing pipeline.
    """

    search_service: str
    index_name: str
    storage_container: str
    storage_account: str | None = None
    storage_account_key: str | None = None
    storage_account_connection_string: str | None = None
    embedding_model: str | None = "text-embedding-ada-002"
    embedding_deployment: str | None = None
    embedding_dimension: int = 1536
    azure_openai_endpoint: str | None = None
    delete_existing: bool = False

    @property
    def search_endpoint(self) -> str:
        """Return the full endpoint URL for the Azure AI Search service."""
        return f"https://{self.search_service}.search.windows.net"

class CreateAISearchIndex:
    """
    Synchronous builder that assembles an Azure AI Search pipeline and triggers
    the first index run.

    This class encapsulates the full workflow for creating or updating an
    Azure AI Search index, data source, skillset, and indexer, and then
    running the indexer to populate the index from a blob container.
    """

    def __init__(
        self, cfg: CreateAISearchIndexConfig, *, log_level: str | int = "INFO"
    ) -> None:
        """
        Initialize the CreateAISearchIndex orchestrator.

        Args:
            cfg (CreateAISearchIndexConfig): Configuration dataclass with
                pipeline settings.
            log_level (str | int, optional): Logging verbosity. Defaults to "INFO".
        """
        self.cfg = cfg
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
            level=log_level,
        )
        self.logger = logging.getLogger("create-ai-search-index")

        # Credential: prefer DefaultAzureCredential, fallback to API key
        self.credential: AzureKeyCredential | DefaultAzureCredential
        api_key = None  # Optionally load from env
        if api_key:
            self.credential = AzureKeyCredential(api_key)
        else:
            self.credential = DefaultAzureCredential(
                exclude_interactive_browser_credential=False
            )

        self.index_client = SearchIndexClient(
            self.cfg.search_endpoint, self.credential
        )
        self.indexer_client = SearchIndexerClient(
            self.cfg.search_endpoint, self.credential
        )

    def run(self) -> None:
        """
        Execute the full pipeline: optionally tear down, then create or update
        index schema, data source, skillset, indexer, and finally run the indexer.
        """
        if self.cfg.delete_existing:
            self._teardown_pipeline()
        self._ensure_index_schema()
        self._ensure_data_source()
        self._ensure_skillset()
        self._ensure_indexer()
        self._run_indexer()

    def _ensure_index_schema(self) -> None:
        """
        Create or update the Azure AI Search index schema.

        Raises:
            Exception: If the index creation or update fails.
        """
        self.logger.info("Ensuring index schema '%s'...", self.cfg.index_name)

        fields = [
            SearchField(name="parent_id", type=SearchFieldDataType.String),
            SearchField(name="title", type=SearchFieldDataType.String),
            SearchField(
                name="locations",
                type="Collection(Edm.String)",
                filterable=True,
            ),
            SearchField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                key=True,
                sortable=True,
                filterable=True,
                facetable=True,
                analyzer_name="keyword",
            ),
            SearchField(
                name="chunk",
                type=SearchFieldDataType.String,
                sortable=False,
                filterable=False,
                facetable=False,
            ),
            SearchField(
                name="text_vector",
                type="Collection(Edm.Single)",
                vector_search_dimensions=self.cfg.embedding_dimension,
                vector_search_profile_name="myHnswProfile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="myHnsw"),
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                    vectorizer_name="myOpenAI",
                )
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name="myOpenAI",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=self.cfg.azure_openai_endpoint,
                        deployment_name=self.cfg.embedding_deployment,
                        model_name=self.cfg.embedding_model
                    ),
                ),
            ],
        )

        cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=300)

        index = SearchIndex(
            name=self.cfg.index_name,
            fields=fields,
            vector_search=vector_search,
            cors_options=cors_options
        )

        try:
            self.index_client.create_or_update_index(index)
            self.logger.info("Index schema ensured.")
        except Exception as ex:
            self.logger.error("Failed to create/update index: %s", ex)
            raise

    def _get_storage_account_connection_string(self) -> str:
        """
        Assemble and return the storage account connection string.

        Returns:
            str: The storage account connection string.

        Raises:
            ValueError: If required storage account information is missing.
        """
        if self.cfg.storage_account_connection_string:
            return self.cfg.storage_account_connection_string
        if not self.cfg.storage_account or not self.cfg.storage_account_key:
            raise ValueError(
                "Both storage_account and storage_account_key are required if "
                "connection string is not provided."
            )
        return (
            f"DefaultEndpointsProtocol=https;AccountName={self.cfg.storage_account};"
            f"AccountKey={self.cfg.storage_account_key};"
            f"EndpointSuffix=core.windows.net"
        )

    def _ensure_data_source(self) -> None:
        """
        Create or update the data source connection for the indexer.

        Raises:
            Exception: If the data source creation or update fails.
        """
        self.logger.info("Ensuring data source connection...")
        ds_name = f"{self.cfg.index_name}-blob-ds"
        connection_string = self._get_storage_account_connection_string()
        ds = SearchIndexerDataSourceConnection(
            name=ds_name,
            type="azureblob",
            connection_string=connection_string,
            container=SearchIndexerDataContainer(name=self.cfg.storage_container),
            description="Blob container for RAG documents"
        )
        try:
            self.indexer_client.create_or_update_data_source_connection(ds)
            self.logger.info("Data source ensured.")
        except Exception as ex:
            self.logger.error("Failed to create/update data source: %s", ex)
            raise

    def _ensure_skillset(self) -> None:
        """
        Create or update the skillset for chunking and embedding.

        Raises:
            Exception: If the skillset creation or update fails.
        """
        self.logger.info("Ensuring skillset...")
        skillset_name = f"{self.cfg.index_name}-skillset"
        skills: list[SearchIndexerSkill] = [
            SplitSkill(
                name="splitSkill",
                description="Split content into chunks",
                context="/document/content",
                text_split_mode="pages",
                maximum_page_length=2000,
                inputs=[
                    InputFieldMappingEntry(name="text", source="/document/content")
                ],
                outputs=[
                    OutputFieldMappingEntry(name="pages", target_name="chunks")
                ],
            ),
            # TODO: Add a supported embedding skill here, such as a CustomSkill
            # or CognitiveSkill, if required.
            # AzureOpenAIVectorizer is not a valid SearchIndexerSkill and
            # cannot be used here.
        ]
        skillset = SearchIndexerSkillset(
            name=skillset_name,
            skills=skills,
            description="Chunking and embedding skillset"
        )
        try:
            self.indexer_client.create_or_update_skillset(skillset)
            self.logger.info("Skillset ensured.")
        except Exception as ex:
            self.logger.error("Failed to create/update skillset: %s", ex)
            raise

    def _ensure_indexer(self) -> None:
        """
        Create or update the indexer that connects the data source, skillset, and index.

        Raises:
            Exception: If the indexer creation or update fails.
        """
        self.logger.info("Ensuring indexer...")
        indexer_name = f"{self.cfg.index_name}-indexer"
        ds_name = f"{self.cfg.index_name}-blob-ds"
        skillset_name = f"{self.cfg.index_name}-skillset"
        indexer = SearchIndexer(
            name=indexer_name,
            data_source_name=ds_name,
            target_index_name=self.cfg.index_name,
            skillset_name=skillset_name,
            description="Indexer for RAG pipeline",
            parameters=IndexingParameters(
                configuration=IndexingParametersConfiguration(
                    parsing_mode="default"
                )
            )
        )
        try:
            self.indexer_client.create_or_update_indexer(indexer)
            self.logger.info("Indexer ensured.")
        except Exception as ex:
            self.logger.error("Failed to create/update indexer: %s", ex)
            raise

    def _run_indexer(self) -> None:
        """
        Run the indexer and poll for completion, logging progress and errors.

        Raises:
            RuntimeError: If the indexer fails or does not complete in time.
            Exception: If running the indexer fails.
        """
        indexer_name = f"{self.cfg.index_name}-indexer"
        self.logger.info("Running indexer '%s'...", indexer_name)
        try:
            self.indexer_client.run_indexer(indexer_name)
            # Poll for status (simple loop)
            for _ in range(60):
                status = self.indexer_client.get_indexer_status(indexer_name)
                if status.status == "inProgress":
                    self.logger.info("Indexer running...")
                    time.sleep(10)
                elif status.status == "success":
                    self.logger.info("Indexer completed successfully.")
                    return
                else:
                    error_message = getattr(
                        status.last_result, "error_message", "Unknown error"
                    )
                    self.logger.error("Indexer failed: %s", error_message)
                    raise RuntimeError(f"Indexer failed: {error_message}")
            self.logger.warning("Indexer did not complete within expected time.")
        except Exception as ex:
            self.logger.error("Failed to run indexer: %s", ex)
            raise

    def _teardown_pipeline(self) -> None:
        """
        Delete indexer, skillset, data source, and index if they exist.

        This is called if --delete-existing is set to ensure a clean pipeline.
        """
        self.logger.info("Tearing down existing pipeline resources...")
        indexer_name = f"{self.cfg.index_name}-indexer"
        skillset_name = f"{self.cfg.index_name}-skillset"
        ds_name = f"{self.cfg.index_name}-blob-ds"
        # Delete indexer
        try:
            self.indexer_client.delete_indexer(indexer_name)
        except Exception:
            pass
        # Delete skillset
        try:
            self.indexer_client.delete_skillset(skillset_name)
        except Exception:
            pass
        # Delete data source
        try:
            self.indexer_client.delete_data_source_connection(ds_name)
        except Exception:
            pass
        # Delete index
        try:
            self.index_client.delete_index(self.cfg.index_name)
        except Exception:
            pass
        self.logger.info("Pipeline teardown complete.")
