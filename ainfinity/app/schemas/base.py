"""Base enums and constants for schemas"""

from enum import Enum


class JobStatus(str, Enum):
    """Status of a training job"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class InfraProvider(str, Enum):
    """Infrastructure provider options"""

    VAST = "vast"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class AttnImpl(str, Enum):
    """Attention implementation options"""

    FLASH_ATTENTION = "flash_attention_2"
    SPDA = "sdpa"
    EAGER = "eager"


class SchedulerType(str, Enum):
    """Types of learning rate schedulers"""

    LINEAR = "linear"
    COSINE = "cosine"
    CONSTANT = "constant"


class EvalStrategy(str, Enum):
    """Evaluation strategy options"""

    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class SaveStrategy(str, Enum):
    """Model saving strategy options"""

    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class DatalogSchema(str, Enum):
    """Dataset schema types"""

    CONVERSATION = "conversation"
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    PREFERENCE_RANKING = "preference_ranking"
    ENTITY_RELATIONSHIP_EXTRACTION = "entity_relationship_extraction"


class DataSource(str, Enum):
    """Data source types"""

    DATALOG = "datalog"
    HUGGINGFACE = "huggingface"


class PaddingStrategy(str, Enum):
    """Padding strategy options for tokenization"""

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
