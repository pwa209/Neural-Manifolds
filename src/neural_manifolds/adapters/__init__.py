"""Dataset-specific, label-separated cohort and contrast adapters."""

from .datasets import (
    ADAPTERS,
    CogitateMEEGAdapter,
    DreamSerialAwakeningsAdapter,
    FigshareDoCRestingAdapter,
    MendeleyDoCPSGAdapter,
    PropofolFMRIAdapter,
    PropofolTMSEEGAdapter,
    PsiConnectAdapter,
    SomatosensoryReportTaskAdapter,
    TactileDetectionAdapter,
    get_adapter,
)
from .models import (
    AdapterError,
    AnalysisUnit,
    EncoderInput,
    SchemaError,
    SignalSelector,
    UnresolvedMetadataError,
    assert_label_free_encoder_payload,
    encoding_view,
)

__all__ = [
    "ADAPTERS",
    "AdapterError",
    "AnalysisUnit",
    "CogitateMEEGAdapter",
    "DreamSerialAwakeningsAdapter",
    "EncoderInput",
    "FigshareDoCRestingAdapter",
    "MendeleyDoCPSGAdapter",
    "PropofolFMRIAdapter",
    "PropofolTMSEEGAdapter",
    "PsiConnectAdapter",
    "SchemaError",
    "SignalSelector",
    "SomatosensoryReportTaskAdapter",
    "TactileDetectionAdapter",
    "UnresolvedMetadataError",
    "assert_label_free_encoder_payload",
    "encoding_view",
    "get_adapter",
]
