from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PipelineSection:
    """Collects ordered pipeline steps and global seed."""
    steps: List[str] = field(default_factory=list)
    seed: int = 0


@dataclass
class ModelConfig:
    """Specifies model id, device, and dtype."""
    name: str
    device: str = "auto"
    dtype: str = "float32"


@dataclass
class GenerationConfig:
    """Controls generation budget and sampling mode."""
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


@dataclass
class PromptConfig:
    """Holds the prompt template used for inference."""
    template: str
    strip_response: bool = True


@dataclass
class DataSplitConfig:
    """Describes dataset path, cache destination, and optional limit."""
    path: Path
    cache: Path
    limit: Optional[int] = None


@dataclass
class DataConfig:
    """Bundles train/eval dataset descriptions."""
    train: DataSplitConfig
    eval: DataSplitConfig


@dataclass
class GateConfig:
    """Configures the optional logistic gate."""
    enabled: bool = False
    threshold: float = 0.5
    lr: float = 1e-2
    epochs: int = 100
    state_path: Optional[Path] = None
    dataset_path: Optional[Path] = None


@dataclass
class IndexConfig:
    """Encodes latent index hyperparameters."""
    train_cache: Path
    index_path: Path
    metadata_path: Path
    k: int = 5
    normalize: bool = True
    temperature: float = 0.0
    base_prior: float = 0.0
    min_confidence: float = 0.0
    override_threshold: float = 0.5
    weighting: str = "softmax"
    gate: GateConfig = field(default_factory=GateConfig)


@dataclass
class OutputConfig:
    """Captures file locations for metrics and predictions."""
    metrics_path: Path
    flips_path: Path
    predictions_path: Optional[Path] = None


@dataclass
class Config:
    """Top-level configuration object for the kNoT pipeline."""
    pipeline: PipelineSection
    model: ModelConfig
    generation: GenerationConfig
    prompt: PromptConfig
    data: DataConfig
    index: IndexConfig
    output: OutputConfig


def _as_path(value: Any) -> Path:
    """Converts objects into Path instances."""
    return value if isinstance(value, Path) else Path(value)


def _load_pipeline(raw: Dict[str, Any]) -> PipelineSection:
    """Constructs the pipeline section from raw dict values."""
    steps = list(raw.get("steps", []))
    seed = int(raw.get("seed", 0))
    return PipelineSection(steps=steps, seed=seed)


def _load_model(raw: Dict[str, Any]) -> ModelConfig:
    """Parses model configuration fields."""
    return ModelConfig(
        name=str(raw["name"]),
        device=str(raw.get("device", "auto")),
        dtype=str(raw.get("dtype", "float32")),
    )


def _load_generation(raw: Dict[str, Any]) -> GenerationConfig:
    """Parses generation settings from the config."""
    return GenerationConfig(
        max_new_tokens=int(raw.get("max_new_tokens", 32)),
        temperature=float(raw.get("temperature", 0.0)),
        top_p=float(raw.get("top_p", 1.0)),
        do_sample=bool(raw.get("do_sample", False)),
    )


def _load_prompt(raw: Dict[str, Any]) -> PromptConfig:
    """Builds the prompt configuration."""
    return PromptConfig(
        template=str(raw["template"]),
        strip_response=bool(raw.get("strip_response", True)),
    )


def _load_data_split(raw: Dict[str, Any]) -> DataSplitConfig:
    """Creates a data split descriptor."""
    return DataSplitConfig(
        path=_as_path(raw["path"]),
        cache=_as_path(raw["cache"]),
        limit=(None if raw.get("limit") is None else int(raw["limit"])),
    )


def _load_data(raw: Dict[str, Any]) -> DataConfig:
    """Builds the aggregate data configuration."""
    return DataConfig(
        train=_load_data_split(raw["train"]),
        eval=_load_data_split(raw["eval"]),
    )


def _load_gate(raw: Dict[str, Any]) -> GateConfig:
    """Parses the gate block, falling back to defaults."""
    if not raw:
        return GateConfig()
    return GateConfig(
        enabled=bool(raw.get("enabled", False)),
        threshold=float(raw.get("threshold", 0.5)),
        lr=float(raw.get("lr", 1e-2)),
        epochs=int(raw.get("epochs", 100)),
        state_path=(
            None if raw.get("state_path") is None else _as_path(raw["state_path"])
        ),
        dataset_path=(
            None if raw.get("dataset_path") is None else _as_path(raw["dataset_path"])
        ),
    )


def _load_index(raw: Dict[str, Any], train_cache: Path) -> IndexConfig:
    """Assembles index parameters including gate wiring."""
    gate = _load_gate(raw.get("gate", {}))
    cache_path = _as_path(raw.get("train_cache", train_cache))
    index_path_raw = raw.get("index_path")
    if index_path_raw is None:
        index_path = cache_path.with_suffix(".faiss")
    else:
        index_path = _as_path(index_path_raw)
    metadata_path_raw = raw.get("metadata_path")
    if metadata_path_raw is None:
        metadata_path = Path(f"{index_path}.meta.json")
    else:
        metadata_path = _as_path(metadata_path_raw)
    return IndexConfig(
        train_cache=cache_path,
        index_path=index_path,
        metadata_path=metadata_path,
        k=int(raw.get("k", 5)),
        normalize=bool(raw.get("normalize", True)),
        temperature=float(raw.get("temperature", 0.0)),
        base_prior=float(raw.get("base_prior", 0.0)),
        min_confidence=float(raw.get("min_confidence", 0.0)),
        override_threshold=float(raw.get("override_threshold", 0.5)),
        weighting=str(raw.get("weighting", "softmax")),
        gate=gate,
    )


def _load_output(raw: Dict[str, Any]) -> OutputConfig:
    """Extracts output path configuration."""
    return OutputConfig(
        metrics_path=_as_path(raw["metrics_path"]),
        flips_path=_as_path(raw["flips_path"]),
        predictions_path=(
            None if raw.get("predictions_path") is None else _as_path(raw["predictions_path"])
        ),
    )


def load_config(raw: Dict[str, Any]) -> Config:
    """Builds the full configuration object from raw YAML."""
    pipeline = _load_pipeline(raw["pipeline"])
    model = _load_model(raw["model"])
    generation = _load_generation(raw.get("generation", {}))
    prompt = _load_prompt(raw["prompt"])
    data = _load_data(raw["data"])
    index = _load_index(raw.get("index", {}), data.train.cache)
    output = _load_output(raw["output"])
    return Config(
        pipeline=pipeline,
        model=model,
        generation=generation,
        prompt=prompt,
        data=data,
        index=index,
        output=output,
    )
