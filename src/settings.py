from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


class YamlConfig(BaseSettings):
    def __init__(self, yaml_path: str) -> None:
        if not Path(yaml_path).exists():
            raise FileNotFoundError(yaml_path)

        super().__init__(yaml_path=yaml_path)

    model_config = SettingsConfigDict(frozen=True, extra="ignore")

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs) -> tuple:
        yaml_path = kwargs["init_settings"].init_kwargs["yaml_path"]
        return (YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path),)


class Hyperparams(YamlConfig):
    batch_size: int | None = None
    samples_per_class: int | None = None
    epochs: int | None = None

    learning_rate: float | None = None
    lr_start_factor: float | None = None
    warmup_steps: int | None = None
    eta_min: float | None = None

    temperature: float | None = None
    weight_decay: float | None = None
    projection_dimension: int | None = None

    freeze_layers: int | None = None
    freeze_embeddings: bool | None = None
    num_classes: int | None = None


class Settings(YamlConfig):
    tracking_uri: str | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    train_dataset_path: str | None = None
    valid_dataset_path: str | None = None
    test_dataset_path: str | None = None

    model_directory: str | None = None
    pretrained_model_directory: str | None = None
    device: str  = "cpu"
