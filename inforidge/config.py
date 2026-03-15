import pydantic


class InforidgeConfig(pydantic.BaseModel):
    enabled: bool = False
    max_batches: int = 1
    max_tokens: int = 256
    sample_mode: str = "last_token"
    target_embedding_mode: str = "lookup"  # one of: lookup, second_pass
    normalize_vectors: bool = True
    kernel_sigma: float = 1.0
    sample_seed: int = 0
    include_H: bool = True
    include_L: bool = True
    log_wandb: bool = True
    save_csv: bool = True


class InforidgeActMIConfig(pydantic.BaseModel):
    enabled: bool = False
    max_batches: int = 1
    log_wandb: bool = True
    save_csv: bool = True
