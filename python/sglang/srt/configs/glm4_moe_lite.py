from transformers import PretrainedConfig


class Glm4MoeLiteConfig(PretrainedConfig):
    """Minimal config registration for GLM4-MoE-Lite checkpoints.

    The checkpoint provides all required fields in `config.json`, so we only need
    to register `model_type` to let AutoConfig resolve it.
    """

    model_type = "glm4_moe_lite"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

