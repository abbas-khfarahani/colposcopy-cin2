"""Pre-specified experiment ladder (B0–B12). Architecture search frozen after B11."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    name: str
    kind: str
    description: str
    backbone: str = "resnet50"
    model_type: str = "image"
    center_prior: bool = True
    pooling: str = "max"
    loss: str = "bce"
    patient_weights: bool = True
    balanced_sampler: bool = False
    ssl: str | None = None
    pooling_source: str | None = None
    stack_source: str | None = None
    confirm: bool = False


LADDER: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        id="B0",
        name="clinical_only",
        kind="clinical",
        description="Logistic regression on age + HPV + cytology (no images).",
        confirm=True,
    ),
    ExperimentSpec(
        id="B1",
        name="resnet50_gap_max",
        kind="image",
        description="ResNet50 global-average-pool @224, max person pooling, ImageNet.",
        backbone="resnet50",
        model_type="image",
        pooling="max",
        confirm=True,
    ),
    ExperimentSpec(
        id="B2",
        name="convnext_tiny_max",
        kind="image",
        description="ConvNeXt-Tiny @224, max pooling, ImageNet architecture control.",
        backbone="convnext_tiny",
        model_type="image",
        pooling="max",
    ),
    ExperimentSpec(
        id="B3",
        name="resnet50_attn_noprior",
        kind="image",
        description="ResNet50 spatial attention, center prior OFF.",
        backbone="resnet50",
        model_type="image_attn",
        center_prior=False,
        pooling="max",
        confirm=True,
    ),
    ExperimentSpec(
        id="B4",
        name="resnet50_attn_center",
        kind="image",
        description="ResNet50 spatial attention, center prior ON (σ=0.45).",
        backbone="resnet50",
        model_type="image_attn",
        center_prior=True,
        pooling="max",
        confirm=True,
    ),
    ExperimentSpec(
        id="B5",
        name="pooling_ablation",
        kind="pooling",
        description="Re-aggregate B1–B4 image logits with max / mean / LSE.",
        pooling_source="auto",
    ),
    ExperimentSpec(
        id="B6",
        name="resnet50_attn_mil",
        kind="mil",
        description="Best spatial-attention encoder + gated patient MIL.",
        backbone="resnet50",
        model_type="image_attn",
        center_prior=True,
        confirm=True,
    ),
    ExperimentSpec(
        id="B7",
        name="mil_balanced_sampler",
        kind="mil",
        description="Attention + patient MIL with patient-balanced bag sampling.",
        backbone="resnet50",
        model_type="image_attn",
        center_prior=True,
        balanced_sampler=True,
    ),
    ExperimentSpec(
        id="B8",
        name="mil_focal",
        kind="mil",
        description="Attention + patient MIL with focal γ=2.",
        backbone="resnet50",
        model_type="image_attn",
        center_prior=True,
        loss="focal",
    ),
    ExperimentSpec(
        id="B9",
        name="ssl_simclr_mil",
        kind="ssl_mil",
        description="Fold-safe SimCLR pretrain → attention MIL fine-tune.",
        backbone="resnet50",
        model_type="image_attn",
        center_prior=True,
        ssl="simclr",
    ),
    ExperimentSpec(
        id="B10",
        name="ssl_micle_mil",
        kind="ssl_mil",
        description="Fold-safe patient-aware (MICLe-like) SSL → attention MIL fine-tune.",
        backbone="resnet50",
        model_type="image_attn",
        center_prior=True,
        ssl="micle",
    ),
    ExperimentSpec(
        id="B11",
        name="best_image_clinical_stack",
        kind="stack",
        description="Late L2 logistic stack: image logit + age/HPV/cytology.",
        stack_source="auto",
    ),
    ExperimentSpec(
        id="B12",
        name="nested_frozen_select",
        kind="nested",
        description="Nested selection among frozen B0/B1/B4/B11 using inner-val only; outer test is never used to pick the model.",
    ),
)


def spec_by_id(experiment_id: str) -> ExperimentSpec:
    for spec in LADDER:
        if spec.id == experiment_id:
            return spec
    raise KeyError(experiment_id)
