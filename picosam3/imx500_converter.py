import itertools
import os

import model_compression_toolkit as mct
import onnx
import torch
from torch.utils.data import DataLoader

from model_compression.dataset import PicoSAMDataset, custom_collate
from model_compression.model import PicoSAM3

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_CKPT            = os.path.join(REPO_ROOT, "checkpoints", "PicoSAM3_SAM3_student_epoch10.pt")
ONNX_OUT              = os.path.join(REPO_ROOT, "checkpoints", "PicoSAM3_quantized.onnx")
PRUNED_CKPT           = os.path.join(REPO_ROOT, "checkpoints", "PicoSAM3_pruned.pt")
IMAGE_DIR             = os.path.join(REPO_ROOT, "dataset", "val2017")
ANN_FILE              = os.path.join(REPO_ROOT, "dataset", "annotations", "instances_val2017.json")
CACHE_DIR             = os.path.join(REPO_ROOT, "dataset", "teacher_sam3_logits")

IMAGE_SIZE            = 96
BATCH_SIZE            = 8
N_CALIBRATION_BATCHES = 10  # Increase for better PTQ calibration

# ── Toggle flags ────────────────────────────────────────────────────────────
ENABLE_PRUNING         = False  # Structured pruning before quantization
ENABLE_MIXED_PRECISION = False  # Mixed bit-width PTQ (requires GPU)
ENABLE_BIAS_CORRECTION = False  # Compensates quantization-induced bias shifts
# ────────────────────────────────────────────────────────────────────────────


def get_representative_dataset(loader):
    loader_iter = itertools.cycle(loader)

    def representative_dataset_gen():
        for _ in range(N_CALIBRATION_BATCHES):
            images, *_ = next(loader_iter)
            yield [images]

    return representative_dataset_gen


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset = PicoSAMDataset(IMAGE_DIR, ANN_FILE, IMAGE_SIZE, CACHE_DIR, require_cache=False)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         collate_fn=custom_collate, num_workers=4)
    representative_dataset_gen = get_representative_dataset(loader)

    # ── Load model ──────────────────────────────────────────────────────────
    model = PicoSAM3(in_channels=3)
    model.load_state_dict(torch.load(MODEL_CKPT, map_location="cpu"))
    model.eval()
    print(f"Loaded checkpoint: {MODEL_CKPT}")

    # ── Pruning (optional) ──────────────────────────────────────────────────
    # Removes entire channels that contribute least to the output.
    # Run before quantization — a smaller model is easier to quantize accurately.
    # Lower target_sparsity = less aggressive pruning (0.1 = remove ~10% of channels).
    # After pruning, the model is saved and reloaded so PTQ sees the pruned weights.
    #
    # if ENABLE_PRUNING:
    #     pruning_config = mct.pruning.PruningConfig(
    #         num_score_approximations=32,         # Higher = more accurate sensitivity scoring, slower
    #     )
    #     resource_utilization = mct.core.ResourceUtilization(
    #         weights_memory=0.75 * sum(                # Target 75% of original weight memory
    #             p.numel() * p.element_size() for p in model.parameters()
    #         )
    #     )
    #     pruned_model, pruning_info = mct.pruning.pytorch_pruning_for_structured_pruning(
    #         model=model,
    #         target_resource_utilization=resource_utilization,
    #         representative_data_gen=representative_dataset_gen,
    #         pruning_config=pruning_config,
    #     )
    #     torch.save(pruned_model.state_dict(), PRUNED_CKPT)
    #     print(f"Pruned model saved to: {PRUNED_CKPT}")
    #     print(pruning_info)
    #     model = pruned_model
    #
    # To enable: set ENABLE_PRUNING = True above, then uncomment this block.

    if ENABLE_PRUNING:
        raise NotImplementedError("Uncomment the pruning block above to use pruning.")

    # ── PTQ ─────────────────────────────────────────────────────────────────
    target_platform_cap = mct.get_target_platform_capabilities(
        "pytorch", "imx500", target_platform_version="v3"
    )

    # ── Mixed precision (optional) ──────────────────────────────────────────
    # Assigns different bit-widths per layer based on sensitivity to quantization.
    # Sensitive layers (e.g. first/last conv) keep more bits; insensitive layers
    # are aggressively quantized. Requires GPU and takes longer than uniform PTQ.
    # Adjust candidates to control which bit-widths are considered per layer.
    #
    # if ENABLE_MIXED_PRECISION:
    #     mixed_precision_config = mct.core.MixedPrecisionQuantizationConfig(
    #         num_of_images=32,             # Images used for sensitivity analysis
    #         use_hessian_based_scores=True # Better sensitivity scoring, slower
    #     )
    #     core_config = mct.core.CoreConfig(
    #         mixed_precision_config=mixed_precision_config,
    #     )
    #     resource_utilization = mct.core.ResourceUtilization(
    #         weights_memory=0.75 * sum(        # Target 75% of uniform 8-bit memory
    #             p.numel() for p in model.parameters()
    #         )
    #     )
    # else:
    #     core_config          = mct.core.CoreConfig()
    #     resource_utilization = None
    #
    # To enable: set ENABLE_MIXED_PRECISION = True above, then uncomment this
    # block and replace core_config/resource_utilization in the PTQ call below.

    if ENABLE_MIXED_PRECISION:
        raise NotImplementedError("Uncomment the mixed precision block above to use it.")

    core_config          = mct.core.CoreConfig()
    resource_utilization = None

    # ── Bias correction (optional) ──────────────────────────────────────────
    # Corrects the statistical shift in layer outputs caused by quantization.
    # Free accuracy gains in most cases — low risk to enable.
    #
    # if ENABLE_BIAS_CORRECTION:
    #     core_config = mct.core.CoreConfig(
    #         quantization_config=mct.core.QuantizationConfig(
    #             weights_bias_correction=True,
    #         )
    #     )
    #
    # To enable: set ENABLE_BIAS_CORRECTION = True above, then uncomment this
    # block and remove the core_config = mct.core.CoreConfig() line above.

    if ENABLE_BIAS_CORRECTION:
        raise NotImplementedError("Uncomment the bias correction block above to use it.")

    print("Running PTQ...")
    quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
        in_module=model,
        representative_data_gen=representative_dataset_gen,
        target_platform_capabilities=target_platform_cap,
        core_config=core_config,
        # resource_utilization=resource_utilization,  # Uncomment when using mixed precision
    )
    print("PTQ complete.")
    print(quantization_info)

    # ── Export to ONNX ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(ONNX_OUT), exist_ok=True)
    mct.exporter.pytorch_export_model(
        model=quantized_model,
        save_model_path=ONNX_OUT,
        repr_dataset=representative_dataset_gen,
    )
    print(f"Exported quantized ONNX model to: {ONNX_OUT}")

    onnx.checker.check_model(ONNX_OUT)
    print("ONNX model check passed.")


if __name__ == "__main__":
    main()
