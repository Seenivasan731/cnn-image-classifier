"""CIFAR-10 Custom Photo Prediction Web Application.

Applied Robotics and AI Project (ARAIP)
University of Hertfordshire / PSB Academy
Student: Seenivasan M

This module provides a Gradio-based web interface allowing users to upload
their own custom photos, resize and normalize them to CIFAR-10 dimensions (32x32),
and classify them into one of the 10 categories using the trained CNN model.
"""

import os
import sys
import argparse
from typing import Tuple, Dict, Optional, Any
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import gradio as gr

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import (
    CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
)
from src.train import (
    BaselineCNN,
    load_checkpoint,
    get_device,
    DEFAULT_MODEL_PATH,
)

# Global model and device references
_MODEL: Optional[BaselineCNN] = None
_DEVICE: Optional[torch.device] = None
_MODEL_STATUS: str = "Uninitialized"


def get_inference_transform() -> transforms.Compose:
    """Return image transform pipeline to convert custom user photos to CIFAR-10 format."""
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def load_app_model(
    model_path: str = DEFAULT_MODEL_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[BaselineCNN, torch.device, str]:
    """Load model weights from checkpoint or initialize architecture."""
    if device is None:
        device = get_device()

    model = BaselineCNN(num_classes=len(CLASSES)).to(device)
    model.eval()

    if os.path.exists(model_path):
        try:
            model, checkpoint = load_checkpoint(model, model_path, device=device)
            val_acc = checkpoint.get("val_acc", "N/A") if isinstance(checkpoint, dict) else "N/A"
            status = f"Loaded checkpoint '{os.path.basename(model_path)}' (Val Acc: {val_acc})"
        except Exception as exc:
            status = f"Failed to load checkpoint ({exc}); using initialized weights"
    else:
        status = f"Checkpoint not found at '{os.path.basename(model_path)}'; using initialized weights"

    return model, device, status


def predict_custom_image(image: Optional[Image.Image]) -> Tuple[str, Dict[str, float]]:
    """Predict class labels and confidence scores for a custom uploaded photo.

    Args:
        image: PIL Image uploaded by user through Gradio.

    Returns:
        Tuple of (formatted top prediction string, dictionary of class probabilities).
    """
    global _MODEL, _DEVICE
    if image is None:
        return "Please upload an image.", {}

    if _MODEL is None or _DEVICE is None:
        _MODEL, _DEVICE, _ = load_app_model()

    # Convert to RGB (handles RGBA, grayscale, palette modes)
    image_rgb = image.convert("RGB")

    # Apply preprocessing (resize 32x32, ToTensor, Normalize)
    transform = get_inference_transform()
    tensor = transform(image_rgb).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        logits = _MODEL(tensor)
        probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()

    # Build confidence dictionary for Gradio Label
    confidences = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}

    # Determine top predicted class
    top_idx = int(probabilities.argmax())
    top_class = CLASSES[top_idx]
    top_prob = probabilities[top_idx] * 100.0

    summary_text = f"**{top_class.upper()}** ({top_prob:.1f}% confidence)"
    return summary_text, confidences


def create_app(model_path: str = DEFAULT_MODEL_PATH) -> gr.Blocks:
    """Build and configure the Gradio web interface."""
    global _MODEL, _DEVICE, _MODEL_STATUS
    _MODEL, _DEVICE, _MODEL_STATUS = load_app_model(model_path=model_path)

    description_md = f"""
    ### Predict on Your Own Custom Photos
    Upload any picture from your computer or phone (e.g. an airplane, car, cat, dog, bird, etc.).
    The image is automatically converted to RGB, resized to 32×32 pixels, normalized,
    and classified by the PyTorch CNN model.

    **CIFAR-10 Categories:** {', '.join(CLASSES)}
    """

    with gr.Blocks(title="CIFAR-10 Image Classifier") as demo:
        gr.Markdown("# 🤖 CNN Image Classifier (CIFAR-10)")
        gr.Markdown(description_md)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Custom Photo")
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Classify Image", variant="primary")
                status_box = gr.Textbox(
                    value=_MODEL_STATUS,
                    label="Model Status",
                    interactive=False,
                )

            with gr.Column(scale=1):
                top_prediction = gr.Markdown("### Top Prediction: Waiting for input...")
                label_output = gr.Label(num_top_classes=5, label="Top 5 Confidence Scores")

        submit_btn.click(
            fn=predict_custom_image,
            inputs=[input_image],
            outputs=[top_prediction, label_output],
        )

        clear_btn.click(
            fn=lambda: (None, "### Top Prediction: Waiting for input...", {}),
            inputs=[],
            outputs=[input_image, top_prediction, label_output],
        )

        gr.Markdown("---")
        gr.Markdown(
            "**Course:** Applied Robotics and AI Project (ARAIP) | "
            "**Student:** Seenivasan M | "
            "**Institution:** University of Hertfordshire / PSB Academy"
        )

    return demo


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the web app."""
    parser = argparse.ArgumentParser(description="Run Gradio Web App for custom CIFAR-10 image predictions")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to .pth checkpoint file")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--test-only", action="store_true", help="Test prediction pipeline on synthetic image and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 65)
    print("CIFAR-10 Custom Image Prediction Web App (Gradio)")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model path:   {args.model_path}")
    print("=" * 65)

    if args.test_only:
        print("\n[Test Mode] Generating synthetic test image...")
        test_img = Image.new("RGB", (128, 128), color=(120, 150, 200))
        top_res, probs = predict_custom_image(test_img)
        print(f"Prediction result: {top_res}")
        print("Top 3 probabilities:")
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for name, p in sorted_probs:
            print(f"  - {name}: {p*100:.2f}%")
        print("\nPipeline test passed successfully.")
        sys.exit(0)

    app = create_app(model_path=args.model_path)
    print(f"\nLaunching Gradio server on http://{args.host}:{args.port}...")
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
