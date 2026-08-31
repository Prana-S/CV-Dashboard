#!/usr/bin/env python3
"""Open a small desktop dashboard for testing the trained image classifier."""

from __future__ import annotations

import argparse
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageTk

from live_dashboard import PredictionResult, draw_prediction, predict_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models/plastic_pulse_week3_tuned.keras"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/week4/demo_dashboard"
SUPPORTED_IMAGES = (("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*"))


class DemoDashboard:
    def __init__(self, root: tk.Tk, model_path: Path, output_dir: Path) -> None:
        self.root = root
        self.model_path = model_path
        self.output_dir = output_dir
        self.model: tf.keras.Model | None = None
        self.image_path: Path | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None

        self.root.title("Marine Debris Image Classifier")
        self.root.geometry("1080x720")
        self.root.minsize(920, 620)
        self.root.configure(bg="#eef6f6")

        self.status_text = tk.StringVar(value="Loading trained model...")
        self.result_text = tk.StringVar(value="")
        self.confidence_text = tk.StringVar(value="")
        self.file_text = tk.StringVar(value="No image selected")

        self._build_layout()
        self._load_model_in_background()

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg="#083344", height=92)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="Marine Debris Image Classifier",
            font=("Arial", 24, "bold"),
            fg="white",
            bg="#083344",
        ).pack(anchor="w", padx=34, pady=(18, 2))
        tk.Label(
            header,
            text="Test a new image with the trained MobileNetV2 model",
            font=("Arial", 11),
            fg="#b9dde0",
            bg="#083344",
        ).pack(anchor="w", padx=35)

        content = tk.Frame(self.root, bg="#eef6f6")
        content.pack(fill="both", expand=True, padx=28, pady=24)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)

        preview_card = tk.Frame(content, bg="white", highlightbackground="#bfd5d9", highlightthickness=1)
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        preview_card.grid_rowconfigure(0, weight=1)
        preview_card.grid_columnconfigure(0, weight=1)

        self.preview_label = tk.Label(
            preview_card,
            text="Image preview",
            font=("Arial", 18, "bold"),
            fg="#64748b",
            bg="white",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)

        tk.Label(
            preview_card,
            textvariable=self.file_text,
            font=("Arial", 10),
            fg="#526873",
            bg="white",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 14))

        control_card = tk.Frame(content, bg="white", highlightbackground="#bfd5d9", highlightthickness=1)
        control_card.grid(row=0, column=1, sticky="nsew")

        tk.Label(
            control_card,
            text="Test your image",
            font=("Arial", 21, "bold"),
            fg="#083344",
            bg="white",
        ).pack(anchor="w", padx=26, pady=(28, 8))

        tk.Label(
            control_card,
            text="Select a JPG, PNG, BMP, or WebP image from your computer.",
            wraplength=330,
            justify="left",
            font=("Arial", 11),
            fg="#526873",
            bg="white",
        ).pack(anchor="w", padx=26, pady=(0, 18))

        self.choose_button = ttk.Button(control_card, text="Choose image", command=self.choose_image)
        self.choose_button.pack(fill="x", padx=26, pady=(0, 10), ipady=7)

        self.run_button = ttk.Button(control_card, text="Run classification", command=self.run_classification)
        self.run_button.pack(fill="x", padx=26, pady=(0, 22), ipady=7)
        self.run_button.state(["disabled"])

        result_label = tk.Label(
            control_card,
            textvariable=self.result_text,
            wraplength=310,
            justify="left",
            font=("Arial", 22, "bold"),
            fg="#083344",
            bg="white",
        )
        result_label.pack(anchor="w", padx=26, pady=(0, 3))

        confidence_label = tk.Label(
            control_card,
            textvariable=self.confidence_text,
            font=("Arial", 14, "bold"),
            fg="#e76f51",
            bg="white",
        )
        confidence_label.pack(anchor="w", padx=26, pady=(0, 18))

        tk.Label(
            control_card,
            textvariable=self.status_text,
            font=("Arial", 10, "bold"),
            fg="#0e7490",
            bg="white",
        ).pack(anchor="w", padx=26, pady=(0, 16))

        tk.Label(
            control_card,
            text=(
                "The model classifies the complete image as Marine Life or Plastic Debris. "
                "The rectangle shows the analyzed image region, not a detected object."
            ),
            wraplength=330,
            justify="left",
            font=("Arial", 9),
            fg="#64748b",
            bg="white",
        ).pack(side="bottom", anchor="w", padx=26, pady=24)

    def _load_model_in_background(self) -> None:
        def load() -> None:
            try:
                model = tf.keras.models.load_model(self.model_path)
            except Exception as error:
                self.root.after(0, lambda error=error: self._show_model_error(error))
                return
            self.root.after(0, lambda: self._model_ready(model))

        threading.Thread(target=load, daemon=True).start()

    def _model_ready(self, model: tf.keras.Model) -> None:
        self.model = model
        self.status_text.set("Model ready")
        if self.image_path is not None:
            self.run_button.state(["!disabled"])

    def _show_model_error(self, error: Exception) -> None:
        self.status_text.set("Model could not be loaded")
        messagebox.showerror("Model error", str(error))

    def choose_image(self) -> None:
        selected = filedialog.askopenfilename(title="Choose an image", filetypes=SUPPORTED_IMAGES)
        if not selected:
            return

        image_path = Path(selected)
        try:
            preview = self._read_image(image_path)
        except Exception as error:
            messagebox.showerror("Image error", f"The selected image could not be opened.\n\n{error}")
            return

        self.image_path = image_path
        self.file_text.set(image_path.name)
        self.result_text.set("Image ready")
        self.confidence_text.set("")
        self.status_text.set("Ready to classify")
        self._show_preview(preview)
        if self.model is not None:
            self.run_button.state(["!disabled"])

    def _read_image(self, image_path: Path) -> Image.Image:
        with Image.open(image_path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")

    def _show_preview(self, image: Image.Image) -> None:
        preview = image.copy()
        preview.thumbnail((630, 520), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self.preview_photo, text="")

    def run_classification(self) -> None:
        if self.image_path is None or self.model is None:
            return

        self.run_button.state(["disabled"])
        self.choose_button.state(["disabled"])
        self.status_text.set("Analyzing image...")
        self.result_text.set("Working...")
        self.confidence_text.set("")
        threading.Thread(target=self._classify_selected_image, daemon=True).start()

    def _classify_selected_image(self) -> None:
        assert self.image_path is not None
        assert self.model is not None

        try:
            pil_image = self._read_image(self.image_path)
            rgb_array = np.asarray(pil_image)
            bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            result = predict_image(self.model, bgr_image, threshold=0.5)
            annotated = draw_prediction(bgr_image, result, self.image_path)
            output_path = self._save_result(annotated, self.image_path)
        except Exception as error:
            self.root.after(0, lambda error=error: self._show_prediction_error(error))
            return

        self.root.after(0, lambda: self._show_result(result, annotated, output_path))

    def _save_result(self, annotated: np.ndarray, source_path: Path) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = self.output_dir / f"{source_path.stem}_{timestamp}_annotated.jpg"
        if not cv2.imwrite(str(output_path), annotated):
            raise OSError(f"Could not save the annotated image to {output_path}")
        return output_path

    def _show_result(self, result: PredictionResult, annotated: np.ndarray, output_path: Path) -> None:
        rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self._show_preview(Image.fromarray(rgb_annotated))
        self.result_text.set(result.label)
        self.confidence_text.set(f"Confidence: {result.confidence:.1%}")
        self.status_text.set(f"Saved result: {output_path.name}")
        self.choose_button.state(["!disabled"])
        self.run_button.state(["!disabled"])

    def _show_prediction_error(self, error: Exception) -> None:
        self.status_text.set("The image could not be classified")
        self.result_text.set("Try another image")
        self.choose_button.state(["!disabled"])
        self.run_button.state(["!disabled"])
        messagebox.showerror("Prediction error", str(error))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the desktop image-classifier demo dashboard.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Trained model was not found: {args.model_path}")

    root = tk.Tk()
    DemoDashboard(root, args.model_path, args.output_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
