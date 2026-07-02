#!/usr/bin/env python3
"""
color_correct_gui.py - Desktop app for color-correcting faded photographs.

Requirements: pillow, numpy, tkinter
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ============================================================================
# Core color-correction engine (identical math to the CLI script)
# ============================================================================

def percentile_stretch(channel: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Linearly stretch one channel so its low/high percentile values map to 0/255."""
    low_val = np.percentile(channel, low_pct)
    high_val = np.percentile(channel, high_pct)
    if high_val <= low_val:
        return channel.astype(np.float32)
    return (channel.astype(np.float32) - low_val) * (255.0 / (high_val - low_val))


def auto_balance(rgb: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Apply percentile_stretch independently to each of the R, G, B channels."""
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = percentile_stretch(rgb[:, :, c], low_pct, high_pct)
    return np.clip(out, 0, 255)


def gray_world(rgb: np.ndarray) -> np.ndarray:
    """Scale each channel so its mean matches the overall gray mean."""
    out = rgb.astype(np.float32).copy()
    means = out.reshape(-1, 3).mean(axis=0)
    target = means.mean()
    for c in range(3):
        if means[c] > 1e-6:
            out[:, :, c] *= target / means[c]
    return np.clip(out, 0, 255)


def boost_saturation(rgb: np.ndarray, factor: float) -> np.ndarray:
    """Push pixels away from (factor > 1) or toward (factor < 1) their gray value."""
    if factor == 1.0:
        return rgb
    luminance = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.stack([luminance] * 3, axis=-1)
    return np.clip(gray + (rgb - gray) * factor, 0, 255)


def correct_rgb(rgb: np.ndarray, method: str, low_pct: float, high_pct: float,
                 saturation: float) -> np.ndarray:
    """Run the full correction pipeline on a float32 HxWx3 array; return uint8 array."""
    if method == "balance":
        rgb = auto_balance(rgb, low_pct, high_pct)
    else:
        rgb = gray_world(rgb)
    rgb = boost_saturation(rgb, saturation)
    return rgb.astype(np.uint8)


def load_rgb_and_alpha(img: Image.Image):
    """Return (rgb_float32_array, alpha_uint8_array_or_None) for any PIL image mode."""
    has_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
    rgba = img.convert("RGBA") if has_alpha else img.convert("RGB")
    arr = np.array(rgba).astype(np.float32)
    if has_alpha:
        return arr[:, :, :3], arr[:, :, 3].astype(np.uint8)
    return arr, None


def correct_image_file(src: Path, dst: Path, method: str, low_pct: float, high_pct: float,
                        saturation: float, quality: int = 95) -> None:
    """Full-resolution correct-and-save. Used by Save As and batch processing."""
    img = Image.open(src)
    img.load()
    rgb, alpha = load_rgb_and_alpha(img)
    corrected = correct_rgb(rgb, method, low_pct, high_pct, saturation)

    if alpha is not None:
        out_img = Image.fromarray(np.dstack([corrected, alpha]), mode="RGBA")
    else:
        out_img = Image.fromarray(corrected, mode="RGB")

    save_kwargs = {}
    if img.info.get("exif"):
        save_kwargs["exif"] = img.info["exif"]
    if img.info.get("icc_profile"):
        save_kwargs["icc_profile"] = img.info["icc_profile"]
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs["quality"] = quality
        if out_img.mode == "RGBA":
            out_img = out_img.convert("RGB")  # JPEG has no alpha channel

    dst.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(dst, **save_kwargs)


# ============================================================================
# GUI
# ============================================================================

PREVIEW_MAX_SIDE = 420  # downsampled preview size in px, keeps slider drags instant
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ColorCorrectApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Faded Photo Color Correct")
        self.root.geometry("1040x700")
        self.root.minsize(820, 560)

        self.source_path: Optional[Path] = None
        self.full_rgb: Optional[np.ndarray] = None       # full-res float32 RGB, for saving
        self.preview_rgb: Optional[np.ndarray] = None    # downsampled float32 RGB, for live preview
        self.original_preview_tk = None                  # keep references so Tk doesn't GC them
        self.corrected_preview_tk = None

        self.method_var = tk.StringVar(value="balance")
        self.low_var = tk.DoubleVar(value=0.5)
        self.high_var = tk.DoubleVar(value=99.5)
        self.saturation_var = tk.DoubleVar(value=1.15)
        self.status_var = tk.StringVar(value="Open an image to get started.")

        self._build_layout()
        self._update_control_state()

    # ---- layout -------------------------------------------------------

    def _build_layout(self):
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side="top", fill="x")
        ttk.Button(toolbar, text="Open Image…", command=self.open_image).pack(side="left")
        self.save_btn = ttk.Button(toolbar, text="Save As…", command=self.save_image, state="disabled")
        self.save_btn.pack(side="left", padx=6)
        ttk.Button(toolbar, text="Batch Process Folder…", command=self.batch_process).pack(side="left")

        body = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        body.pack(side="top", fill="both", expand=True)

        # --- sidebar controls ---
        sidebar = ttk.Frame(body, padding=8, width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="Method", font=("", 11, "bold")).pack(anchor="w")
        ttk.Radiobutton(sidebar, text="Balance (stronger)", variable=self.method_var,
                         value="balance", command=self._on_setting_changed).pack(anchor="w", pady=2)
        ttk.Radiobutton(sidebar, text="Gray World (gentler)", variable=self.method_var,
                         value="grayworld", command=self._on_setting_changed).pack(anchor="w", pady=2)

        ttk.Separator(sidebar).pack(fill="x", pady=10)

        self.low_label = ttk.Label(sidebar, text="Low clip: 0.5%")
        self.low_label.pack(anchor="w")
        self.low_scale = ttk.Scale(sidebar, from_=0, to=10, orient="horizontal",
                                    variable=self.low_var, command=self._on_low_changed)
        self.low_scale.pack(fill="x")

        self.high_label = ttk.Label(sidebar, text="High clip: 99.5%")
        self.high_label.pack(anchor="w", pady=(8, 0))
        self.high_scale = ttk.Scale(sidebar, from_=90, to=100, orient="horizontal",
                                     variable=self.high_var, command=self._on_high_changed)
        self.high_scale.pack(fill="x")

        ttk.Separator(sidebar).pack(fill="x", pady=10)

        self.sat_label = ttk.Label(sidebar, text="Saturation: 1.15x")
        self.sat_label.pack(anchor="w")
        self.sat_scale = ttk.Scale(sidebar, from_=0.5, to=2.0, orient="horizontal",
                                    variable=self.saturation_var, command=self._on_sat_changed)
        self.sat_scale.pack(fill="x")

        ttk.Separator(sidebar).pack(fill="x", pady=10)
        ttk.Button(sidebar, text="Reset to Defaults", command=self.reset_defaults).pack(fill="x")

        # --- before / after preview ---
        preview = ttk.Frame(body)
        preview.pack(side="left", fill="both", expand=True, padx=(12, 0))
        preview.columnconfigure(0, weight=1)
        preview.columnconfigure(1, weight=1)
        preview.rowconfigure(1, weight=1)

        ttk.Label(preview, text="Original", font=("", 10, "bold")).grid(row=0, column=0, pady=(0, 4))
        ttk.Label(preview, text="Corrected", font=("", 10, "bold")).grid(row=0, column=1, pady=(0, 4))

        self.original_canvas = tk.Label(preview, background="#222222")
        self.original_canvas.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.corrected_canvas = tk.Label(preview, background="#222222")
        self.corrected_canvas.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=4)
        status_bar.pack(side="bottom", fill="x")

    # ---- control state helpers -----------------------------------------

    def _update_control_state(self):
        state = "normal" if self.method_var.get() == "balance" else "disabled"
        self.low_scale.configure(state=state)
        self.high_scale.configure(state=state)

    def _on_setting_changed(self):
        self._update_control_state()
        self._refresh_preview()

    def _on_low_changed(self, _value=None):
        self.low_label.configure(text=f"Low clip: {self.low_var.get():.1f}%")
        self._refresh_preview()

    def _on_high_changed(self, _value=None):
        self.high_label.configure(text=f"High clip: {self.high_var.get():.1f}%")
        self._refresh_preview()

    def _on_sat_changed(self, _value=None):
        self.sat_label.configure(text=f"Saturation: {self.saturation_var.get():.2f}x")
        self._refresh_preview()

    def reset_defaults(self):
        self.method_var.set("balance")
        self.low_var.set(0.5)
        self.high_var.set(99.5)
        self.saturation_var.set(1.15)
        self.low_label.configure(text="Low clip: 0.5%")
        self.high_label.configure(text="High clip: 99.5%")
        self.sat_label.configure(text="Saturation: 1.15x")
        self._update_control_state()
        self._refresh_preview()

    # ---- open / preview -------------------------------------------------

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose a photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self._load_image(Path(path))
        except Exception as e:
            messagebox.showerror("Couldn't open image", str(e))

    def _load_image(self, path: Path):
        img = Image.open(path)
        img.load()
        rgb, _alpha = load_rgb_and_alpha(img)
        self.source_path = path
        self.full_rgb = rgb

        preview_img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        preview_img.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE), Image.Resampling.LANCZOS)
        self.preview_rgb = np.array(preview_img).astype(np.float32)

        self.original_preview_tk = ImageTk.PhotoImage(preview_img)
        self.original_canvas.configure(image=self.original_preview_tk)

        self.save_btn.configure(state="normal")
        self.status_var.set(f"{path.name}  —  {img.width}×{img.height}px")
        self._refresh_preview()

    def _refresh_preview(self):
        if self.preview_rgb is None:
            return
        corrected = correct_rgb(
            self.preview_rgb, self.method_var.get(),
            self.low_var.get(), self.high_var.get(), self.saturation_var.get(),
        )
        img = Image.fromarray(corrected, mode="RGB")
        self.corrected_preview_tk = ImageTk.PhotoImage(img)
        self.corrected_canvas.configure(image=self.corrected_preview_tk)

    # ---- save / batch -----------------------------------------------------

    def save_image(self):
        if self.source_path is None:
            return
        default_name = f"{self.source_path.stem}_corrected{self.source_path.suffix}"
        out_path = filedialog.asksaveasfilename(
            title="Save corrected photo",
            initialfile=default_name,
            defaultextension=self.source_path.suffix,
            filetypes=[("Same format", f"*{self.source_path.suffix}"), ("All files", "*.*")],
        )
        if not out_path:
            return

        method = self.method_var.get()
        low, high, sat = self.low_var.get(), self.high_var.get(), self.saturation_var.get()
        src = self.source_path

        def work():
            try:
                correct_image_file(src, Path(out_path), method, low, high, sat)
                self.root.after(0, lambda: self.status_var.set(f"Saved: {out_path}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Save failed", str(e)))

        self.status_var.set("Saving…")
        threading.Thread(target=work, daemon=True).start()

    def batch_process(self):
        in_dir = filedialog.askdirectory(title="Folder of photos to correct")
        if not in_dir:
            return
        out_dir = filedialog.askdirectory(title="Folder to save corrected photos into")
        if not out_dir:
            return

        in_dir, out_dir = Path(in_dir), Path(out_dir)
        files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            messagebox.showinfo("No images found", f"No supported image files found in {in_dir}")
            return

        method = self.method_var.get()
        low, high, sat = self.low_var.get(), self.high_var.get(), self.saturation_var.get()

        def work():
            done, failed = 0, []
            for i, f in enumerate(files, 1):
                try:
                    correct_image_file(f, out_dir / f"{f.stem}_corrected{f.suffix}", method, low, high, sat)
                    done += 1
                except Exception as e:
                    failed.append(f"{f.name}: {e}")
                self.root.after(0, lambda i=i, n=len(files): self.status_var.set(f"Processing {i}/{n}…"))

            def finish():
                msg = f"{done}/{len(files)} photos corrected → {out_dir}"
                if failed:
                    msg += f"\n\n{len(failed)} failed:\n" + "\n".join(failed[:10])
                self.status_var.set(f"Batch complete: {done}/{len(files)} saved")
                messagebox.showinfo("Batch complete", msg)

            self.root.after(0, finish)

        self.status_var.set("Starting batch…")
        threading.Thread(target=work, daemon=True).start()


def main():
    root = tk.Tk()
    ColorCorrectApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
