#!/bin/bash

set -e
cd "$(dirname "$0")"

echo "Setting up a virtual environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install pillow numpy pyinstaller -q

echo "Building ColorCorrect.app..."
pyinstaller --windowed --name "ColorCorrect" --noconfirm --clean color_correct_gui.py

echo ""
echo "Done! Your app is at: dist/ColorCorrect.app"
echo "Drag it into /Applications if you'd like it there permanently."
echo ""
echo "First launch only: macOS will likely say it can't verify the developer"
echo "(this is normal for an app you built yourself, not a bug)."
echo "Right-click (Control-click) ColorCorrect.app -> Open -> Open, once."
