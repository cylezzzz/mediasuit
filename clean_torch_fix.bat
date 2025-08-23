@echo off
echo ============================================
echo   LocalMediaSuite - Saubere PyTorch Fix
echo ============================================
echo.

echo [1/5] Entferne ALLE PyTorch-Pakete komplett...
pip uninstall torch torchvision torchaudio xformers accelerate diffusers transformers -y

echo.
echo [2/5] Lösche alle Caches...
rmdir /s /q "%USERPROFILE%\.cache\torch" 2>nul
rmdir /s /q "%USERPROFILE%\.cache\huggingface" 2>nul
rmdir /s /q "%USERPROFILE%\.cache\pip" 2>nul

echo.
echo [3/5] Installiere konsistente PyTorch-Version 2.4.0 + CUDA 11.8...
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 --index-url https://download.pytorch.org/whl/cu118

echo.
echo [4/5] Installiere kompatible xformers...
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118

echo.
echo [5/5] Reinstalliere ML-Bibliotheken...
pip install diffusers==0.30.3 transformers==4.44.2 accelerate==0.33.0

echo.
echo ============================================
echo   Installation abgeschlossen!
echo ============================================
echo.

echo Teste komplette Installation:
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA verfügbar: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')

try:
    import diffusers
    print(f'Diffusers Version: {diffusers.__version__}')
    print('✅ Diffusers erfolgreich geladen')
except Exception as e:
    print(f'❌ Diffusers Fehler: {e}')

try:
    import xformers
    print(f'xFormers Version: {xformers.__version__}')
    print('✅ xFormers erfolgreich geladen')
except Exception as e:
    print(f'❌ xFormers Fehler: {e}')
"

echo.
echo ============================================
echo Starte LocalMediaSuite...
python start.py