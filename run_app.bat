@echo off
title Waste Segregation Image Classifier & Smart Waste Segregation Assistant
cd /d "%~dp0"

echo =====================================================================
echo  Waste Segregation Image Classifier & Smart Waste Segregation Assistant
echo  Candidate: AGGIBADALA VENKATA DURGA NAGA PRASAD (23ME1A5402)
echo  Guide    : Ms. G. Hema Sri
echo  Domain   : Deep Learning & Computer Vision
echo =====================================================================
echo.

set PYTHON_EXE=C:\Users\durga\AppData\Local\Programs\Python\Python311\python.exe

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

echo [1/3] Verifying Dataset and Benchmark Samples...
"%PYTHON_EXE%" dataset_downloader.py

if not exist "waste_classifier_model.pth" (
    echo.
    echo [2/3] Training Initial CNN Classifier Model...
    "%PYTHON_EXE%" train.py --epochs 8
) else (
    echo.
    echo [2/3] Trained Model Checkpoint 'waste_classifier_model.pth' Found!
)

echo.
echo [3/3] Launching Smart-Bin Sorting Assistant Web UI...
echo =====================================================================
"%PYTHON_EXE%" -m streamlit run app.py
pause
