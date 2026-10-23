@echo off
title Push Waste Classifier to GitHub
cd /d "%~dp0"

echo =====================================================================
echo  Pushing Waste Segregation Classifier to GitHub
echo =====================================================================
echo.

set GIT_EXE=C:\Users\durga\AppData\Local\Microsoft\WinGet\Packages\Git.MinGit_Microsoft.Winget.Source_8wekyb3d8bbwe\cmd\git.exe
if not exist "%GIT_EXE%" set GIT_EXE=git

set /p REPO_URL="Enter your GitHub Repository URL (e.g. https://github.com/your-username/waste-segregation-classifier.git): "

if "%REPO_URL%"=="" (
    echo.
    echo [ERROR] No GitHub URL entered. Please create a repository on https://github.com/new first.
    pause
    exit /b
)

echo.
echo [1/3] Setting remote origin...
"%GIT_EXE%" remote remove origin >nul 2>&1
"%GIT_EXE%" remote add origin %REPO_URL%

echo [2/3] Renaming branch to main...
"%GIT_EXE%" branch -M main

echo [3/3] Pushing all files, model weights, and UI code to GitHub...
"%GIT_EXE%" push -u origin main

echo.
echo =====================================================================
echo  SUCCESS! Your code is now live on GitHub!
echo  Next step: Deploy for FREE on Streamlit Cloud (https://share.streamlit.io)
echo =====================================================================
pause
