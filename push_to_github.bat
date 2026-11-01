@echo off
title Push Waste Classifier to GitHub
cd /d "%~dp0"

echo =====================================================================
echo  Pushing Waste Segregation Classifier to GitHub
echo  Repository: https://github.com/durga-prasad02/waste-segregation-classifier
echo =====================================================================
echo.

set PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\GitHub.cli_Microsoft.Winget.Source_8wekyb3d8bbwe\bin;%LOCALAPPDATA%\Microsoft\WinGet\Packages\Git.MinGit_Microsoft.Winget.Source_8wekyb3d8bbwe\cmd;%PATH%

echo [1/4] Checking GitHub Login Status...
gh auth status >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo -----------------------------------------------------------------
    echo  You are not logged in to GitHub yet.
    echo  A browser window will now open to authenticate your account.
    echo -----------------------------------------------------------------
    echo.
    gh auth login --web -p https
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERROR] GitHub login was cancelled or failed.
        pause
        exit /b 1
    )
)

echo.
echo [2/4] Configuring Git Credentials Helper...
gh auth setup-git

echo.
echo [3/4] Setting remote repository origin...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/durga-prasad02/waste-segregation-classifier.git
git branch -M main

echo.
echo [4/4] Pushing code, model weights, and sample images to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =====================================================================
    echo  SUCCESS! All code is now pushed to GitHub!
    echo.
    echo  Now go back to Streamlit Cloud (https://share.streamlit.io/deploy):
    echo  1. Refresh the Streamlit deployment page.
    echo  2. The red warning for 'main' and 'app.py' will turn green/valid!
    echo  3. Click 'Deploy!'
    echo =====================================================================
) else (
    echo.
    echo [ERROR] Git push failed. Please review the message above.
)
pause
