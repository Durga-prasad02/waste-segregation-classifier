@echo off
title Push Waste Classifier to GitHub
cd /d "%~dp0"

echo =====================================================================
echo  Pushing Waste Segregation Classifier to:
echo  https://github.com/durga-prasad02/waste-segregation-classifier.git
echo =====================================================================
echo.

set GIT_EXE=C:\Users\durga\AppData\Local\Microsoft\WinGet\Packages\Git.MinGit_Microsoft.Winget.Source_8wekyb3d8bbwe\cmd\git.exe
if not exist "%GIT_EXE%" set GIT_EXE=git

echo [1/3] Setting remote repository origin...
"%GIT_EXE%" remote remove origin >nul 2>&1
"%GIT_EXE%" remote add origin https://github.com/durga-prasad02/waste-segregation-classifier.git

echo [2/3] Setting branch to main...
"%GIT_EXE%" branch -M main

echo [3/3] Pushing files and model weights to GitHub...
echo (If a browser window pops up, click 'Sign in with your browser')
echo.
"%GIT_EXE%" push -u origin main

echo.
echo =====================================================================
echo  SUCCESS! All code is now pushed to GitHub!
echo =====================================================================
pause
