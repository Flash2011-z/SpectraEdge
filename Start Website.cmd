@echo off
setlocal
title SpectraEdge - Local Website
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
    echo Node.js is required. Install Node.js 22.13 or newer, then run this file again.
    pause
    exit /b 1
)
where npm.cmd >nul 2>nul
if errorlevel 1 (
    echo npm was not found. Repair your Node.js installation, then try again.
    pause
    exit /b 1
)
if not exist "%~dp0frontend\node_modules\vinext\package.json" (
    echo Frontend dependencies are missing.
    echo Open a terminal in this folder and run: npm.cmd run setup
    pause
    exit /b 1
)
echo Starting SpectraEdge...
echo Open the Local address shown below in your browser.
echo Keep this terminal open. Press Ctrl+C to stop the website.
echo.
call npm.cmd run dev
if errorlevel 1 (
    echo.
    echo SpectraEdge could not start. Read the error above.
    pause
    exit /b 1
)
endlocal
