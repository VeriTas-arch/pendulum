@REM version 1.0.0
@echo off
chcp 65001 >nul
title OVOLAB - Format Script
cd %~dp0
type logo.txt
echo.

REM This script is used to check the code format in the project.
REM Please run with conda environment activated.
cd %~dp0..
echo Current formatting settings: [max-line-length=88][skip magic trailing comma]
black . --diff --color --line-length=88 --skip-magic-trailing-comma
echo Run isort
isort . --check-only --diff --color
pause
