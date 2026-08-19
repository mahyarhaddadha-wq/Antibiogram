@echo off
setlocal

rem Always run from the folder this .bat file lives in, regardless of where
rem it was double-clicked from (so it finds batch_process_antibiogram.py
rem and the notebook next to it).
cd /d "%~dp0"

rem Prefer the standard Windows "py" launcher; fall back to "python" if it's
rem not on PATH (some installs only register "python").
where py >nul 2>nul
if %errorlevel%==0 (
    py batch_process_antibiogram.py %*
    goto :done
)

where python >nul 2>nul
if %errorlevel%==0 (
    python batch_process_antibiogram.py %*
    goto :done
)

echo Python was not found on PATH.
echo Install it from https://www.python.org/downloads/ and make sure to
echo check "Add python.exe to PATH" during setup, then run this again.
pause
exit /b 1

:done
echo.
echo Done. Press any key to close this window.
pause >nul
