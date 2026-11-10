@echo off
chcp 65001 > nul
title Semantic Communication - DeepSC Demo
echo ======================================================================
echo          HE THONG TRUYEN THONG NGU NGHIA DEEPSC (TRANSFORMER JSCC)
echo                       KHOI CHAY WEB DEMO
echo ======================================================================
echo.

cd /d "%~dp0"
set USERPROFILE=%~dp0
set STREAMLIT_CONFIG_DIR=%~dp0.streamlit
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

REM Kiem tra python
set PY_CMD=
py -3.11 --version >nul 2>nul && set PY_CMD=py -3.11
if not defined PY_CMD py --version >nul 2>nul && set PY_CMD=py
if not defined PY_CMD python --version >nul 2>nul && set PY_CMD=python

if not defined PY_CMD goto NO_PYTHON

REM Kiem tra streamlit
%PY_CMD% -c "import streamlit" >nul 2>nul
if %ERRORLEVEL% NEQ 0 goto INSTALL_REQS

:START_DEMO
echo [*] Python va cac thu vien da san sang.
echo [*] Dang khoi dong Web Demo tai: http://localhost:8501
echo [*] Neu trinh duyet khong tu dong bat, ban hay vao: http://localhost:8501
echo [*] Nhan Ctrl + C hoac dong cua so nay de dung chuong trinh.
echo.
%PY_CMD% -m streamlit run demo/app.py --server.port 8501
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [*] Dang thu cong du phong 8502...
    %PY_CMD% -m streamlit run demo/app.py --server.port 8502
)
goto END

:INSTALL_REQS
echo [*] Dang cai dat thu vien lan dau (chi mat 2-3 phut)...
%PY_CMD% -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    %PY_CMD% -m pip install streamlit torch matplotlib numpy pandas
)
goto START_DEMO

:NO_PYTHON
echo [LOI] Khong tim thay Python tren may tinh cua ban!
echo.
echo [HUONG DAN KHAC PHUC]:
echo 1. Vao web https://www.python.org/downloads/ tai Python 3.10 hoac 3.11.
echo 2. Khi cai dat, NHO TICH VAO O: "Add Python to PATH" (o duoi cung).
echo 3. Sau khi cai xong, hay nhap dup lai vao file run_demo_full.bat.
echo.

:END
pause
