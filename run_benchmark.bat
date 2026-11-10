@echo off
chcp 65001 > nul
title DeepSC - Chay Khao Sat Hieu Nang (Xuat JSON Benchmark)
echo ======================================================================
echo       CHUONG TRINH DO LUONG VA XUAT SO LIEU BENCHMARK JSON (DEEPSC)
echo ======================================================================
echo.

cd /d "%~dp0"

REM Kiem tra python
set PY_CMD=
py -3.11 --version >nul 2>nul && set PY_CMD=py -3.11
if not defined PY_CMD py --version >nul 2>nul && set PY_CMD=py
if not defined PY_CMD python --version >nul 2>nul && set PY_CMD=python

if not defined PY_CMD goto NO_PYTHON

echo Chon che do khao sat danh gia:
echo.
echo  [1] Chay thu nghiem nhanh (100 mau lay deu tu ca 2 dataset - khoang 15 giay)
echo  [2] Chay 1,000 mau ket hop ca 2 dataset (500 OPUS + 500 Tatoeba - khoang 2 phut)
echo  [3] Chay TOAN BO CA 2 LOAI DATASET KIEM THU (OPUS-100 Test + Tatoeba: 25,172 mau)
echo  [4] Chay toan bo OPUS-100 Test Split (2,000 mau)
echo  [5] Chay toan bo Tatoeba (23,172 mau)
echo  [6] Chay so luong mau tuy chon tu ca 2 dataset (Nhap so)
echo.
set /p opt="Nhap lua chon cua ban [1-6]: "
if defined opt set opt=%opt: =%

if "%opt%"=="1" goto OPT_1
if "%opt%"=="2" goto OPT_2
if "%opt%"=="3" goto OPT_3
if "%opt%"=="4" goto OPT_4
if "%opt%"=="5" goto OPT_5
if "%opt%"=="6" goto OPT_6
goto INVALID_OPT

:OPT_1
echo.
echo [*] Dang chay 100 mau ket hop ca OPUS-100 va Tatoeba...
%PY_CMD% evaluation/run_benchmark_cache.py --dataset all --samples 100 --channel awgn
goto FINISH

:OPT_2
echo.
echo [*] Dang chay 1,000 mau ket hop ca OPUS-100 va Tatoeba...
%PY_CMD% evaluation/run_benchmark_cache.py --dataset all --samples 1000 --channel awgn
goto FINISH

:OPT_3
echo.
echo [*] Dang chay TOAN BO CA 2 LOAI DATASET KIEM THU: 25,172 mau
echo [*] Luu y: Tien trinh tren CPU se mat khoang 15-20 phut. Vui long cho...
%PY_CMD% evaluation/run_benchmark_cache.py --dataset all --samples 0 --channel awgn
goto FINISH

:OPT_4
echo.
echo [*] Dang chay toan bo 2,000 mau OPUS-100 Test...
%PY_CMD% evaluation/run_benchmark_cache.py --dataset opus_test --samples 0 --channel awgn
goto FINISH

:OPT_5
echo.
echo [*] Dang chay toan bo 23,172 mau Tatoeba...
%PY_CMD% evaluation/run_benchmark_cache.py --dataset tatoeba --samples 0 --channel awgn
goto FINISH

:OPT_6
echo.
set /p custom_samples="Nhap so luong mau can test tu ca 2 bo (vi du: 500, 2000): "
echo.
echo [*] Dang chay so mau yeu cau lay deu tu ca OPUS va Tatoeba...
%PY_CMD% evaluation/run_benchmark_cache.py --dataset all --samples %custom_samples% --channel awgn
goto FINISH

:INVALID_OPT
echo.
echo [!] Lua chon khong hop le.
goto FINISH

:NO_PYTHON
echo [LOI] Khong tim thay Python tren he thong!
goto FINISH

:FINISH
echo.
echo ======================================================================
echo  Ket qua JSON da duoc luu tai thu muc: data\benchmarks\
echo  Ban co the mo Web Demo (run_demo_full.bat) -^> Chon "Khao Sat Hieu Nang Kenh"
echo  -^> "Tai len tep ket qua tu may tinh" va chon tep vua tao!
echo ======================================================================
echo.
pause
