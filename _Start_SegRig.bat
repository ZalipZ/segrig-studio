@echo off
:: หาที่อยู่ของ Conda ในเครื่องเพื่อเปิดระบบก่อน
IF EXIST "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
) ELSE IF EXIST "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
) ELSE IF EXIST "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    call "C:\ProgramData\miniconda3\Scripts\activate.bat"
) ELSE (
    echo "Conda path not found! Please open Anaconda Prompt manually."
    pause
    exit
)

:: เปิดใช้งาน Environment และรันเซิร์ฟเวอร์
call conda activate segrig
cd /d C:\AI\app
python segrig_studio.py
pause