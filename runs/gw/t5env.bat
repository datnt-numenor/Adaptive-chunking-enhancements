@echo off
rem Dat san bien cho cac lenh Table 5. KHONG chua key (key nam trong D:\Project\keys\*.env).
rem   call D:\Project\adaptive-chunking\runs\gw\t5env.bat         -> bat .venv (prepare, generate, export)
rem   call D:\Project\adaptive-chunking\runs\gw\t5env.bat judge   -> bat .venv-judge (buoc cham diem)
cd /d D:\Project\Adaptive-chunking-enhancements
if /i "%~1"=="judge" (
    call D:\Project\Adaptive-chunking-enhancements\.venv-judge\Scripts\activate.bat
) else (
    call D:\Project\adaptive-chunking\.venv\Scripts\activate.bat
)
set R=D:\Project\adaptive-chunking\runs
set K=D:\Project\keys
set GW=%R%\gw
set IN=--qa %R%\table5_v2\qa-full\qa_frozen.jsonl --retrieval-dir %R%\table5_v2_results\table5_v2\retrieval-full --evaluation %R%\table5_v2_results\table5_v2\evaluation-full\retrieval_evaluation.json
set D=python -X utf8 %GW%\table5_run.py --model gpt-4o-mini-2024-07-18 --price-in 0.15 --price-out 0.60
set OUT=%R%\t5_dev_gw_4omini
echo [t5env] thu muc: %CD%
echo [t5env] OUT=%OUT%
where python
