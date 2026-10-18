# Runs step 3 (perplexity) then step 2 (grid); everything is logged to run.log
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONIOENCODING = "utf-8"
& .\.venv\Scripts\python.exe ppl.py 2>&1 | Tee-Object -FilePath run.log
& .\.venv\Scripts\python.exe grid.py 2>&1 | Tee-Object -FilePath run.log -Append
"ALL DONE" | Tee-Object -FilePath run.log -Append
