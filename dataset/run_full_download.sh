#!/usr/bin/env bash
cd "/Users/macos/Documents/PRT661/software"
echo "=== Full BTS download started: $(date) ===" >> dataset/download.log
python3 ingestion/download_bts.py >> dataset/download.log 2>&1
exit_code=$?
echo "=== Full BTS download finished: $(date), exit=${exit_code} ===" >> dataset/download.log
if [ "$exit_code" -eq 0 ]; then
    touch dataset/DOWNLOAD_DONE
else
    touch dataset/DOWNLOAD_FAILED
fi
