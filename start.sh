#!/bin/bash
conda init bash
source ~/.bashrc
conda activate CryptoTrader
pm2 start "python -u /CryptoTrader/src/Trader/manage.py run --no-reload" --name "TraderBot" --output /CryptoTrader/stdout.log --error /CryptoTrader/stderr.log
