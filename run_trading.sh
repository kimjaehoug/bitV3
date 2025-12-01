#!/bin/bash

# 바이낸스 API 키 설정 (여기에 실제 키를 입력하세요)
export BINANCE_API_KEY="your_api_key_here"
export BINANCE_API_SECRET="your_api_secret_here"

# 실거래 실행
python realtime_trading.py --trade

