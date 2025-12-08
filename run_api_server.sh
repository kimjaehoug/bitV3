#!/bin/bash

# API 서버 실행 스크립트

echo "=" * 60
echo "백엔드 API 서버 시작"
echo "=" * 60

# 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# API 서버 실행
python api_server.py



