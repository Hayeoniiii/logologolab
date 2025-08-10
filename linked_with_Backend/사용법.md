패키지 설치
```python
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install \
  transformers==4.47.0 \
  accelerate==1.2.0 \
  peft>=0.14.0 \
  wandb \
  datasets \
  packaging \
  tqdm \
  sentencepiece \
  huggingface_hub \
  fastapi \
  python-multipart \
  uvicorn

pip install git+https://github.com/huggingface/diffusers.git
huggingface-cli login #허깅페이스 로그인
```

fastapi 서버 실행
```python
uvicorn app:app --host 0.0.0.0 --port 8000
```
