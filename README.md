# text-to-image 로고 생성 모델 
사용자가 원하는 텍스트 프롬프트를 입력하면 나만의 브랜드 스타일에 맞춘 로고 이미지를 생성<br>
본 프로젝트는 Hugging Face에서 공개한 FLUX.1-dev 텍스트-투-이미지 모델을 기반으로 브랜드 로고 생성에 특화된 스타일로 LoRA 파인튜닝을 적용한 결과물입니다.

### 프로젝트 개요
베이스 모델: black-forest-labs/FLUX.1-dev<br>
파인튜닝 기법: LoRA (Low-Rank Adaptation)<br>
학습 방식: DreamBooth 기반 커스텀 이미지 학습<br>
데이터셋: 다양한 로고 스타일 이미지를 직접 수집하여 전처리한 뒤, Hugging Face Datasets에 업로드 (ID: logologolab)<br>
사용 목적: 사용자가 원하는 텍스트 프롬프트를 입력하면, 나만의 브랜드 스타일에 맞춘 로고 이미지를 생성할 수 있음<br>   

### 기대효과 
디자인 비용 및 시간을 절감
더욱 정교한 텍스트 기반 이미지 생성을 실현하며 기존 AI 모델의 한계 극복



