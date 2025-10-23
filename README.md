# RAG System (Pinecone + OpenAI)

이 프로젝트는 Pinecone 벡터 데이터베이스와 OpenAI LLM을 사용한 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 파일 구조

- `config.py` - API 키 및 설정 관리
- `RAG_Embedding.ipynb` - 문서 임베딩 및 Pinecone 업서트 전용 노트북
- `RAG_Chatbot.ipynb` - 챗봇 실행 전용 노트북

## 설정 방법

### 1. API 키 설정

`config.py` 파일을 열어서 실제 API 키 값을 입력하세요:

```python
# OpenAI API Key
OPENAI_API_KEY = "sk-your-actual-openai-api-key"

# Pinecone API Key  
PINECONE_API_KEY = "your-actual-pinecone-api-key"
```

### 2. 문서 경로 설정

`config.py`에서 문서 폴더 경로를 사용자 환경에 맞게 수정하세요:

```python
DOCUMENT_PATHS = {
    "products": r"C:\your\path\to\products\folder",
    "company": r"C:\your\path\to\company\folder"
}
```

## 사용 방법

### 1. 문서 임베딩 (RAG_Embedding.ipynb)

1. 노트북을 열고 셀을 순서대로 실행
2. 문서가 Pinecone에 업로드됩니다
3. 한 번만 실행하면 됩니다

### 2. 챗봇 실행 (RAG_Chatbot.ipynb)

1. 노트북을 열고 셀을 순서대로 실행
2. 챗봇이 시작됩니다
3. 질문을 입력하여 대화하세요
4. `exit` 또는 `quit`으로 종료

## 주요 기능

- **하이브리드 검색**: 벡터 검색 + BM25 키워드 검색
- **LLM 기반 질의 이해**: 의도 분류, 질의 재작성, 키워드 확장
- **대화 메모리**: 최근 3턴 대화 기억
- **집계 기능**: 개수/총합 등 집계 질문 처리
- **디버그 출력**: 검색 과정 상세 표시

## 설정 옵션

`config.py`에서 다음 설정을 조정할 수 있습니다:

- 검색 가중치 (벡터/BM25)
- 컨텍스트 길이 및 개수
- 청킹 크기 및 오버랩
- 모델 이름
- Pinecone 인덱스 설정