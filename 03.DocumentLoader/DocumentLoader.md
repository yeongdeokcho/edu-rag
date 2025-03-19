# 도큐먼트 로드(Document Loader)


## 1. 도큐먼트 로더란?
다양한 형식과 소스의 문서를 읽고 표준화된 형태로 변환하여 RAG 시스템에서 처리할 수 있는 형태로 준비하는 도구

### 도큐먼트 로더의 기본 기능
- 파일 시스템, 웹, 데이터베이스 등 다양한 소스에서 데이터 추출
- 다양한 형식(PDF, Word, HTML, 이미지 등)의 파일을 텍스트로 변환
- 문서의 구조와 메타데이터 보존
- 인코딩 및 특수 문자 처리

### 도큐먼트의 기본 구조
일반적으로 로더가 처리한 문서는 다음과 같은 구조를 가집니다:
```python
{
    "page_content": "문서의 실제 텍스트 내용",
    "metadata": {
        "source": "파일 경로 또는 URL",
        "author": "작성자",
        "page": 1,
        "created_at": "생성 일자",
        # 기타 메타데이터
    }
}
```

---

## 2. 도큐먼트 로더의 중요성

### RAG 파이프라인에서의 역할
- RAG 시스템의 **첫 단계**로 전체 성능의 기반을 형성
- 정확하고 완전한 데이터 추출은 후속 단계의 성공을 좌우
- "Garbage In, Garbage Out" 원칙: 낮은 품질의 입력 데이터는 낮은 품질의 결과로 이어짐

### 데이터 품질에 미치는 영향
- 텍스트 추출의 정확성과 완전성
- 문서 구조의 보존
- 특수 문자, 인코딩, 다국어 지원
- 메타데이터 보존

---

## 3. 주요 파일 형식별 로더

### 텍스트 기반 문서
- **TextLoader**
    - 가장 단순한 형태의 로더, 텍스트파일 로드하는 로더
    - 인코딩 처리가 주요 고려사항
    - [텍스트 로더 예제](./01-TXT-Loader.ipynb)

- **PDF 로더**
    - 텍스트 추출 방식:
        - PyPDFLoader, PyMuPDFLoader,  **PDFPlumberLoader**, PyPDFium2Loader, PDFMinerLoader 등 활용
        - OCR(Optical Character Recognition) 통합 가능
    - 구조적 요소(제목, 목차, 페이지 번호) 보존 문제
    - 이미지, 표, 그래프 처리 방법
    - [PDF 로더 예제](./02-PDF-Loader.ipynb)

- **HPW 문서 로더 (hpw)**
    - LangChain에 integration 되지 않아 직접 구현한 `HWPLoader` 사용
    - langchain_teddynote.document_loaders의 HWPLoader 사용
    - 표, 이미지 등 임베디드 콘텐츠 처리 보완
    - [HWP 로더 예제](./03-HWP-Loader.ipynb)    

- **Word 문서 로더 (DOCX/DOC)**
    - Docx2txtLoader, UnstructuredWordDocumentLoader 등 사용
    - 서식 정보(볼드, 이탤릭 등) 처리 옵션
    - 표, 이미지 등 임베디드 콘텐츠 처리
    - [Word 로더 예제](./04-Word-Loader.ipynb)    

### 구조화된 데이터
- **CSV/Excel 로더**
    - UnstructuredExcelLoader를 사용하여 Microsoft Excel 파일 로드
    - pandas의 read_excel() 활용한 데이터 처리
    - 행/열 구조를 텍스트로 변환하는 전략
    - 수치 데이터와 텍스트 데이터의 혼합 처리
    - [CSV 로더 예제](./05-CSV-Loader.ipynb)    
    - [Excel 로더 예제](./06-Excel-Loader.ipynb)            

- **JSON/XML 로더**
    - JSONLoader를 통해 중첩된 구조 처리
    - 키-값 쌍의 의미를 보존하는 텍스트 변환
    - 스키마 정보 활용
    - [JSON 로더 예제](./07-JSON-Loader.ipynb)        

### 프레젠테이션 및 기타 형식
- **PowerPoint 로더 (PPTX/PPT)**
  - langchain_community.document_loaders의 UnstructuredPowerPointLoader 사용 
  - 슬라이드 구조 보존 문제
  - 노트와 본문 텍스트 처리
  - [PowerPoint 로더 예제](./08-PowerPoint-Loader.ipynb)       

- **LayoutAnalysis 로더**
    - 문서의 구조를 이해하고 요소 간 관계를 파악하여 보다 정확한 문서 분석
    - PDF, 이미지 등 다양한 형식의 문서에서 레이아웃 분석 수행
    - 문서의 구조적 요소(제목, 단락, 표, 이미지 등)를 자동으로 인식 및 추출
    - OCR 기능 지원 (선택적)
    - [LayoutAnalysis 로더 예제](./10-UpstageLayoutAnalysisLoader.ipynb)   

---

## 4. 웹 기반 데이터 소스 로더

### 웹 페이지 및 HTML 로더
- **기본 HTML 로더**
    - bs4 라이브러리 사용하여 웹 페이지 파싱, WebBaseLoader 사용
    - HTML 태그 제거 및 의미 있는 텍스트 추출
    - 메타 태그, 타이틀 등 중요 메타데이터 보존
    - [WebBase 로더 예제](./09-WebBase-Loader.ipynb)       

 
- **위키피디아 로더**
    - 위키 마크업 처리
    - 링크, 인용, 참조 정보 처리

- **소셜 미디어 API 로더**
    - Twitter, Reddit 등 소셜 데이터 활용
    - API 제한 및 인증 처리

---

## 5. 데이터베이스 및 API 로더

### 데이터베이스 커넥터
- **SQL 데이터베이스 로더**
    - PostgreSQL, MySQL, SQLite 등 연결
    - 쿼리 결과를 문서로 변환
    - 증분 로딩 전략

- **NoSQL 데이터베이스 로더**
    - MongoDB, Elasticsearch 등 연결
    - 문서 구조 보존

### API 통합
- **REST API 로더**
    - API 응답 처리 및 형식 변환
    - 페이지네이션 및 레이트 리밋 관리
    - 인증 및 보안

- **GraphQL 로더**
    - 쿼리 최적화
    - 중첩 데이터 구조 처리

---

## 6. 로더 선택 시 고려사항

### 데이터 소스 특성
- 데이터 형식 및 구조
- 데이터 크기 및 양
- 갱신 빈도 및 실시간성 요구사항

### 시스템 요구사항
- 처리 속도 및 효율성
- 메모리 사용량
- 병렬 처리 능력

### 통합 용이성
- 기존 시스템과의 호환성
- 필요한 의존성 및 라이브러리
- 유지보수 및 업데이트 용이성

### 기능적 요구사항
- 메타데이터 처리 능력
- 비텍스트 콘텐츠 처리
- 다국어 지원

---

## 7. 효율적인 도큐먼트 로딩 전략

### 대용량 데이터 처리
- **청크 기반 로딩**
    - 메모리 효율적인 처리
    - 스트리밍 접근법

- **병렬 처리**
    - 멀티프로세싱 vs 멀티스레딩
    - 작업 큐 및 풀 관리

### 증분 로딩
- 변경된 문서만 업데이트
- 타임스탬프 및 해시 기반 변경 탐지

### 에러 처리 및 재시도
- 예외 처리 전략
- 로깅 및 모니터링
- 자동 재시도 메커니즘

---

## 8. 도큐먼트 메타데이터 관리

### 주요 메타데이터 필드
- **소스 정보**: 파일 경로, 파일명, URL, 데이터베이스 테이블 등
- **시간 정보**: 생성일, 수정일, 로딩일
- **저자 및 권한 정보**
- **버전 정보**
- **컨텍스트 정보**: 카테고리, 태그, 분류 등

### 메타데이터 추가 방법
- 자동 태깅 및 분류
- 엔티티 추출 및 연결
- 요약 정보 생성

### 메타데이터 활용
- 검색 필터링, 콘텐츠 선별 및 필터링 
- 결과 랭킹 및 관련성 계산

---

**참고 자료:**
- [실습코드] : [LangChain 한국어 튜토리얼](https://github.com/teddylee777/langchain-kr) 
- LangChain 문서: [Document Loaders](https://python.langchain.com/en/latest/modules/indexes/document_loaders.html)
- HuggingFace의 Document Loaders 가이드