## 아래 내용 VS CODE 에 설정
- extensions 클릭 > black formatter 설치
- CTRL + Shift + P 클릭 (Windows)
- User Settings(Json) 선택
- 아래 내용 복사하여 저장 


{
    "editor.fontVariations": false, // 글꼴 스타일 변형을 사용하지 않습니다.
    "editor.defaultFormatter": "ms-python.black-formatter",  // 에디터의 기본 코드 포맷터로 Black을 지정합니다.
    "editor.autoClosingBrackets": "never",  // 자동 괄호 닫기 기능을 사용하지 않습니다.
    "editor.autoClosingQuotes": "never",  // 자동 인용 부호 닫기 기능을 사용하지 않습니다.
    "editor.inlineSuggest.enabled": true,  // 인라인 자동 완성을 활성화합니다.
    "editor.parameterHints.enabled": true,  // 함수 호출 시 파라미터 정보를 보여주는 힌트를 활성화합니다.
    "editor.quickSuggestions": {
        "other": true,  // 기타 텍스트에서는 제안을 활성화합니다.
        "comments": false,  // 주석에서는 제안을 비활성화합니다.
        "strings": false  // 문자열 내에서는 제안을 비활성화합니다.
    },
    "editor.quickSuggestionsDelay": 10,  // 빠른 제안이 나타나기까지의 지연 시간(밀리초)을 설정합니다.
    "editor.suggestOnTriggerCharacters": true,  // 트리거 문자 입력 시 자동으로 제안을 표시합니다.
    "editor.suggest.localityBonus": true,  // 커서 근처에 나타나는 단어를 우선적으로 제안합니다.
    "editor.wordBasedSuggestions": "matchingDocuments",  // 문서 내 매칭된 단어 기반의 제안을 활성화합니다.
    "editor.acceptSuggestionOnCommitCharacter": true,  // 커밋 문자에서 제안을 수락합니다.
    "editor.acceptSuggestionOnEnter": "on",  // Enter 키를 누를 때 제안을 수락합니다.

    // 터미널 설정
    "terminal.integrated.fontSize": 12,  // 터미널 내부의 글꼴 크기를 20으로 설정합니다.
    "terminal.integrated.inheritEnv": false,  // 터미널이 시스템 환경 변수를 상속받지 않도록 설정합니다.

    // Git 설정
    "git.autofetch": true,  // Git 저장소를 자동으로 fetch하는 기능을 활성화합니다.

    // 마크다운 설정
    "markdown.preview.fontSize": 20,  // 마크다운 미리보기에서 사용할 글꼴 크기를 20으로 설정합니다.
    "markdown.styles": [  // 마크다운 스타일시트 URL을 설정합니다.
        "https://use.fontawesome.com/releases/v5.7.1/css/all.css",
        "markdown-stype.css"
    ],

    // 소스 코드 관리 설정
    "scm.inputFontSize": 12,  // 소스 코드 관리 입력 필드의 글꼴 크기를 20으로 설정합니다.

    // 워크벤치 설정
    "workbench.startupEditor": "none",  // VS Code 시작 시 에디터를 열지 않도록 설정합니다.

    // 보안 설정
    "security.workspace.trust.untrustedFiles": "open",  // 신뢰되지 않은 파일을 열 때 자동으로 엽니다.

    // 탐색기 설정
    "explorer.confirmDelete": false,  // 파일을 삭제할 때 확인 메시지를 표시하지 않습니다.
    "explorer.compactFolders": false,  // 탐색기에서 폴더를 컴팩트하게 표시하지 않습니다.
    "explorer.confirmDragAndDrop": false,  // 드래그 앤 드롭을 할 때 확인 메시지를 표시하지 않습니다.
    "explorer.confirmPasteNative": false,  // 네이티브 파일을 붙여넣을 때 확인 메시지를 표시하지 않습니다.

    // 파일 설정
    "files.autoSave": "afterDelay",  // 파일을 변경 후 일정 시간이 지나면 자동 저장합니다.

    // 브레드크럼 설정
    "breadcrumbs.enabled": false,  // 파일 경로 내비게이션을 비활성화합니다.

    // Jupyter 설정
    "jupyter.themeMatplotlibPlots": true,  // Jupyter에서 Matplotlib 플롯을 현재 테마에 맞게 색상화합니다.
    "jupyter.askForKernelRestart": false,  // Jupyter 커널 재시작을 요청하지 않습니다.
    "jupyter.widgetScriptSources": [  // Jupyter 위젯을 위한 스크립트 소스를 설정합니다.
        "jsdelivr.com",
        "unpkg.com"
    ],

    // 노트북 설정
    "notebook.output.wordWrap": true,  // 노트북 출력에서 자동 줄바꿈을 활성화합니다.
    "notebook.formatOnSave.enabled": true,  // 노트북 저장 시 자동으로 포맷합니다.
    "notebook.output.scrolling": true,  // 노트북 출력을 스크롤 가능하게 설정합니다.
    "notebook.lineNumbers": "on",  // 노트북에서 줄 번호를 표시합니다.

    // CSS 설정
    "css.lint.emptyRules": "ignore",  // CSS 린트에서 빈 규칙을 무시합니다.

    // 언어별 설정 (Python)
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnType": true,
        "editor.formatOnSave": true,
        "editor.parameterHints.enabled": true,
    }
}