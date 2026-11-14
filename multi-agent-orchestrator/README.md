# Multi-Agent Orchestrator (Claude Code × OpenAI Codex)

하나의 개발 요청을 입력하면 **Claude Planner**가 작업을 DAG로 분해하고, **Claude Code / Codex CLI 에이전트**가
각자 독립된 **Git worktree**에서 병렬로 개발한 뒤, 검증된 커밋만 **통합 브랜치**(`orchestrator/<run-id>`)에
cherry-pick 하고, **통합 테스트 + Claude↔Codex 교차 리뷰 + 자동 수정 루프**를 거쳐 실행 보고서를 만드는
CLI 오케스트레이터입니다.

> main 브랜치는 절대 자동으로 변경되지 않으며, push도 하지 않습니다. 최종 병합은 사용자가 직접 합니다.

```
사용자 요청 ─▶ Planner(claude -p, 읽기 전용, JSON 스키마) ─▶ 계획 검증(ID·범위·DAG 순환 검사)
        ─▶ DAG 스케줄러 (max_parallel_workers)
             ├─ task A: codex exec  (worktree A, workspace-write sandbox)
             ├─ task B: claude -p   (worktree B, --restricted + dontAsk + allowlist)
             └─ task C: (A 완료 후) 통합 브랜치 HEAD에서 시작
        ─▶ 결과 검증(성공 여부·변경 존재·허용 범위·금지 경로·삭제) ─▶ 검증된 파일만 커밋
        ─▶ 통합 worktree에서 cherry-pick (충돌 시 --abort 후 CONFLICT 보고)
        ─▶ 통합 테스트 + 교차 리뷰(동일 커밋, 구조화 JSON) ─▶ 문제 시 수정 작업 생성(최대 2라운드)
        ─▶ report.md
```

## 1. 설치 (Windows 10/11)

PowerShell 에서:

```powershell
# 1) Python 3.11+ 와 Git
winget install --id Python.Python.3.12 -e
winget install --id Git.Git -e
git config --global core.longpaths true        # worktree 경로가 길어질 수 있음

# 2) Node.js (npm으로 설치하는 CLI용)
winget install --id OpenJS.NodeJS.LTS -e

# 3) Claude Code CLI (둘 중 하나) - 설치 후 `claude` 를 한 번 실행해 직접 로그인
irm https://claude.ai/install.ps1 | iex
#   또는: npm install -g @anthropic-ai/claude-code

# 4) Codex CLI - 설치 후 직접 로그인
npm install -g @openai/codex
codex login

# 5) 오케스트레이터
cd multi-agent-orchestrator
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .            # `orchestrator` 명령 등록 (또는: pip install -r requirements.txt 후 python main.py ...)

# 6) 환경 점검 (1단계)
orchestrator doctor
```

인증(로그인, API 키 설정)은 보안상 오케스트레이터가 하지 않습니다. 사용자가 직접 수행하세요.
`doctor`는 Python/Git/worktree/node/claude/codex 설치 여부와 Codex 로그인 상태(`codex login status`)를 확인합니다.

macOS / Linux 도 동일하게 동작합니다 (`python3 -m venv .venv && . .venv/bin/activate`).

## 2. 사용법

```bash
# 개발 요청 실행 (대상 저장소는 git 저장소여야 함; 기본은 현재 디렉터리, 현재 브랜치 기준)
orchestrator run "주식 자동매매 프로그램 개발" --repo C:\work\my-project

# 계획만 먼저 만들고 검토 후 실행
orchestrator run "요청" --repo . --plan-only
orchestrator plan            # 계획 확인
orchestrator resume          # 계획대로 실행

# AI planner 대신 직접 작성한 계획(JSON) 사용
orchestrator run "요청" --repo . --plan-file my-plan.json

orchestrator status [RUN_ID]          # 실행 상태 / 작업별 상태 / 활성 에이전트 (RUN_ID 생략 시 최근 실행)
orchestrator status --all             # 모든 실행 목록 (= orchestrator list)
orchestrator plan [RUN_ID] [--json]   # 작업 계획
orchestrator logs [RUN_ID] [--task T] [-f]          # 오케스트레이터 이벤트 로그 (follow 가능)
orchestrator logs [RUN_ID] --task T --agent         # 해당 작업 에이전트의 출력 로그
orchestrator stop [RUN_ID]            # 다른 터미널에서 중단 요청 → 에이전트 프로세스 트리 종료
orchestrator cancel RUN_ID TASK_ID    # 개별 작업 취소
orchestrator resume [RUN_ID] [--retry-failed]       # 중단/비정상 종료된 실행 재개
orchestrator report [RUN_ID] [--write]              # 최종 보고서
```

실행 중 `Ctrl+C` 를 누르면 실행 중인 에이전트를 모두 종료하고 상태를 `STOPPED`로 남깁니다 (`resume` 가능).

결과물 확인과 병합(수동):

```bash
git log --oneline main..orchestrator/<run-id>
git diff main...orchestrator/<run-id>
git switch main
git merge --no-ff orchestrator/<run-id>
```

### 상태 값
- 작업: `PENDING → READY → RUNNING → VALIDATING → COMPLETED`, 그 외 `FAILED`, `CONFLICT`, `BLOCKED`(선행 작업 실패), `CANCELLED`
- 실행: `PLANNING`, `PLANNED`, `RUNNING`, `VERIFYING`, `REPAIRING`, `SUCCEEDED`, `UNVERIFIED`(테스트 명령 없음),
  `NEEDS_ATTENTION`(실패/충돌/테스트 실패/리뷰 미완료 잔존), `FAILED`(계획 실패 등), `STOPPED`
- 프로세스가 죽은 채 `RUNNING`인 실행은 `status`에서 `INTERRUPTED`로 표시됩니다.

**테스트가 실패하거나 실행되지 않았는데 SUCCEEDED로 보고하는 경우는 없습니다.**

## 3. 설정 (`config.yaml`)

모든 키는 선택 사항이며 기본값은 `orchestrator/config.py` 의 `DEFAULT_CONFIG`에 있습니다. 주요 항목:

| 키 | 기본값 | 설명 |
|---|---|---|
| `agents.planner` | `claude` | 계획 담당 |
| `agents.workers` | codex/backend, claude/frontend, codex/testing | planner에 주는 역할 힌트 (역할은 작업별로 동적 배정) |
| `agents.reviewers` | `[claude, codex]` | 교차 리뷰어 |
| `agents.fallback_on_unavailable` | `true` | CLI 미설치/미로그인 시 다른 제공자로 재배정 |
| `providers.<p>.command` | `null` | 실행 파일. 예: `["C:/Users/me/.local/bin/claude.exe"]` |
| `providers.<p>.model` / `planner_model` / `reviewer_model` | `null` | 모델 지정 (작업별 `model` 도 가능) |
| `providers.claude.worker_allowed_tools` | Read/Write/Edit/Glob/Grep, pytest·npm test 등 | 사전 허용 도구(그 외 모두 거부) |
| `providers.codex.worker_sandbox` | `workspace-write` | 리뷰/계획은 `read-only` |
| `limits.max_parallel_workers` / `max_parallel_reviewers` | 3 / 2 | 동시 실행 수 |
| `limits.max_repair_rounds` | 2 | 자동 수정 최대 횟수 |
| `limits.max_task_retries` | 1 | 실패한 작업의 재시도(오류 피드백 포함) |
| `limits.task_timeout_seconds` | 1800 | 에이전트별 시간 제한 |
| `limits.max_total_cost_usd` | `null` | 누적 비용 한도 (Claude가 보고하는 비용 기준) |
| `validation.test_command` | `auto` | 인자 배열. `auto`는 `npm test` / `python -m pytest` 감지 |
| `validation.scope_violation` | `fail` | 담당 범위 밖 변경 시 거부 |
| `validation.allow_deletions` | `false` | 기존 파일 삭제 거부 |
| `validation.forbidden_paths` | `.git/**`, `.env`, `*.pem`, workflows … | 절대 변경 불가 |
| `review.required` | `true` | 리뷰어 하나라도 실행 불가/실패하면 `NEEDS_ATTENTION` |
| `storage.data_dir` | `data` | SQLite DB, 로그, worktree 위치 (config 파일 기준 상대경로) |

Claude만 설치된 환경이라면 `agents.reviewers: [claude]` 로 두거나 `review.required: false` 로 설정하세요
(이 경우 "교차" 리뷰가 아니라는 점이 보고서에 드러납니다).

## 4. 보안 모델과 안전장치

| 요구사항 | 구현 |
|---|---|
| main 무단 변경 금지 | 사용자의 checkout은 전혀 건드리지 않음(checkout/reset 없음). 모든 쓰기는 `data/worktrees/` 아래 전용 worktree에서. `Git` 래퍼가 보호 브랜치(main/master/develop/…)와 오케스트레이터가 만들지 않은 브랜치의 변경을 코드 수준에서 거부. 종료 시 기준 브랜치 SHA 변화를 검사·보고 |
| 자동 push 금지 | push 코드 자체가 없음. `auto_push: true` 설정은 거부 |
| force push / reset --hard 금지 | `Git.run` 이 `push`, `reset`, `clean`, `checkout`, `--force`, `--hard`, `-D` 등을 허용 목록 밖으로 차단 |
| 사용자 파일 무단 삭제 금지 | 기준 커밋에 있던 파일 삭제 시 검증 실패(`allow_deletions: false`). worktree 삭제는 `data/worktrees` 내부만 허용 |
| CLI 무제한 권한 금지 | Claude: `--restricted`(사용자 설정 무시, 코드 실행 도구 제거, 파일 도구를 작업 디렉터리로 제한) + `--tools` + `--permission-mode dontAsk` + `--allowedTools` 허용 목록 + `--disallowedTools`. `--dangerously-skip-permissions` 미사용. Codex: `--sandbox workspace-write`(리뷰/계획은 `read-only`), `approval_policy="never"`, `--dangerously-bypass-…` 미사용 |
| 비밀정보 로그 노출 방지 | 모든 로그/DB 기록 전 `Redactor`가 API 키 패턴(sk-, sk-ant-, ghp_, Bearer …)과 `*KEY/TOKEN/SECRET*` 환경변수 값을 마스킹. 프롬프트는 명령행이 아닌 stdin으로 전달 |
| 작업 디렉터리 제한 | 에이전트별 worktree + CLI 권한 제한 + 변경 범위(glob) 검증 |
| 동시 실행/시간 제한 | 세마포어 기반 동시 실행 수, 에이전트별 타임아웃 → 프로세스 트리 강제 종료 |
| 비용 관리 | 에이전트별 비용/토큰 사용량을 SQLite에 기록, `max_total_cost_usd`, Claude `--max-budget-usd` |
| 쉘 문자열 금지 | 모든 하위 프로세스는 인자 배열로 실행. `test_command` 문자열 설정은 거부. Windows npm `.cmd` 셰임은 `node <script.js>`로 해석해 cmd.exe 재해석을 피함 |
| Windows 하위 프로세스 종료 | `CREATE_NEW_PROCESS_GROUP` + `taskkill /T /F`, POSIX는 새 세션 + `killpg` |

실측으로 확인한 사항 (Claude Code 2.1.325): `--restricted` 없이 `Write`만 허용하면 작업 폴더 밖 쓰기가 **가능했고**,
`--restricted --tools … --permission-mode dontAsk --allowedTools …` 조합에서는 작업 폴더 밖 쓰기와 허용 목록 밖 Bash가
**거부됨**을 확인했습니다. 그래서 이 조합을 기본값으로 사용합니다.

남아 있는 한계: worktree는 보안 격리가 아닙니다. 허용된 테스트 명령(예: `python -m pytest`)은 임의 코드를 실행할 수
있으므로, 신뢰할 수 없는 저장소에서는 `worker_allowed_tools` 에서 테스트 실행 권한을 빼거나 VM/컨테이너에서 실행하세요.
Codex `workspace-write` 샌드박스의 Windows 지원 수준은 Codex 버전에 따라 다릅니다.

## 5. 동작 상세

- **작업 분해**: planner 출력은 JSON 스키마(`--json-schema` / `--output-schema`)로 강제되고, ID 정규화, 중복·미존재 의존성,
  순환(위상 정렬), 안전하지 않은 경로, 작업 수 제한, 병렬 작업 간 범위 겹침(경고)을 검사합니다. 공통 인터페이스는
  `shared_contracts`로 모든 워커 프롬프트에 주입되고, 필요하면 "contracts" 작업이 먼저 실행되도록 planner에 지시합니다.
- **기준 커밋**: 독립 작업은 모두 동일한 기준 커밋에서, 의존 작업/수정 작업은 선행 작업이 반영된 통합 브랜치 HEAD에서 시작합니다.
- **검증 후 커밋**: 검증을 통과한 경로만 `git add -- <paths>` 로 커밋합니다(`__pycache__`, `node_modules` 등 생성물 제외).
  커밋에는 `Orchestrator-Task: <run>/<task>` 트레일러가 붙어 크래시 복구 시 통합 여부를 판별합니다.
- **통합**: 통합 브랜치 전용 worktree에서 `cherry-pick -x`. 충돌 시 `--abort` 하여 통합 브랜치를 그대로 두고
  충돌 파일을 기록(`CONFLICT`). 수정 라운드에서 해당 작업을 최신 통합 HEAD 위에서 충돌 정보와 함께 재실행합니다.
- **교차 리뷰**: 같은 통합 커밋을 기준으로 리뷰어들이 독립적으로 병렬 리뷰. 리뷰어 프롬프트는 "다른 제공자가 작성한 코드"를
  중점 검토하도록 지정합니다. 결과(verdict, 심각도, 파일, 라인, 권고)는 `reviews` 테이블에 저장됩니다.
  `critical`/`high` 지적은 수정 작업을 유발합니다. AI 리뷰만으로는 통과 처리하지 않으며 테스트 결과가 반드시 필요합니다.
- **수정 루프**: 테스트 실패/차단 리뷰/충돌 → 실패 출력과 지적 사항을 담은 `repair-rN` 작업 생성 → 관련 파일을 작성한
  제공자에게 배정 → 검증·통합 → 재테스트·재리뷰. `max_repair_rounds` 초과 시 `NEEDS_ATTENTION`으로 종료.
- **재개**: 모든 상태 전이는 즉시 SQLite(WAL)에 기록됩니다. `resume`은 남은 에이전트 프로세스(고아)를 종료하고,
  미완료 cherry-pick을 abort하고, 통합 브랜치의 트레일러로 완료 작업을 복구한 뒤, 중단된 작업만 다시 실행합니다.

데이터 위치(`data/`): `orchestrator.db`, `runs/<run-id>/logs/*.log`(정리된 로그), `*.raw.jsonl`(원본 이벤트),
`runs/<run-id>/diffs/`, `runs/<run-id>/tests/`, `runs/<run-id>/report.md`, `worktrees/<run-id>/…`(실행 중에만 존재).

## 6. 프로젝트 구조

```
multi-agent-orchestrator/
├── main.py                    # python main.py <command>
├── config.yaml / requirements.txt / pyproject.toml
├── orchestrator/
│   ├── cli.py                 # 터미널 UI (엔진과 분리 → 향후 웹 대시보드는 engine + SQLite만 사용)
│   ├── engine.py              # 실행 엔진 (계획→DAG→통합→검증→수정→보고)
│   ├── planner.py             # 계획 스키마/프롬프트/검증(DAG)
│   ├── scheduler.py           # DAG 스케줄러
│   ├── agent_runner.py        # 비동기 하위 프로세스 실행, 타임아웃, 취소, 실시간 로그
│   ├── worktree_manager.py    # Git 안전 래퍼 + worktree 관리
│   ├── integrator.py          # cherry-pick 통합 / 충돌 처리
│   ├── review_manager.py      # 교차 리뷰
│   ├── result_validator.py    # 결과 검증
│   ├── test_runner.py         # 테스트 실행
│   ├── state_store.py         # SQLite
│   ├── report.py / doctor.py / security.py / procutil.py / config.py / models.py
│   ├── agents/                # claude_adapter.py, codex_adapter.py, fake_agent.py(테스트용 CLI 대역)
│   └── prompts/               # planner.md, worker.md, reviewer.md, repair.md
├── tests/                     # test_e2e_scenarios.py, test_units.py, test_live.py
└── data/                      # 실행 데이터 (git 제외)
```

## 7. 테스트

```bash
pip install -r requirements.txt
python -m pytest            # 결정적 테스트 (실제 git·worktree·하위 프로세스 사용, AI 호출 없음)
ORCH_LIVE_MODEL=claude-sonnet-5-5 python -m pytest -m live -s tests/test_live.py   # 실제 CLI 호출 (비용 발생)
```

결정적 테스트에서는 `orchestrator/agents/fake_agent.py` 가 실제 `claude -p --output-format stream-json` /
`codex exec --json -o …` 와 **동일한 인자와 출력 형식**을 받아 처리하므로, 어댑터의 명령 생성·파싱, 프로세스 관리,
worktree, cherry-pick, 상태 저장이 모두 실제 코드 경로로 검증됩니다.

## 8. 문제 해결

| 증상 | 해결 |
|---|---|
| `doctor`에서 codex `not authenticated` | `codex login` 실행. 그동안은 codex 작업이 claude로 재배정됨(`fallback_on_unavailable`) |
| `... is a batch shim that could not be resolved` | `providers.codex.command: ["node", "C:/Users/<you>/AppData/Roaming/npm/node_modules/@openai/codex/bin/codex.js"]` 처럼 지정 |
| `Filename too long` (Windows) | `git config --global core.longpaths true`, `storage.data_dir` 를 `C:/orch-data` 같이 짧은 경로로 |
| 작업이 `FAILED: changes outside the allowed scope` | planner가 정한 `files` 범위를 확인. 필요하면 `--plan-file` 로 범위를 수정하거나 `scope_violation: warn` |
| `tool call(s) were denied by the permission policy` | 에이전트가 허용 목록 밖 명령을 시도함. 필요한 명령만 `worker_allowed_tools` 에 추가 (예: `Bash(npm run lint:*)`) |
| 실행이 `INTERRUPTED` 로 표시 | 오케스트레이터 프로세스가 비정상 종료됨 → `orchestrator resume` |
| `NEEDS_ATTENTION` | `orchestrator report` 로 실패 원인 확인 → 수정 후 `resume --retry-failed` 또는 통합 브랜치에서 직접 수정 |
| 테스트가 통합 환경에서만 실패 | 테스트 의존성(가상환경 패키지)이 오케스트레이터를 실행한 Python 환경에 설치되어 있는지 확인. `test_command` 를 venv의 python 경로로 지정 가능 |
| 한글이 깨짐 | Windows Terminal 사용 권장. 오케스트레이터는 UTF-8로 출력·저장하며 git `core.quotePath=false` 로 한글 파일명을 처리 |

## 9. 현재 한계 / 다음 단계

- 웹 대시보드(4단계)는 미구현. 엔진은 UI와 분리되어 있고 모든 상태가 SQLite에 있으므로 읽기 전용 대시보드를 바로 붙일 수 있습니다.
- Codex 실제 호출은 이 저장소 개발 환경에서 인증/네트워크가 없어 검증하지 못했습니다(인자 호환성은 codex-cli 0.169.0 실물로 확인).
- 비용: Claude는 CLI가 보고하는 USD 비용, Codex는 토큰 사용량만 기록합니다.
- 충돌은 자동 해결하지 않고, 해당 작업을 최신 통합 결과 위에서 재실행하는 방식으로만 처리합니다.
