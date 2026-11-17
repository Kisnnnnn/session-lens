"""Evidence-based local activity labels. Never execute or display recorded code."""
from __future__ import annotations

import ast
import json
import os
import re
import shlex
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

LABELS = {
    "read_files": ("查看文件与代码", "读取文件、目录或检索代码"),
    "tests": ("运行测试", "记录包含明确的测试启动操作；测试结果需另行核实"),
    "inspect_ui": ("检查界面或图片", "启动浏览器检查、截取界面或查看图片"),
    "git": ("检查代码改动", "读取 Git 状态、差异或提交记录"),
    "plugin": ("更新插件", "记录包含插件添加操作；不代表安装已经完成"),
    "edit_files": ("修改文件", "记录包含补丁应用或文件写入操作"),
    "local_records": ("读取本地记录", "记录包含本地 SQLite 连接操作；不推断查询结果"),
    "read_projects": ("查看项目列表", "读取可用项目的信息"),
    "delegate": ("分派协作任务", "分派或补充子 agent 任务；分工文字需由主聊天明确提供"),
    "operation_progress": ("跟进本地操作", "与已启动的本地操作交互；不推断任务结果"),
    "local": ("执行本地操作", "记录只说明运行了脚本，具体用途未识别"),
    "auxiliary": ("使用辅助工具", "记录未提供足够信息来识别具体用途"),
    "mixed": ("执行多项本地操作", "包含多个类别的本地操作"),
}
NOTES_SOURCE = "主聊天提供的分工说明"
_MAX_SCRIPT = 200000
_NOTES_LOCK = threading.Lock()


def _quoted_end(text, start):
    quote = text[start]
    cursor = start + 1
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
        elif text[cursor] == quote:
            return cursor + 1
        else:
            cursor += 1
    return len(text)


def _without_literals(text):
    """Mask strings/comments while preserving offsets for finite JS/Python scans."""
    output = list(text)
    cursor = 0
    while cursor < len(text):
        if text[cursor] in "\"'`":
            end = _quoted_end(text, cursor)
        elif text.startswith("//", cursor) or text[cursor] == "#":
            end = text.find("\n", cursor)
            end = len(text) if end < 0 else end
        elif text.startswith("/*", cursor):
            end = text.find("*/", cursor + 2)
            end = len(text) if end < 0 else end + 2
        else:
            cursor += 1
            continue
        for index in range(cursor, end):
            if text[index] != "\n":
                output[index] = " "
        cursor = end
    return "".join(output)


def _object_command(arguments):
    if isinstance(arguments, dict):
        value = arguments.get("cmd")
        return value if isinstance(value, str) else None
    if not isinstance(arguments, str):
        return None
    try:
        value = json.loads(arguments)
    except (ValueError, TypeError):
        return None
    return _object_command(value)


def _js_command(arguments):
    masked = _without_literals(arguments)
    match = re.search(r"\bcmd\s*:", masked)
    if not match:
        # Quoted JSON keys are masked; JSON parsing handles those safely.
        return _object_command(arguments)
    start = match.end()
    while start < len(arguments) and arguments[start].isspace():
        start += 1
    if start >= len(arguments) or arguments[start] not in "\"'`":
        return None
    end = _quoted_end(arguments, start)
    literal = arguments[start:end]
    if literal.startswith("`"):
        return literal[1:-1] if literal.endswith("`") and "${" not in literal else None
    try:
        value = ast.literal_eval(literal)
        return value if isinstance(value, str) else None
    except (ValueError, SyntaxError):
        return None


def _script_categories(script):
    clean = _without_literals(script)
    categories = set()
    if re.search(r"\bsqlite3\s*\.\s*connect\s*\(", clean):
        categories.add("local_records")
    if re.search(r"\.\s*(write_text|write_bytes)\s*\(", clean):
        categories.add("edit_files")
    if re.search(r"\.\s*(read_text|read_bytes|readFile|readFileSync|readdir|readdirSync)\s*\(", clean):
        categories.add("read_files")
    if re.search(r"\b(chromium|firefox|webkit)\s*\.\s*launch\s*\(|\bpage\s*\.\s*screenshot\s*\(", clean):
        categories.add("inspect_ui")
    return categories


def _command_categories(command):
    if not isinstance(command, str) or len(command) > _MAX_SCRIPT:
        return {"local"}
    lines = command.strip().splitlines()
    if len(lines) > 1 and not re.match(r"^(?:\S*/)?(?:python[\d.]*|node)\b", lines[0].strip()):
        categories = set()
        for line in lines:
            if line.strip() and not line.lstrip().startswith("#"):
                categories.update(_command_categories(line))
        return categories or {"local"}
    categories = set()
    # Only command positions count. A quoted word such as echo 'pytest' does not.
    first_line = command.strip().splitlines()[0] if command.strip() else ""
    try:
        lexer = shlex.shlex(first_line, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        return {"local"}
    commands = [[]]
    for word in words:
        if word and all(character in ";&|" for character in word):
            commands.append([])
        else:
            commands[-1].append(word)
    for tokens in commands:
        if not tokens:
            continue
        executable = Path(tokens[0]).name
        args = tokens[1:]
        if executable in ("cat", "sed", "rg", "ls", "head", "tail", "find", "wc", "less"):
            categories.add("read_files")
        elif executable in ("pytest", "py.test") or (re.fullmatch(r"python[\d.]*", executable) and any(args[index:index + 2] in (["-m", "unittest"], ["-m", "pytest"]) for index in range(len(args)))):
            categories.add("tests")
        elif executable in ("npm", "pnpm", "yarn") and (args[:1] == ["test"] or (args[:1] == ["run"] and len(args) > 1 and (args[1] == "test" or args[1].startswith("test:")))):
            categories.add("tests")
        elif executable == "node" and args and re.fullmatch(r"qa-[\w.-]+\.cjs", Path(args[0]).name):
            categories.add("tests")
        elif executable == "git":
            cursor = 0
            while cursor < len(args) and args[cursor].startswith("-"):
                cursor += 2 if args[cursor] in ("-C", "-c", "--git-dir", "--work-tree") else 1
            categories.add("git" if cursor < len(args) and args[cursor] in ("status", "diff", "log", "show", "rev-parse", "ls-files") else "local")
        elif executable == "codex" and args[:2] == ["plugin", "add"]:
            categories.add("plugin")
        elif executable == "apply_patch":
            categories.add("edit_files")
        elif re.fullmatch(r"python[\d.]*|node", executable):
            categories.update(_script_categories(command) or {"local"})
        else:
            categories.add("local")
    return categories or {"local"}


def _known_tool(name, arguments=None):
    short = re.split(r"__|\.", name)[-1]
    if short in ("spawn_agent", "followup_task"):
        # Deliberately do not inspect delegate messages, encrypted or otherwise.
        return {"delegate"}
    if short == "exec_command":
        return _command_categories(_object_command(arguments))
    if short == "apply_patch":
        return {"edit_files"}
    if short in ("view_image", "screenshot"):
        return {"inspect_ui"}
    if short == "list_projects":
        return {"read_projects"}
    if short in ("read_file", "read_files", "list_files", "search_files"):
        return {"read_files"}
    if short in ("write_stdin", "wait"):
        return {"operation_progress"}
    return {"auxiliary"}


def _wrapped_categories(script):
    if not isinstance(script, str) or len(script) > _MAX_SCRIPT:
        return {"local"}
    clean = _without_literals(script)
    calls = list(re.finditer(r"\btools\.([A-Za-z_]\w*)\s*\(", clean))
    categories = set()
    for match in calls:
        name = match.group(1)
        depth, end = 1, match.end()
        while end < len(clean) and depth:
            if clean[end] == "(":
                depth += 1
            elif clean[end] == ")":
                depth -= 1
            end += 1
        arguments = script[match.end():end - 1] if depth == 0 else ""
        if name == "exec_command":
            categories.update(_command_categories(_js_command(arguments)))
        else:
            categories.update(_known_tool(name))
    return categories or _script_categories(script) or {"local"}


def _describe(categories):
    values = sorted({value for value in categories if isinstance(value, str) and value in LABELS}) or ["auxiliary"]
    category = values[0] if len(values) == 1 else "mixed"
    title, detail = LABELS[category]
    if len(values) > 1:
        known = [value for value in values if value not in ("local", "auxiliary", "operation_progress", "mixed")]
        if known:
            title = "、".join(LABELS[value][0] for value in known[:2])
            if len(values) > min(2, len(known)):
                title += "等操作"
            detail = "识别到：" + "、".join(LABELS[value][0] for value in known)
        else:
            detail = "包含用途未识别的本地操作或辅助工具调用"
        if "local" in values or "auxiliary" in values:
            detail += "；另有操作用途未识别"
        if "operation_progress" in values:
            detail += "；并查询已启动操作的进度"
        detail += "。操作记录本身不能证明任务结果"
    return category, title, detail, values


def classify_activity(payload) -> dict:
    payload = payload if isinstance(payload, dict) else {}
    name = payload.get("name") if isinstance(payload.get("name"), str) else ""
    namespace = payload.get("namespace") if isinstance(payload.get("namespace"), str) else ""
    tool_name = (namespace + "." + name if namespace and not name.startswith(namespace + ".") else name)[:160]
    short = re.split(r"__|\.", name)[-1]
    if short == "exec":
        categories = _wrapped_categories(payload.get("input"))
    else:
        categories = _known_tool(tool_name, payload.get("arguments"))
    category, title, detail, values = _describe(categories)
    return {"category": category, "categories": values, "title": title, "detail": detail, "toolName": tool_name}


def _time(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


def group_activities(events) -> list:
    """Group only adjacent tool operations with matching category and status."""
    if not isinstance(events, list):
        return []
    ordered = [dict(event) for event in events if isinstance(event, dict)]
    ordered.sort(key=lambda event: _time(event.get("at")) or 0, reverse=True)
    result = []
    for event in ordered:
        if event.get("kind") != "tool":
            prior = result[-1] if result else None
            current_time = _time(event.get("at"))
            first_time = _time(prior.get("firstAt", prior.get("at"))) if prior else None
            latest_time = _time(prior.get("at")) if prior else None
            if (event.get("kind") in ("connection", "timeout") and prior and prior.get("kind") == event["kind"]
                    and current_time is not None and first_time is not None and latest_time is not None
                    and 0 <= first_time - current_time <= 180 and 0 <= latest_time - current_time <= 600):
                prior["count"] = prior.get("count", 1) + 1
                prior["firstAt"] = event["at"]
                continue
            result.append(event)
            continue
        category = event.get("category") if event.get("category") in LABELS else "auxiliary"
        categories = event.get("categories") if isinstance(event.get("categories"), list) else [category]
        category, title, detail, categories = _describe(categories)
        status = event.get("status") if event.get("status") in ("running", "returned", "failed", "unknown") else "unknown"
        at = event.get("at")
        tool_name = event.get("toolName") if isinstance(event.get("toolName"), str) else ""
        call_id = event.get("callId") if isinstance(event.get("callId"), str) else None
        prior = result[-1] if result else None
        current_time = _time(at)
        adjacent_time = _time(prior.get("firstAt")) if prior else None
        group_time = _time(prior.get("at")) if prior else None
        merge = bool(prior and prior.get("kind") == "tool" and prior.get("category") == category and prior.get("status") == status and current_time is not None and adjacent_time is not None and group_time is not None and 0 <= adjacent_time - current_time <= 180 and 0 <= group_time - current_time <= 600)
        if merge:
            prior["count"] += 1
            prior["firstAt"] = at
            prior["statusCounts"][status] += 1
            _, prior["title"], prior["detail"], prior["categories"] = _describe(prior["categories"] + categories)
            if tool_name and tool_name not in prior["sourceTools"]:
                prior["sourceTools"].append(tool_name[:160])
            if call_id and call_id not in prior["callIds"]:
                prior["callIds"].append(call_id[:200])
        else:
            result.append({"at": at, "firstAt": at, "kind": "tool", "category": category, "categories": categories, "title": title, "detail": detail, "details": [detail], "toolName": tool_name[:160], "sourceTools": [tool_name[:160]] if tool_name else [], "callId": call_id, "callIds": [call_id[:200]] if call_id else [], "status": status, "statusCounts": {status: 1}, "count": 1})
    return result


def _identifier(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 200 or ":" in value or any(ord(character) < 32 for character in value):
        raise ValueError(label + "无效")
    return value.strip()


def _note_text(value, label, limit, required=False):
    if not isinstance(value, str):
        raise ValueError(label + "必须是文字")
    value = " ".join(value.split())
    if (required and not value) or len(value) > limit:
        raise ValueError(label + "长度必须在允许范围内")
    return value


def read_agent_notes(path) -> dict:
    """Return a sourceThreadId:agentId index, degrading malformed files to {}."""
    try:
        file = Path(path)
        if file.stat().st_size > 2_000_000:
            return {}
        data = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        notes = {}
        for entry in list(data.values())[:5000]:
            if not isinstance(entry, dict):
                continue
            try:
                source_id = _identifier(entry.get("sourceThreadId"), "原会话 ID")
                agent_id = _identifier(entry.get("agentId"), "Agent ID")
                label = _note_text(entry.get("label"), "分工名称", 40, required=True)
                description = _note_text(entry.get("description"), "分工说明", 160)
            except ValueError:
                continue
            if entry.get("source") != NOTES_SOURCE:
                continue
            notes[source_id + ":" + agent_id] = {"sourceThreadId": source_id, "agentId": agent_id, "label": label, "description": description, "source": NOTES_SOURCE, "updatedAt": entry.get("updatedAt") if isinstance(entry.get("updatedAt"), str) else None}
        return notes
    except (OSError, ValueError, TypeError):
        return {}


def write_agent_note(path, source_thread_id, agent_id, label, description) -> dict:
    source_id = _identifier(source_thread_id, "原会话 ID")
    child_id = _identifier(agent_id, "Agent ID")
    entry = {"sourceThreadId": source_id, "agentId": child_id, "label": _note_text(label, "分工名称", 40, required=True), "description": _note_text(description, "分工说明", 160), "source": NOTES_SOURCE, "updatedAt": datetime.now(timezone.utc).isoformat()}
    path = Path(path)
    with _NOTES_LOCK:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        notes = read_agent_notes(path)
        notes[source_id + ":" + child_id] = entry
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".agent-notes-", delete=False) as output:
                temporary = Path(output.name)
                json.dump(notes, output, ensure_ascii=False, indent=2)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return entry
