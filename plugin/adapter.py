"""Read-only, best-effort adapters for locally persisted Codex sessions."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
import re
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

from activity import classify_activity, group_activities, read_agent_notes


def iso(value=None):
    if value is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        if isinstance(value, (int, float)):
            if value > 10**11:
                value /= 1000
            return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except (ValueError, TypeError, OverflowError):
        return None


def epoch(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError, AttributeError):
        return 0


def columns(connection, table):
    return {row[1] for row in connection.execute('PRAGMA table_info("%s")' % table)}


@contextmanager
def readonly(path):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.3)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        yield connection
    finally:
        connection.close()


class SessionReader:
    """Take bounded snapshots without modifying Codex data or reading credentials."""

    def __init__(self, codex_home: Path | None = None):
        self.home = Path(codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser().resolve()
        self._cache = {}
        self._project_state = (None, {})
        self._lock = threading.Lock()

    def snapshot(self):
        with self._lock:
            return self._snapshot()

    def _snapshot(self):
        sampled = iso()
        result = {"sampledAt": sampled, "source": {"kind": "local", "status": "unavailable", "message": "未找到可读取的本地会话数据库"}, "sessions": [], "workspaces": []}
        database = self.home / "state_5.sqlite"
        if not database.is_file():
            return result
        try:
            with readonly(database) as connection:
                available = columns(connection, "threads")
                if "id" not in available:
                    result["source"]["message"] = "会话数据库结构暂不支持"
                    return result
                desired = ["id", "title", "rollout_path", "cwd", "project_id", "updated_at", "updated_at_ms", "created_at", "model", "reasoning_effort", "agent_role", "agent_path", "attached_thread_parent_id", "history_mode", "archived"]
                selected = [name for name in desired if name in available]
                order = "updated_at" if "updated_at" in available else "id"
                query = 'SELECT ' + ','.join('"%s"' % name for name in selected) + ' FROM threads'
                if "archived" in available:
                    query += " WHERE archived=0"
                rows = [dict(row) for row in connection.execute(query + ' ORDER BY "%s" DESC LIMIT 240' % order)]
                edges = {}
                edge_columns = columns(connection, "thread_spawn_edges")
                if {"parent_thread_id", "child_thread_id"} <= edge_columns:
                    edges = {row[1]: row[0] for row in connection.execute("SELECT parent_thread_id, child_thread_id FROM thread_spawn_edges")}
                projects, assignments, aliases, projectless = self._projects(connection)
        except (sqlite3.Error, OSError):
            result["source"]["message"] = "本地会话数据暂不可读，稍后可重试"
            return result
        for row in rows:
            row["_parent"] = row.get("attached_thread_parent_id") or edges.get(row["id"])
            project_id = row.get("project_id")
            if not project_id and row["id"] not in projectless:
                project_id = assignments.get(row["id"])
            mapped_id = aliases.get(project_id, project_id)
            project_id = mapped_id if mapped_id in projects else project_id
            project = projects.get(project_id)
            row["_project_id"] = project_id if project else None
            row["_project_name"] = project["name"] if project else "未归属项目"
        roots = [row for row in rows if not row["_parent"]][:40]
        current = str(Path.cwd())
        current_rows = [row for row in rows if not row["_parent"] and row.get("cwd") == current]
        if current_rows and all(row["id"] != current_rows[0]["id"] for row in roots):
            roots = roots[:39] + current_rows[:1]
        root_ids = {row["id"] for row in roots}
        wanted = roots + [row for row in rows if row["_parent"] in root_ids]
        logs = self._logs({row["id"] for row in wanted})
        sessions = {row["id"]: self._session(row, sampled, logs.get(row["id"], [])) for row in wanted}
        notes = read_agent_notes(self.home / "artifacts" / "codex-session-helper" / "agent-notes.json")
        visible_paths = {str(Path(row["rollout_path"]).expanduser().resolve()) for row in wanted if row.get("rollout_path")}
        self._cache = {path: state for path, state in self._cache.items() if path in visible_paths}
        for session in sessions.values():
            parent = sessions.get(session["parentId"])
            if parent:
                note = notes.get(session["parentId"] + ":" + session["id"], {})
                parent["agents"].append({"id": session["id"], "role": session["agentRole"], "displayName": note.get("label"), "description": note.get("description"), "noteSource": note.get("source"), "status": session["status"]["code"], "label": session["status"]["label"], "lastEventAt": session["lastEventAt"]})
        result["sessions"] = [sessions[row["id"]] for row in roots]
        groups = {project_id: {"key": "project:" + project_id, "path": project["roots"][0] if project["roots"] else "", "name": project["name"], "count": 0}
                  for project_id, project in projects.items()}
        for session in result["sessions"]:
            key = session["projectId"]
            groups.setdefault(key, {"key": "unassigned", "path": "", "name": "未归属项目", "count": 0})["count"] += 1
        result["projects"] = list(groups.values())
        result["workspaces"] = result["projects"]
        result["source"] = {"kind": "local", "status": "connected", "message": "只读本地会话与已保存项目归属；目录不是项目，计数仅包含最近会话样本。没有新事件不等于执行卡死。"}
        return result

    def _projects(self, connection):
        """Use saved project identity, never infer membership from a shared cwd."""
        state_path = self.home / ".codex-global-state.json"
        try:
            stat = state_path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != self._project_state[0]:
                data = json.loads(state_path.read_text())
                if not isinstance(data, dict):
                    raise ValueError("Unsupported desktop state")
                keys = ["local-projects", "project-order", "thread-project-assignments", "projectless-thread-ids", "app-server-project-id-by-legacy-project-id-by-host"]
                self._project_state = (signature, {key: data.get(key) for key in keys})
        except (OSError, ValueError):
            self._project_state = (None, {})
        state = self._project_state[1]
        projects = {}
        if {"id", "name"} <= columns(connection, "projects"):
            roots = {}
            if {"project_id", "path", "position"} <= columns(connection, "project_roots"):
                for row in connection.execute("SELECT project_id,path FROM project_roots ORDER BY position"):
                    roots.setdefault(row[0], []).append(row[1])
            order = "position" if "position" in columns(connection, "projects") else "id"
            for row in connection.execute('SELECT id,name FROM projects ORDER BY "' + order + '"'):
                projects[row[0]] = {"name": row[1], "roots": roots.get(row[0], [])}
        else:
            local = state.get("local-projects") or {}
            if isinstance(local, dict):
                order = state.get("project-order") or []
                for key in list(order) + [key for key in local if key not in order]:
                    project = local.get(key)
                    if isinstance(project, dict) and project.get("name"):
                        projects[key] = {"name": project["name"], "roots": project.get("rootPaths") or []}
        alias_groups = state.get("app-server-project-id-by-legacy-project-id-by-host") or {}
        aliases = {}
        if isinstance(alias_groups, dict):
            for host, mapping in alias_groups.items():
                if host.startswith("local:") and Path(host[6:]).expanduser().resolve() == self.home and isinstance(mapping, dict):
                    aliases = mapping
                    break
        assignments = {}
        saved = state.get("thread-project-assignments") or {}
        if isinstance(saved, dict):
            for thread_id, value in saved.items():
                if isinstance(value, dict) and value.get("projectKind") == "local":
                    assignments[thread_id] = value.get("projectId")
        projectless = state.get("projectless-thread-ids") or []
        return projects, assignments, aliases, set(projectless) if isinstance(projectless, list) else set()

    def _logs(self, ids):
        """Read a bounded warning/error window; never expose serialized prompts."""
        if not ids or not (self.home / "logs_2.sqlite").is_file():
            return {}
        found = {}
        try:
            with readonly(self.home / "logs_2.sqlite") as connection:
                if not {"ts", "level", "thread_id", "feedback_log_body", "id"} <= columns(connection, "logs"):
                    return found
                sql = "SELECT ts,thread_id,feedback_log_body FROM logs WHERE level IN ('WARN','ERROR') AND thread_id IN (%s) ORDER BY id DESC LIMIT 200" % ','.join('?' for _ in ids)
                for row in connection.execute(sql, list(ids)):
                    body = row["feedback_log_body"] or ""
                    at = iso(row["ts"])
                    if not at:
                        continue
                    if re.search(r"at capacity|model.*capacity", body, re.I):
                        match = re.search(r"in\s+([\d.]+)s", body)
                        retry_at = iso(epoch(at) + float(match.group(1))) if match else None
                        entry = {"at": at, "kind": "capacity", "title": "模型服务繁忙", "detail": "日志记录模型容量不足" + ("，已安排延迟重试" if retry_at else ""), "retryAt": retry_at}
                    elif re.search(r"timed out|timeout", body, re.I):
                        entry = {"at": at, "kind": "timeout", "title": "操作曾等待超时", "detail": "该次操作等待超时；查看较新的活动，判断之后是否恢复"}
                    elif re.search(r"stream disconnected", body, re.I):
                        entry = {"at": at, "kind": "connection", "title": "连接曾中断", "detail": "当时连接中断；查看较新的活动，判断之后是否恢复"}
                    else:
                        continue
                    found.setdefault(row["thread_id"], []).append(entry)
        except (sqlite3.Error, OSError):
            pass
        return found

    def _read(self, raw_path):
        """Maintain byte offsets and only commit complete JSONL lines."""
        if not raw_path:
            return None
        path = Path(raw_path).expanduser().resolve()
        allowed = [self.home / "sessions", self.home / "archived_sessions"]
        if not any(path.is_relative_to(root) for root in allowed):
            return None
        try:
            info = path.stat()
            if not path.is_file():
                return None
            key = str(path)
            state = self._cache.get(key)
            identity = (info.st_dev, info.st_ino)
            changed_anchor = False
            if state and state["offset"] and info.st_size >= state["offset"]:
                with path.open("rb") as probe:
                    probe.seek(max(0, state["offset"] - 64))
                    changed_anchor = probe.read(min(64, state["offset"])) != state.get("anchor")
            if not state or state["identity"] != identity or info.st_size < state["offset"] or changed_anchor:
                state = {"identity": identity, "offset": 0, "last": None, "active": None, "terminal": None, "terminalKind": None, "context": {}, "compactions": [], "events": [], "pending": {}, "anchor": b""}
                self._cache[key] = state
            with path.open("rb") as stream:
                stream.seek(state["offset"])
                while True:
                    line = stream.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        break
                    state["offset"] = stream.tell()
                    try:
                        item = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    self._consume(state, item)
                stream.seek(max(0, state["offset"] - 64))
                state["anchor"] = stream.read(min(64, state["offset"]))
            return state
        except (OSError, ValueError):
            return None

    @staticmethod
    def _consume(state, item):
        at = iso(item.get("timestamp")) if item.get("timestamp") else None
        if not at:
            return
        payload = item.get("payload")
        if not isinstance(payload, dict):
            return
        outer = item.get("type")
        kind = payload.get("type")
        if outer == "compacted" or kind == "context_compacted":
            state["compactions"].append({"at": at})
            state["compactions"] = state["compactions"][-30:]
            state["events"].append({"at": at, "kind": "compaction", "title": "已整理较早的聊天内容", "detail": "记录到一次上下文压缩，用较少内容保留后续工作所需状态"})
        elif outer == "event_msg":
            if kind == "token_count":
                info = payload.get("info")
                if isinstance(info, dict):
                    usage = info.get("last_token_usage") or {}
                    count = usage.get("input_tokens") if isinstance(usage, dict) else None
                    window = info.get("model_context_window")
                    state["context"] = {"inputTokens": count if isinstance(count, int) else None, "windowTokens": window if isinstance(window, int) else None, "sampledAt": at}
            elif kind in ("task_started", "user_message"):
                state["active"] = at
                for event in state["pending"].values():
                    event["status"] = "unknown"
                state["pending"] = {}
                state["events"].append({"at": at, "kind": "started", "title": "开始处理新一轮请求", "detail": "收到用户消息或记录到本轮执行开始"})
            elif kind in ("task_complete", "turn_complete", "turn_aborted", "error"):
                state["terminal"] = at
                state["terminalKind"] = kind
                for event in state["pending"].values():
                    event["status"] = "unknown"
                state["pending"] = {}
                labels = {"task_complete": "本轮处理结束", "turn_complete": "本轮处理结束", "turn_aborted": "本轮处理已停止", "error": "本轮处理遇到错误"}
                detail = "AI 已结束这一轮处理；是否完成整个需求，还需结合回复内容判断" if kind in ("task_complete", "turn_complete") else "记录到本轮停止或错误，不能据此推断需求已完成"
                state["events"].append({"at": at, "kind": kind, "title": labels[kind], "detail": detail})
            elif kind in ("agent_message", "agent_reasoning"):
                # Deliberately do not read or return message/reasoning text.
                pass
            else:
                return
        elif outer == "response_item":
            if kind in ("function_call", "custom_tool_call"):
                call_id = payload.get("call_id") or payload.get("id") or "tool"
                event = {"at": at, "kind": "tool", "callId": str(call_id)[:100], "status": "running", **classify_activity(payload)}
                state["pending"][call_id] = event
                state["events"].append(event)
            elif kind in ("function_call_output", "custom_tool_call_output"):
                event = state["pending"].pop(payload.get("call_id"), None)
                if event:
                    event["status"] = "returned"
                    event["returnedAt"] = at
                    output = payload.get("output")
                    if isinstance(output, str) and len(output) <= 65536:
                        try:
                            output = json.loads(output)
                        except ValueError:
                            output = None
                    if isinstance(output, dict):
                        exit_code = output.get("exit_code")
                        if output.get("isError") is True or (isinstance(exit_code, int) and exit_code != 0):
                            event["status"] = "failed"
            elif kind == "message" and payload.get("role") == "assistant":
                pass
            else:
                return
        else:
            return
        state["last"] = at
        state["events"] = state["events"][-25:]

    def _session(self, row, sampled, logs):
        state = self._read(row.get("rollout_path"))
        updated = iso(row.get("updated_at_ms") or row.get("updated_at") or row.get("created_at"))
        last = (state or {}).get("last") or updated
        context = dict((state or {}).get("context") or {"inputTokens": None, "windowTokens": None, "sampledAt": None})
        count, window = context.get("inputTokens"), context.get("windowTokens")
        context["percent"] = round(count / window * 100, 1) if isinstance(count, int) and isinstance(window, int) and window > 0 else None
        terminal = (state or {}).get("terminal")
        active = (state or {}).get("active")
        code, label, detail, since, retry_at = "unknown", "状态未知", "暂无足够的执行状态记录；没有新事件不代表卡死", last, None
        if terminal and epoch(terminal) >= epoch(active):
            kind = state["terminalKind"]
            code, label = {"task_complete": ("completed", "本轮已结束"), "turn_complete": ("completed", "本轮已结束"), "turn_aborted": ("interrupted", "已停止"), "error": ("failed", "执行失败")}[kind]
            detail, since = "这一轮执行已结束；是否完成整个需求，请结合回复内容判断", terminal
        elif active:
            code, label, detail, since = "running", "执行中（记录）", "会话已开始执行，尚未发现明确终态；具体进展以新事件为准", active
            if state["pending"]:
                code, label = "tool_running", "工具执行中（记录）"
                detail = "最近在处理：" + "、".join(event["title"] for event in list(state["pending"].values())[-3:])
        latest_log = max(logs, key=lambda event: epoch(event["at"]), default=None)
        if latest_log and latest_log["kind"] == "capacity" and epoch(latest_log["at"]) > epoch(terminal) and epoch(latest_log["at"]) >= epoch(last):
            retry_at = latest_log.get("retryAt")
            since = latest_log["at"]
            if retry_at and epoch(retry_at) > epoch(sampled):
                code, label, detail = "retry_wait", "服务繁忙，等待重试", "日志明确记录容量不足及下次重试时间"
            else:
                code, label, detail = "unknown", "繁忙后暂无新事件", "历史记录显示服务繁忙；尚无证据确认当前仍在等待或已恢复"
        events = list((state or {}).get("events", [])) + [{key: value for key, value in event.items() if key != "retryAt"} for event in logs]
        events.sort(key=lambda event: epoch(event["at"]), reverse=True)
        events = group_activities(events)
        title = str(row.get("title") or "未命名会话").splitlines()[0][:90]
        workspace = str(row.get("cwd") or "")
        if state is None:
            detail = "会话记录暂不可见或格式不支持，不能推断执行状态"
        if row.get("history_mode") == "paginated" and epoch(updated) > epoch((state or {}).get("last")) + 2:
            if not latest_log or epoch(latest_log["at"]) < epoch(updated) - 2:
                code, label, detail, since, retry_at = "unknown", "分页历史状态未知", "会话有较新更新，但可见本地事件未覆盖该更新；不能推断当前执行状态", updated, None
        project_id = row.get("_project_id")
        return {"id": row["id"], "title": title, "workspace": workspace, "projectId": project_id, "projectKey": "project:" + project_id if project_id else "unassigned", "project": row.get("_project_name") or "未归属项目", "updatedAt": updated, "lastEventAt": last, "model": row.get("model"), "reasoningEffort": row.get("reasoning_effort"), "parentId": row.get("_parent"), "agentRole": row.get("agent_role") or row.get("agent_path") or "agent", "status": {"code": code, "label": label, "detail": detail, "since": since, "retryAt": retry_at, "evidenceAt": since}, "context": context, "compactions": list((state or {}).get("compactions", [])), "events": events[:20], "agents": []}
