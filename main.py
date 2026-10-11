"""
网页简历数字分身 Agent —— FastAPI + RAG 后端服务

技术栈：
- FastAPI：Web 服务
- BGE-small-zh-v1.5（本地模型）：文本向量化 + 余弦相似度检索
- 任意 OpenAI 兼容大模型接口：生成回答（当前小米 MiMo mimo-v2.6-flash，可在 .env 一键切换其他厂商）

流程：用户提问 → BGE 向量化 → 余弦相似度召回 Top-K 知识片段 → 拼接上下文 → 大模型生成回答
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

# 本地已缓存 BGE 模型时启用离线模式，避免联网校验拖慢启动
if (BASE_DIR / "bge_model").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio
import logging
import secrets
import shutil
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)  # 压掉 httpx 的请求明细噪音
logger = logging.getLogger("digital-twin")

# ==================== 配置区域 ====================
# —— 大模型（任意 OpenAI 兼容接口）——
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.xiaomimimo.com/v1").strip().rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "mimo-v2.6-flash").strip()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
# 深度思考开关：off=禁用 / on=启用 / 留空=不传该参数（智谱、MiMo 支持该参数；不支持的厂商留空）
LLM_THINKING = os.getenv("LLM_THINKING", "").strip().lower()
# 上游并发上限（按服务商限速调整，超限会排队等待而不是报错）
LLM_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_MAX_CONCURRENCY", "1")))

# —— 服务配置 ——
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# —— 管理后台 ——
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()  # 为空则管理后台不可用
ADMIN_SESSION_TTL = 24 * 3600  # 登录有效期（秒）

# —— 数据文件 ——
STATIC_DIR = BASE_DIR / "static"
SITE_CONFIG_FILE = BASE_DIR / "site_config.json"
RESUME_DATA_FILE = STATIC_DIR / "resume_data.js"
KNOWLEDGE_BACKUP_FILE = BASE_DIR / "person_knowledge.backup.json"
SITE_CONFIG_BACKUP_FILE = BASE_DIR / "site_config.backup.json"

# —— 向量库配置 ——
KNOWLEDGE_FILE = BASE_DIR / "person_knowledge.json"
BGE_CACHE_DIR = BASE_DIR / "bge_model"
BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# —— RAG 配置 ——
TOP_K = 4                 # 每次检索召回片段数
SCORE_THRESHOLD = 0.35    # 相似度低于该值视为「知识库无相关内容」
MAX_HISTORY = 8           # 最多携带的历史消息条数（4 轮对话）
MAX_QUERY_CHARS = 1000    # 单条提问最大长度
MAX_TARGET_JOB_CHARS = 50 # 目标岗位字段最大长度

# —— 未找到信息时的固定回复 ——
NO_INFO_REPLY = "我的个人知识库暂无相关内容，无法作答。如需了解更多，请直接联系我。"

# —— 页面默认设置（site_config.json 缺失字段用这里的值兜底；{name} 会自动替换为知识库姓名）——
DEFAULT_SITE_CONFIG: Dict[str, Any] = {
    "page_title": "{name} · 个人主页 & AI 数字分身",
    "header_badge": "HDW",
    "header_name": "胡大为",
    "header_subtitle": "个人履历与经历问答",
    "owner_avatar": "胡",
    "visitor_avatar": "访",
    "agent_title_row": "经历与背景问答",
    "agent_subtitle": "AI 数字分身 · 基于真实履历作答",
    "resume_title_row": "简历预览",
    "status_text": "A4 · Page 1 of 1 · 点击「导出 PDF」可保存为文件",
    "welcome_title": "你好！",
    "welcome_text": "我是{name}的 AI 数字分身，掌握他的项目经历、技术栈、实习经历与求职意向。有任何想了解的，直接问我；也可以点击上方的常见问题快速开始。",
    "chat_placeholder": "输入问题（如：电商系统遇到过什么难点？云邮箱用了哪些技术？）...",
    "quick_questions": ["介绍一下你的项目经历", "你掌握哪些技术栈？", "实习都做了什么？", "期望薪资与城市？"],
    "footer_text": "{name}的个人主页 · 经历问答由 AI 数字分身实时生成，仅供交流参考",
}

# ==================== 全局状态 ====================
embedding_model: Optional[Any] = None   # SentenceTransformer（延迟加载）
knowledge_docs: List[Dict[str, str]] = []
knowledge_vectors: Optional[np.ndarray] = None
person_name: str = "简历主人"
llm_semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENCY)


# ==================== 知识库处理 ====================
def _is_placeholder(value: Any) -> bool:
    """判断是否为未填写的模板占位符（以【开头）"""
    return isinstance(value, str) and value.strip().startswith("【")


def _clean_list(items: Any) -> List[str]:
    """过滤掉空值和占位符，返回干净的字符串列表"""
    if not isinstance(items, list):
        return []
    return [x.strip() for x in items if isinstance(x, str) and x.strip() and not _is_placeholder(x)]


def _clean_text(value: Any) -> str:
    """返回过滤占位符后的文本，空则返回空串"""
    if isinstance(value, str) and value.strip() and not _is_placeholder(value):
        return value.strip()
    return ""


def load_knowledge_base() -> List[Dict[str, str]]:
    """解析 person_knowledge.json，生成用于向量检索的文本片段（自动过滤未填写的占位符）"""
    global person_name

    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(f"知识库文件不存在: {KNOWLEDGE_FILE}")

    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents: List[Dict[str, str]] = []
    basic = data.get("basic_info") or {}
    if _clean_text(basic.get("name")):
        person_name = basic["name"].strip()

    # —— 基础信息 ——
    edu_texts = []
    for edu in basic.get("education") or []:
        if not isinstance(edu, dict):
            continue
        parts = [_clean_text(edu.get(k)) for k in ("school", "degree", "major", "start_time", "end_time")]
        parts = [p for p in parts if p]
        if not parts:
            continue
        text = " ".join(parts)
        courses = _clean_list(edu.get("courses"))
        if courses:
            text += "；主修课程：" + "、".join(courses)
        extras = _clean_text(edu.get("extras"))
        if extras:
            text += "；" + extras
        gpa = _clean_text(edu.get("gpa"))
        if gpa:
            text += "；成绩：" + gpa
        edu_texts.append(text)

    basic_parts = []
    if _clean_text(basic.get("name")):
        basic_parts.append(f"姓名：{basic['name'].strip()}")
    if _clean_text(str(basic.get("age", ""))):
        basic_parts.append(f"年龄：{basic['age']}")
    if _clean_text(basic.get("location")):
        basic_parts.append(f"所在城市：{basic['location'].strip()}")
    if edu_texts:
        basic_parts.append(f"教育经历：{'；'.join(edu_texts)}")
    if _clean_text(basic.get("email")):
        basic_parts.append(f"联系邮箱：{basic['email'].strip()}")
    github_repos = _clean_list(basic.get("github_repos"))
    if github_repos:
        basic_parts.append("GitHub 仓库：" + "、".join(github_repos))
    if basic_parts:
        documents.append({"content": "【基础信息】" + "，".join(basic_parts), "source": "basic_info"})

    # —— 联系方式与开源主页（独立片段：便于「GitHub / 邮箱 / 联系方式」类问题精准召回）——
    contact_parts = []
    if _clean_text(basic.get("email")):
        contact_parts.append(f"联系邮箱：{basic['email'].strip()}")
    if github_repos:
        contact_parts.append("GitHub 仓库：" + "、".join(github_repos))
    if contact_parts:
        documents.append({"content": "【联系方式与开源主页】" + "；".join(contact_parts), "source": "contact"})

    # —— 专业技能 / 证书 ——
    skills = _clean_list(basic.get("skills"))
    if skills:
        documents.append({"content": "【专业技能】" + "、".join(skills), "source": "skills"})

    certs = _clean_list(basic.get("certificates"))
    if certs:
        documents.append({"content": "【证书资质】" + "、".join(certs), "source": "certificates"})

    # —— 求职意向 ——
    intention = basic.get("job_intention") or {}
    positions = _clean_list(intention.get("target_positions"))
    int_parts = []
    if positions:
        int_parts.append("目标岗位：" + "、".join(positions))
    for label, key in (("期望薪资", "expected_salary"), ("期望城市", "expected_city"), ("到岗时间", "available_time")):
        val = _clean_text(intention.get(key))
        if val:
            int_parts.append(f"{label}：{val}")
    if int_parts:
        documents.append({"content": "【求职意向】" + "，".join(int_parts), "source": "job_intention"})

    # —— 项目经验 ——
    # 注意：BGE 模型单段上限约 512 token，长文档会被截断导致检索变差，
    # 因此每个项目拆成「概览 / 职责 / 难点成果」三个片段分别入索引
    for proj in data.get("projects") or []:
        if not isinstance(proj, dict) or _is_placeholder(proj.get("name", "")):
            continue
        name = proj.get("name", "").strip()
        head = f"【项目名称】{name}"

        overview_lines = [head]
        role = _clean_text(proj.get("role"))
        if role:
            overview_lines.append(f"【项目角色】{role}")
        bg = _clean_text(proj.get("background"))
        if bg:
            overview_lines.append(f"【项目背景】{bg}")
        tech = _clean_list(proj.get("tech_stack"))
        if tech:
            overview_lines.append("【技术栈】" + "、".join(tech))
        documents.append({"content": "\n".join(overview_lines), "source": f"project_{name}_overview"})

        duties = _clean_list(proj.get("responsibilities"))
        if duties:
            documents.append({"content": head + "\n【核心职责】" + "、".join(duties), "source": f"project_{name}_duties"})

        detail_lines = [head]
        for label, key in (("难点", "challenges"), ("优化成果", "optimizations")):
            vals = _clean_list(proj.get(key))
            if vals:
                detail_lines.append(f"【{label}】" + "、".join(vals))
        hl = _clean_text(proj.get("code_highlights"))
        if hl:
            detail_lines.append(f"【代码亮点】{hl}")
        if len(detail_lines) > 1:
            documents.append({"content": "\n".join(detail_lines), "source": f"project_{name}_details"})

    # —— 项目经历总览 ——
    # 自动生成摘要片段，提升「介绍一下你的项目经历」这类泛化问法的召回排名
    proj_overview_items = []
    for proj in data.get("projects") or []:
        if not isinstance(proj, dict) or _is_placeholder(proj.get("name", "")):
            continue
        name = proj.get("name", "").strip()
        bg = _clean_text(proj.get("background"))
        brief = bg.split("，")[0].split("。")[0] if bg else ""
        proj_overview_items.append(f"{len(proj_overview_items) + 1}. {name}" + (f"：{brief}" if brief else ""))
    if proj_overview_items:
        documents.append({
            "content": f"【项目经历总览】共 {len(proj_overview_items)} 个项目：" + "；".join(proj_overview_items),
            "source": "projects_overview",
        })

    # —— 实习经历 ——
    for intern in data.get("internships") or []:
        if not isinstance(intern, dict) or _is_placeholder(intern.get("company", "")):
            continue
        lines = [f"【实习公司】{intern.get('company', '').strip()}"]
        pos = _clean_text(intern.get("position"))
        if pos:
            lines.append(f"【实习岗位】{pos}")
        st, et = _clean_text(intern.get("start_time")), _clean_text(intern.get("end_time"))
        if st or et:
            lines.append(f"【实习时间】{st}至{et}")
        for label, key in (("实习职责", "responsibilities"), ("实习成果", "achievements")):
            vals = _clean_list(intern.get(key))
            if vals:
                lines.append(f"【{label}】" + "、".join(vals))
        documents.append({"content": "\n".join(lines), "source": f"internship_{intern.get('company', '')}"})

    # —— 实训经历 ——
    for train in data.get("training") or []:
        if not isinstance(train, dict) or _is_placeholder(train.get("name", "")):
            continue
        lines = [f"【实训名称】{train.get('name', '').strip()}"]
        for label, key in (("实训机构", "institution"), ("实训时间", "time"), ("实训内容", "content"), ("实训成果", "achievement")):
            val = _clean_text(train.get(key))
            if val:
                lines.append(f"【{label}】{val}")
        documents.append({"content": "\n".join(lines), "source": f"training_{train.get('name', '')}"})

    # —— 竞赛经历 ——
    for comp in data.get("competitions") or []:
        if not isinstance(comp, dict) or _is_placeholder(comp.get("name", "")):
            continue
        lines = [f"【竞赛名称】{comp.get('name', '').strip()}"]
        for label, key in (("竞赛时间", "time"), ("竞赛级别", "level"), ("获奖情况", "award"), ("项目简介", "description")):
            val = _clean_text(comp.get(key))
            if val:
                lines.append(f"【{label}】{val}")
        documents.append({"content": "\n".join(lines), "source": f"competition_{comp.get('name', '')}"})

    # —— 自我介绍 / 自我评价 ——
    intro = _clean_text(data.get("self_introduction"))
    if intro:
        documents.append({"content": f"【自我介绍】{intro}", "source": "self_introduction"})

    evaluation = _clean_text(data.get("self_evaluation"))
    if evaluation:
        documents.append({"content": f"【自我评价】{evaluation}", "source": "self_evaluation"})

    # —— 擅长技术问题 ——
    strengths = _clean_list(data.get("technical_strengths"))
    if strengths:
        documents.append({"content": "【擅长技术问题】" + "、".join(strengths), "source": "technical_strengths"})

    return documents


# ==================== 向量检索 ====================
def load_embedding_model() -> Any:
    """加载 BGE 向量模型（优先从本地缓存加载）"""
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("正在加载 BGE 向量模型（首次启动约需 10~30 秒）...")
        embedding_model = SentenceTransformer(
            BGE_MODEL_NAME,
            cache_folder=str(BGE_CACHE_DIR),
            local_files_only=(BASE_DIR / "bge_model").exists(),
        )
        # 兼容不同版本 sentence-transformers 的方法命名
        dim_getter = getattr(embedding_model, "get_embedding_dimension", None) or getattr(
            embedding_model, "get_sentence_embedding_dimension"
        )
        logger.info("BGE 模型加载完成，向量维度: %s", dim_getter())
    return embedding_model


def build_vectorstore() -> None:
    """构建（或重建）BGE 向量索引"""
    global knowledge_docs, knowledge_vectors

    documents = load_knowledge_base()
    knowledge_docs = documents

    if not documents:
        knowledge_vectors = np.zeros((0, 512), dtype=np.float32)
        logger.warning("知识库为空（所有条目均为未填写的占位符），请检查 person_knowledge.json")
        return

    model = load_embedding_model()
    texts = [doc["content"] for doc in documents]
    knowledge_vectors = np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=32), dtype=np.float32)
    logger.info("向量索引构建完成：%d 条知识片段", len(texts))


def ensure_vectorstore() -> None:
    """确保向量库已构建（同步函数，在线程池中调用）"""
    if knowledge_vectors is None:
        build_vectorstore()


def retrieve_context(query: str) -> Tuple[str, List[str], float]:
    """
    检索与问题最相关的知识片段。

    返回 (context, sources, top_score)：
    - top_score 低于阈值时视为知识库无相关内容，context 返回空串
    """
    ensure_vectorstore()

    if not knowledge_docs or knowledge_vectors is None or len(knowledge_docs) == 0:
        return "", [], 0.0

    model = load_embedding_model()
    query_vector = np.asarray(model.encode([query], normalize_embeddings=True), dtype=np.float32)

    # 向量已归一化，点积即余弦相似度
    similarities = (knowledge_vectors @ query_vector.T).flatten()

    k = min(TOP_K, len(knowledge_docs))
    top_indices = np.argsort(similarities)[::-1][:k]
    top_score = float(similarities[top_indices[0]])

    if top_score < SCORE_THRESHOLD:
        return "", [], top_score

    picked = [int(i) for i in top_indices if float(similarities[i]) >= SCORE_THRESHOLD]
    context = "\n\n".join(knowledge_docs[i]["content"] for i in picked)
    sources = [knowledge_docs[i]["source"] for i in picked]
    return context, sources, top_score


# ==================== 页面数据生成 ====================
def load_site_config() -> Dict[str, Any]:
    """读取站点配置（缺失字段用默认值兜底）"""
    config = dict(DEFAULT_SITE_CONFIG)
    if SITE_CONFIG_FILE.exists():
        try:
            with open(SITE_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                config.update({k: v for k, v in saved.items() if v is not None})
        except Exception as exc:  # noqa: BLE001
            logger.warning("site_config.json 读取失败（使用默认配置）：%s", exc)
    return config


def build_resume_data() -> None:
    """生成前端数据文件 static/resume_data.js（公开安全：剔除手机号等隐私字段）"""
    knowledge: Dict[str, Any] = {}
    if KNOWLEDGE_FILE.exists():
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            knowledge = json.load(f)
    # 剔除不对外展示的隐私字段
    basic = knowledge.get("basic_info")
    if isinstance(basic, dict):
        basic.pop("phone", None)

    payload = {"site": load_site_config(), "knowledge": knowledge}
    content = (
        "/* 本文件由后端自动生成，请勿手动修改；在管理后台保存时自动刷新。 */\n"
        "window.RESUME_DATA = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n"
    )
    RESUME_DATA_FILE.write_text(content, encoding="utf-8")
    logger.info("前端数据文件已生成：%s", RESUME_DATA_FILE.name)


# ==================== 大模型调用 ====================
class LLMError(Exception):
    """大模型调用异常（public_message 可直接展示给访客）"""

    def __init__(self, public_message: str):
        super().__init__(public_message)
        self.public_message = public_message


def _llm_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}


def _llm_payload(messages: List[Dict[str, str]], stream: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": stream,
    }
    if LLM_THINKING in ("on", "enabled", "true"):
        payload["thinking"] = {"type": "enabled"}
    elif LLM_THINKING in ("off", "disabled", "false"):
        payload["thinking"] = {"type": "disabled"}
    return payload


def _friendly_http_error(status: int, body: str) -> str:
    """把上游错误转换成对访客友好的提示"""
    if status in (401, 403):
        return "AI 服务鉴权失败（API Key 无效或已过期），请联系网站主人更新配置"
    if status == 404:
        return f"AI 服务配置有误：模型 {LLM_MODEL} 不存在，请检查 .env 中的 LLM_BASE_URL / LLM_MODEL"
    if status == 429:
        return "AI 服务当前请求较多（免费额度限流），请稍等几秒再试"
    if status >= 500:
        return "AI 服务暂时不可用，请稍后重试"
    snippet = body[:200] if body else ""
    return f"AI 服务返回错误（HTTP {status}）：{snippet}"


def ensure_llm_configured() -> None:
    if not LLM_API_KEY:
        raise HTTPException(status_code=503, detail="网站尚未配置大模型 API Key（.env 中的 LLM_API_KEY），请联系网站主人")


async def call_llm(messages: List[Dict[str, str]]) -> str:
    """非流式调用大模型（网络抖动时自动重试一次）"""
    payload = _llm_payload(messages, stream=False)
    last_exc: Optional[Exception] = None

    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                resp = await client.post(f"{LLM_BASE_URL}/chat/completions", headers=_llm_headers(), json=payload)
            if resp.status_code != 200:
                raise LLMError(_friendly_http_error(resp.status_code, resp.text))
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except LLMError:
            raise
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError) as exc:
            last_exc = exc
            if attempt == 1:
                logger.warning("大模型请求失败（%s），1 秒后重试...", type(exc).__name__)
                await asyncio.sleep(1)

    raise LLMError("AI 服务连接失败，请稍后重试") from last_exc


async def stream_llm(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """流式调用大模型，逐段产出文本增量"""
    payload = _llm_payload(messages, stream=True)
    timeout = httpx.Timeout(LLM_TIMEOUT, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{LLM_BASE_URL}/chat/completions", headers=_llm_headers(), json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "ignore")
                raise LLMError(_friendly_http_error(resp.status_code, body))

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk_str = line[5:].strip()
                if chunk_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(chunk_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece


# ==================== Prompt 构建 ====================
def build_system_prompt(context: str, target_job: str = "") -> str:
    """构建系统提示词（知识库上下文为空时按兜底规则回答）"""
    target_job = (target_job or "").strip()[:MAX_TARGET_JOB_CHARS]
    job_context = ""
    if target_job:
        job_context = (
            f"\n【重要】访问者正在以「{target_job}」岗位的视角提问，"
            f"请优先呈现与该岗位相关的项目、技能与经历，突出匹配度。\n"
        )

    knowledge_block = context if context else "（无）"

    return f"""你是「{person_name}」的个人数字分身（AI 简历助手），部署在其个人网站上，负责向访客（HR、面试官、同行）介绍 TA 的简历信息。

回答规则（按优先级严格执行）：
1. 涉及简历主人的个人信息、经历、技能、求职意向的问题，只能依据下方【知识库内容】回答，禁止编造、夸大或臆测任何细节。
2. 【知识库内容】为空或与问题无关时：
   - 若问题在打听简历主人的具体信息 → 直接回答"{NO_INFO_REPLY}"
   - 若是打招呼、闲聊或通用问题 → 友好自然地回应，不要提及知识库
3. 与简历主人无关的行业/技术问题：可以用通用知识回答，但要注明"根据行业经验"，不得冒充是其个人经历。
4. 隐私保护：不主动透露手机号等隐私信息；对方询问联系方式或 GitHub 时，可以提供知识库中的邮箱与 GitHub 链接。
5. 被问到"你会什么 / 有什么技能 / 擅长什么"时，直接列出知识库中的具体技能、技术栈与擅长领域。

回答风格：像一位自信、专业、友好的求职者在交流，中文作答，可适当展开，避免啰嗦。
回答格式：纯文本，不要使用 Markdown 符号（如 **、#、`）；用数字序号或破折号组织要点。
{job_context}
【知识库内容】
{knowledge_block}
"""


# ==================== API 模型 ====================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    target_job: str = ""
    history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    sources: List[str] = []


def _sanitize_history(history: List[ChatMessage]) -> List[Dict[str, str]]:
    """只保留合法的 user/assistant 消息，并对长度做限制"""
    cleaned: List[Dict[str, str]] = []
    for msg in history[-MAX_HISTORY:]:
        role = msg.role if msg.role in ("user", "assistant") else None
        content = (msg.content or "").strip()
        if role and content:
            cleaned.append({"role": role, "content": content[:4000]})
    return cleaned


def build_retrieval_text(query: str, history: List[ChatMessage]) -> str:
    """构造检索文本：追问场景下拼接最近一轮对话，解决「它 / 第二个」等指代问题"""
    if not history:
        return query
    parts: List[str] = []
    for msg in history[-2:]:  # 最近一轮：用户问题 + 助手回答
        text = (msg.content or "").strip()
        if text:
            parts.append(text[:300])
    if not parts:
        return query
    return " ".join(parts) + " " + query


def _build_messages(request: ChatRequest, query: str, context: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": build_system_prompt(context, request.target_job)}]
    messages.extend(_sanitize_history(request.history))
    messages.append({"role": "user", "content": query})
    return messages


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ==================== FastAPI 应用 ====================
@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("正在初始化数字分身服务...")
    try:
        build_vectorstore()
    except Exception as exc:  # 知识库问题不阻塞服务启动，请求时会给出明确报错
        logger.error("向量库初始化失败：%s", exc)
    try:
        build_resume_data()
    except Exception as exc:  # noqa: BLE001
        logger.error("前端数据文件生成失败：%s", exc)
    if not LLM_API_KEY:
        logger.warning("未配置 LLM_API_KEY，对话接口将不可用，请在 .env 中填写")
    else:
        logger.info("大模型：%s @ %s", LLM_MODEL, LLM_BASE_URL)
    yield


app = FastAPI(title="数字分身 Agent API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # 如需收紧，改成你部署简历网站的域名
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/agent/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """数字分身对话接口（一次性返回完整回答）"""
    ensure_llm_configured()

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if len(query) > MAX_QUERY_CHARS:
        raise HTTPException(status_code=400, detail=f"问题过长，请控制在 {MAX_QUERY_CHARS} 字以内")

    loop = asyncio.get_running_loop()
    search_text = build_retrieval_text(query, request.history)
    context, sources, top_score = await loop.run_in_executor(None, retrieve_context, search_text)
    logger.info("问答请求 | 问题=%r | 命中片段=%s | 相似度=%.3f", query[:50], sources, top_score)

    messages = _build_messages(request, query, context)
    try:
        async with llm_semaphore:
            reply = await call_llm(messages)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=exc.public_message)

    return ChatResponse(reply=reply, sources=sources)


@app.post("/agent/chat/stream")
async def chat_stream(request: ChatRequest):
    """数字分身对话接口（SSE 流式返回，打字机效果）"""
    ensure_llm_configured()

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if len(query) > MAX_QUERY_CHARS:
        raise HTTPException(status_code=400, detail=f"问题过长，请控制在 {MAX_QUERY_CHARS} 字以内")

    loop = asyncio.get_running_loop()
    search_text = build_retrieval_text(query, request.history)
    context, sources, top_score = await loop.run_in_executor(None, retrieve_context, search_text)
    logger.info("流式请求 | 问题=%r | 命中片段=%s | 相似度=%.3f", query[:50], sources, top_score)

    messages = _build_messages(request, query, context)

    async def event_generator() -> AsyncGenerator[str, None]:
        yield _sse("meta", {"sources": sources})
        try:
            async with llm_semaphore:
                async for piece in stream_llm(messages):
                    yield _sse("delta", {"content": piece})
        except LLMError as exc:
            yield _sse("error", {"message": exc.public_message})
        except Exception as exc:  # noqa: BLE001
            logger.error("流式生成异常：%r", exc)
            yield _sse("error", {"message": "服务内部错误，请稍后重试"})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/agent/reload-knowledge")
async def reload_knowledge():
    """重新加载 person_knowledge.json，重建向量索引并刷新前台数据文件"""
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, build_vectorstore)
        await loop.run_in_executor(None, build_resume_data)
        return {"status": "success", "message": "知识库重建完成", "documents": len(knowledge_docs)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"知识库重建失败: {exc}")


@app.get("/health")
async def health_check():
    """健康检查：反映 LLM 配置、知识库状态"""
    return {
        "status": "ok",
        "llm": {"model": LLM_MODEL, "base_url": LLM_BASE_URL, "configured": bool(LLM_API_KEY)},
        "embedding": "bge-small-zh-v1.5",
        "knowledge_docs": len(knowledge_docs),
        "top_k": TOP_K,
        "score_threshold": SCORE_THRESHOLD,
    }


# ==================== 管理后台 ====================
_admin_sessions: Dict[str, float] = {}


def _new_admin_session() -> str:
    token = secrets.token_urlsafe(32)
    _admin_sessions[token] = time.time() + ADMIN_SESSION_TTL
    return token


def require_admin(request: Request) -> None:
    """校验管理后台登录态（Cookie 会话）"""
    token = request.cookies.get("dt_admin", "")
    expire = _admin_sessions.get(token)
    if not expire or expire < time.time():
        _admin_sessions.pop(token, None)
        raise HTTPException(status_code=401, detail="未登录或登录已过期，请重新登录")


class AdminLoginBody(BaseModel):
    password: str = ""


class AdminSaveBody(BaseModel):
    knowledge: Dict[str, Any]
    site: Dict[str, Any]


@app.post("/admin/login")
async def admin_login(body: AdminLoginBody, response: Response):
    """管理后台登录（密码在 .env 的 ADMIN_PASSWORD 中配置）"""
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="未配置管理密码：请在 .env 中设置 ADMIN_PASSWORD 后重启服务")
    if body.password != ADMIN_PASSWORD:
        await asyncio.sleep(0.6)  # 减缓暴力尝试
        raise HTTPException(status_code=401, detail="密码错误")
    token = _new_admin_session()
    response.set_cookie("dt_admin", token, max_age=ADMIN_SESSION_TTL, httponly=True, samesite="lax", path="/")
    return {"status": "ok"}


@app.post("/admin/logout")
async def admin_logout(request: Request, response: Response):
    _admin_sessions.pop(request.cookies.get("dt_admin", ""), None)
    response.delete_cookie("dt_admin", path="/")
    return {"status": "ok"}


@app.get("/admin/check")
async def admin_check(request: Request):
    """校验登录态"""
    require_admin(request)
    return {"status": "ok"}


@app.get("/admin/api/data")
async def admin_get_data(request: Request):
    """获取当前知识库与页面配置（完整数据，含手机号，仅登录后可见）"""
    require_admin(request)
    knowledge: Dict[str, Any] = {}
    if KNOWLEDGE_FILE.exists():
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            knowledge = json.load(f)
    return {"knowledge": knowledge, "site": load_site_config()}


@app.post("/admin/api/save")
async def admin_save(body: AdminSaveBody, request: Request):
    """保存知识库与页面配置：备份 → 写盘 → 重建向量库 → 刷新前端数据"""
    require_admin(request)

    if not isinstance(body.knowledge.get("basic_info"), dict):
        raise HTTPException(status_code=400, detail="知识库格式不正确：缺少 basic_info")
    if not isinstance(body.site, dict):
        raise HTTPException(status_code=400, detail="页面配置格式不正确")

    loop = asyncio.get_running_loop()

    def _persist_files() -> None:
        # 先备份上一版（防止误覆盖，可手动回滚）
        if KNOWLEDGE_FILE.exists():
            shutil.copy2(KNOWLEDGE_FILE, KNOWLEDGE_BACKUP_FILE)
        if SITE_CONFIG_FILE.exists():
            shutil.copy2(SITE_CONFIG_FILE, SITE_CONFIG_BACKUP_FILE)
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            json.dump(body.knowledge, f, ensure_ascii=False, indent=2)
            f.write("\n")
        merged_site = dict(DEFAULT_SITE_CONFIG)
        merged_site.update({k: v for k, v in body.site.items() if v is not None})
        with open(SITE_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(merged_site, f, ensure_ascii=False, indent=2)
            f.write("\n")

    try:
        await loop.run_in_executor(None, _persist_files)
    except Exception as exc:  # noqa: BLE001
        logger.error("保存数据失败：%s", exc)
        raise HTTPException(status_code=500, detail=f"保存失败：{exc}")

    try:
        await loop.run_in_executor(None, build_vectorstore)
        await loop.run_in_executor(None, build_resume_data)
    except Exception as exc:  # noqa: BLE001
        logger.error("保存后重建失败：%s", exc)
        return {"status": "warning", "message": f"数据已保存，但向量库/前端数据重建失败：{exc}", "documents": len(knowledge_docs)}

    logger.info("管理后台保存成功：知识片段 %d 条", len(knowledge_docs))
    return {"status": "ok", "message": "保存成功：已重建向量库并刷新主页数据", "documents": len(knowledge_docs)}


# ==================== 静态页面 ====================
INDEX_FILE = STATIC_DIR / "index.html"
ADMIN_FILE = STATIC_DIR / "admin.html"


@app.get("/", include_in_schema=False)
async def root():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE, headers={"Cache-Control": "no-cache"})
    return {"message": "前端页面缺失：请确认 static/index.html 存在"}


@app.get("/index.html", include_in_schema=False)
async def index_page():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE, headers={"Cache-Control": "no-cache"})
    return {"message": "前端页面缺失：请确认 static/index.html 存在"}


@app.get("/admin", include_in_schema=False)
async def admin_page():
    if ADMIN_FILE.exists():
        return FileResponse(ADMIN_FILE, headers={"Cache-Control": "no-cache"})
    return {"message": "管理页面缺失：请确认 static/admin.html 存在"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


class NoCacheStaticFiles(StaticFiles):
    """静态文件加 no-cache 协商缓存：内容改动后刷新即可看到最新版，避免浏览器用到旧缓存"""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


if STATIC_DIR.exists():
    app.mount("/static", NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")


# ==================== 启动 ====================
if __name__ == "__main__":
    import uvicorn

    logger.info("启动数字分身 Agent 服务...")
    logger.info("聊天页面: http://localhost:%d/  |  API文档: http://localhost:%d/docs", PORT, PORT)
    uvicorn.run(app, host=HOST, port=PORT)
