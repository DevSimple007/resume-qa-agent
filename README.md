# 网页简历数字分身 Agent

基于 RAG 架构的 AI 数字分身系统，可嵌入个人简历网站：访客（HR / 面试官）直接向「你」提问，AI 依据简历知识库作答。

- **检索**：BGE-small-zh-v1.5 本地向量模型 + 余弦相似度语义检索（无需额外服务）
- **生成**：任意 OpenAI 兼容大模型接口（当前小米 MiMo `mimo-v2.6-flash`）
- **防幻觉**：低温度 + 相似度阈值过滤 + 严格提示词约束，只答知识库里有的事

## 目录结构

```
├── main.py                 # FastAPI 后端服务（RAG + 大模型调用 + 管理后台接口）
├── person_knowledge.json   # 结构化简历知识库（本地保留，不入库；管理后台编辑）
├── site_config.json        # 页面设置（本地保留，不入库；管理后台编辑）
├── static/
│   ├── index.html          # 前台页面（左：简历纸张；右：AI 经历问答）
│   ├── admin.html          # 管理后台页面（/admin）
│   ├── resume_data.js      # 前台数据文件（后端自动生成，不入库，勿手改）
│   └── vendor/
│       └── tailwind.css    # 编译好的样式文件（由 _tailwind.* 生成，勿手改）
├── _tailwind.config.js     # 样式编译配置（改页面样式类后需重新编译）
├── _tailwind.input.css     # 样式编译入口
├── bge_model/              # BGE 向量模型本地缓存（自动下载，勿删）
├── .env                    # 环境变量（API Key、管理密码等，本地保留，不入库）
├── .env.example            # 配置模板（可安全入库）
├── .gitignore              # 忽略规则（.env / 知识库 / 数据文件等本地隐私）
└── requirements.txt        # Python 依赖
```

## 快速开始

> **隐私说明**：本仓库**不含任何个人数据文件**——`person_knowledge.json`（知识库）、`site_config.json`（页面设置）、`static/resume_data.js`（前端数据）、`.env`（API Key / 管理密码）均通过 `.gitignore` 本地保留、不入库。首次运行时服务会以默认 / 空配置启动，通过管理后台（/admin）保存内容后自动生成这些文件。

### 1. 安装依赖

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> 首次安装会拉取 torch 等依赖，约 2~3 GB，建议使用国内镜像：
> `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 2. 配置大模型 API Key

复制 `.env.example` 为 `.env`，填入 API Key：

```bash
LLM_API_KEY=你的Key
LLM_BASE_URL=https://api.xiaomimimo.com/v1
LLM_MODEL=mimo-v2.6-flash
```

MiMo Key 申请：https://mimo.mi.com → 控制台 → API Keys。

> ⚠️ MiMo 有两套密钥通道，必须搭配使用：`sk-` 开头走 `https://api.xiaomimimo.com/v1`（按量付费）；`tp-`/`ttp-` 开头走 `https://token-plan-cn.xiaomimimo.com/v1`（Token Plan）。用错地址会鉴权失败。

### 3. 启动服务

```bash
python main.py
```

- 聊天页面：http://localhost:8000/
- 管理后台：http://localhost:8000/admin（改简历和页面内容，见下文）
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health （可查看 Key 是否已配置、知识库条数）

首次启动会自动下载 BGE 模型（约 90 MB）。国内网络可在 `.env` 中加 `HF_ENDPOINT=https://hf-mirror.com` 加速。

## 切换大模型（随时可换）

代码不绑定任何厂商，只要是 OpenAI 兼容接口，改 `.env` 三行即可：

| 厂商 | LLM_BASE_URL | LLM_MODEL |
|------|--------------|-----------|
| 小米 MiMo（当前） | `https://api.xiaomimimo.com/v1`（sk- 密钥） | `mimo-v2.6-flash` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4.7-flash`（免费档） |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| 小米 MiMo Token Plan | `https://token-plan-cn.xiaomimimo.com/v1`（tp- 密钥） | `mimo-v2.6-flash` |

> `LLM_THINKING` 为智谱 / MiMo 支持的深度思考开关；不支持的厂商请留空该参数。

## 接口说明

| 接口 | 方法 | 说明 |
|------|------|------|
| `/agent/chat` | POST | 对话（一次性返回完整回答） |
| `/agent/chat/stream` | POST | 对话（SSE 流式返回，前端默认使用） |
| `/agent/reload-knowledge` | POST | 重新加载知识库并重建向量索引 |
| `/health` | GET | 健康检查（LLM 配置 / 知识库状态） |

请求体示例：

```json
{
  "query": "介绍一下你的项目经历",
  "target_job": "全栈开发师",
  "history": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！..."}]
}
```

`history` 为可选的多轮对话上下文（前端自动携带最近 8 条）。

## 管理后台（/admin）—— 改内容看这里

访客看展示页，你自己用管理后台改内容：

1. 浏览器打开 http://localhost:8000/admin
2. 输入管理密码（在 `.env` 的 `ADMIN_PASSWORD` 中配置，**部署到公网前务必修改默认值**）
3. 可修改的内容：
   - **知识库**：基础信息、求职意向、教育背景、技能证书、项目经历、实习经历、实训竞赛、自我描述 —— 这些同时是 AI 问答的依据
   - **页面设置**：问答区头像字符、欢迎语、常见问题建议、顶栏文字、页脚文字等
4. 点「保存并生效」：自动备份上一版 → 写入文件 → 重建 AI 向量库 → 刷新主页数据（无需重启服务，刷新主页即可看到）

说明：
- 每次保存会自动生成 `person_knowledge.backup.json` / `site_config.backup.json`（上一版），误改可手动回滚
- 手机号等隐私字段只会存在于服务端知识库，**不会**进入前台数据文件（`static/resume_data.js`）
- 管理登录有效期 24 小时；密码错误会延迟响应，降低暴力尝试风险

## 知识库维护（命令行方式，可选）

除管理后台外，也可以直接编辑 `person_knowledge.json` 后调用接口热更新：

```bash
curl -X POST http://localhost:8000/agent/reload-knowledge
```

## 修改前端页面样式（可选）

前端样式由 Tailwind 预编译为 `static/vendor/tailwind.css`，**不依赖任何外网 CDN**（国内访问稳定）。如果新增了 `static/index.html` 里没出现过的 Tailwind 类名，需要重新编译，否则新类不生效：

```bash
# 安装一次编译工具（需 Node.js 18+）
npm install tailwindcss@3.4.16

# 在项目根目录执行编译
npx tailwindcss -c _tailwind.config.js -i _tailwind.input.css -o static/vendor/tailwind.css --minify
```

仅修改文字内容（不引入新样式类）时无需重新编译。

## 隐私与数据安全

- `.env`（API Key、管理密码）、`person_knowledge.json`（含手机号、邮箱、住址等个人信息）、`site_config.json`、`static/resume_data.js` 均被 `.gitignore` 忽略，**只存在于本地，不会提交到 Git 仓库**
- 手机号等敏感字段只存在于服务端知识库：不写入前端数据文件（`static/resume_data.js`），也不进入 AI 对话上下文
- 克隆本仓库后，需按「快速开始」自行准备上述本地文件（首次通过管理后台保存后自动生成）
- 部署到公网前，务必修改 `.env` 中的 `ADMIN_PASSWORD`（默认占位 `change-me`）

## 常见问题

- **回答「知识库暂无相关内容」**：问题与知识库相似度低于阈值（0.35），或该信息确实没填进 `person_knowledge.json`。
- **请求变慢 / 排队**：并发上限由 `.env` 的 `LLM_MAX_CONCURRENCY` 控制（MiMo 限速 100 RPM），突发请求会按序排队（不是报错）。
- **HTTP 429**：触发上游限流（RPM / 额度限制），稍等几秒重试即可。
- **API Key 无效**：`.env` 里 `LLM_API_KEY` 过期或填错，更新后重启服务。`/health` 可快速确认配置状态。

## 技术架构

```
访客提问 → BGE 向量化 → 余弦相似度召回 Top-4 知识片段 → 拼接上下文 → 大模型流式生成 → 打字机输出
```

- 相似度低于 0.35 视为「知识库无相关内容」，闲聊类问题则交给大模型自然回应
- 提示词强约束：个人信息类问题只允许依据知识库作答，禁止编造
- 手机号等敏感信息不进入 AI 上下文，被问联系方式时提供邮箱
- 静态目录只暴露 `static/`，`.env`、代码、模型文件不会被 HTTP 访问
