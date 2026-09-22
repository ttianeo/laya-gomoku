# 五子棋 · laya System 1 决策模型

一个基于 [laya](https://pypi.org/project/laya/) System 1 决策模型的五子棋（Gomoku）对弈程序，附带 tkinter 图形界面。laya 是"快而直觉"的单次前向传播决策引擎——本程序刻意不做极大极小/alpha-beta 搜索（那属于慢思考的 System 2），而是用 **一次前向传播 + 快速启发式** 完成每一手落子决策。

## 功能特性

- 15×15 标准棋盘，黑先白后，横/竖/斜五子连珠获胜（长连亦胜）
- AI 落子决策：laya System 1 模型先验 + 棋型启发式融合
- 两种对弈模式：人机对战（玩家执黑）/ AI 自对弈
- 完整 GUI：棋盘绘制、最后一手标记、获胜连线高亮、胜负弹窗提示
- 一键重新开始；AI 计算在后台线程执行，界面不卡顿
- 模型不可用时自动降级为纯启发式，程序始终可玩

## 环境要求

- Python >= 3.12，且本地 Python 带 tkinter（官方安装包与 python-build-standalone 均默认包含）
- 依赖 `laya>=0.3.5`（会自动安装 torch、transformers 等）
- 首次运行会自动从 Hugging Face 下载 `convaiinnovations/laya-multilingual` 模型（约数百 MB，之后走本地缓存）

## 快速开始

使用 [uv](https://docs.astral.sh/uv/)（推荐，仓库自带锁文件）：

```bash
uv sync          # 安装依赖
uv run gomoku.py # 启动图形界面
```

或使用 pip：

```bash
pip install "laya>=0.3.5"
python gomoku.py
```

## 玩法说明

| 操作 | 说明 |
| --- | --- |
| 人机对战 | 你执黑先行，点击棋盘交叉点落子，AI 执白应答 |
| AI 自对弈 | 双方均为 AI，自动连续对局，可观察其决策过程 |
| 重新开始 | 清空棋盘，按当前模式重新开局 |
| 胜负提示 | 成五时红色连线高亮并弹窗提示；棋盘落满为平局 |

状态栏会显示 AI 每手的决策方式（一步成五 / 封堵威胁 / 模型+启发式 / 纯启发式）以及模型置信度与威胁指数。

## 工作原理

### System 1 决策模型的输入/输出映射

`main.py` 演示了 laya 的最小用法：`state`（情境描述）+ `questions`（结构化问题）→ `answers`（带概率的决策）。本程序把同一套接口迁移到落子决策：

| laya 接口 | 五子棋映射 |
| --- | --- |
| `state`（棋盘状态） | `Board.to_state()` 生成紧凑字典：对局说明 + `to_move`（当前玩家）+ `stones_newest_first`（按落子顺序的黑白子列表，新着在前，规避模型 512 token 截断） |
| `questions["move"]`（choice） | 候选落子点（空点、且与已有棋子相邻，按启发式预分取前 20 个）作为单选选项 |
| `questions["danger"]`（noul） | 辅助问句"当前行棋方是否处于危险"，与 `move` 在同一次前向传播中并行评估 |
| `answers["move"]["choice"]` | 最佳落子点标签（如 `"H8"`），解析回 `(row, col)` 即落子位置 |
| `answers["move"]["probabilities"]` | 每个候选点的落子概率，作为策略先验 |
| `answers["move"]["confidence"]` / `answers["danger"]["noul"]` | 置信度 / 威胁指数，显示在状态栏 |

### 决策融合（`System1Agent.decide`）

1. **强制取胜**：己方存在"落子即成五"的点 → 直接落子；
2. **强制封堵**：对方存在"落子即成五"的点 → 封堵；
3. **常规局面**：`总分 = 0.6 × 启发式相对分 + 0.4 × 模型先验概率`，取最大者落子。

启发式部分按（连子数 × 开放端数）评估活四/冲四/活三等棋型，同时承担候选点预筛与模型不可用时的完整兜底。

## 项目结构

```
layademo/
├── gomoku.py        # 五子棋完整程序：规则引擎 + System 1 决策 + tkinter GUI
├── main.py          # laya System 1 决策模型最小示例（邮件分派）
├── pyproject.toml   # 项目与依赖配置
└── README.md
```

`gomoku.py` 内部分为四块，注释中说明了各关键函数的作用与模型集成方式：

1. `Board` —— 规则引擎：落子、五连判定、候选点生成、模型状态序列化；
2. 启发式评估 —— `_line_stats` / `point_score` / `evaluate_point`：棋型打分；
3. `System1Agent` —— 模型加载、候选点先验获取（`_model_prior`）、三段式决策（`decide`）；
4. `GomokuGUI` —— 界面绘制、人机/AI 对弈流程、线程调度与胜负提示。

## 环境变量

| 变量 | 作用 |
| --- | --- |
| `GOMOKU_OFFLINE=1` | 跳过模型加载，使用纯启发式 AI（离线可玩） |
| `LAYA_GOMOKU_MODEL` | 替换模型 ID（默认 `convaiinnovations/laya-multilingual`），也可指向本地模型目录 |
| `HF_TOKEN` | Hugging Face 访问令牌（如下载受限） |

## License

本项目以 [MIT](LICENSE) 许可证开源发布。
