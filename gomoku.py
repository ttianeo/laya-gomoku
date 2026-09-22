"""五子棋(Gomoku) —— 基于 laya System 1 决策模型的完整对弈程序。

【棋盘与获胜条件】
- 棋盘 15×15(BOARD_SIZE),黑先白后,轮流在交叉点落子;
- 任意横/竖/斜方向连成五子及以上(WIN_COUNT)即获胜(自由规则,长连亦胜);
- 棋盘落满无人成五为平局。

【System 1 决策模型的适配 —— 输入/输出映射】
laya 是"快而直觉"的单次前向传播决策引擎(对照:极大极小/alpha-beta 搜索属于
慢思考的 System 2,本程序刻意不做搜索,只用 一次前向 + 快速启发式,贴合
System 1 定位)。main.py 演示的是邮件分派决策,这里把同一套 predict 接口
迁移到"落子决策":

  输入 state(棋盘状态)          —— Board.to_state() 生成紧凑字典:
      game                  对局说明文字;
      to_move               当前行棋方(black/white)   → "当前玩家"输入;
      stones_newest_first   按落子顺序(新着在前)的黑白子列表 → 棋盘状态。
      说明:模型输入序列长度有限(max_len=512),因此用"落子序列"而非
      15×15 矩阵表示棋盘,并把新着排在前,截断时优先保留近期局面。

  输入 questions(问题定义)      —— System1Agent._model_prior():
      move   {"type":"choice"}:候选落子点(空点、且与已有棋子相邻,按启发式
              预分取前 CAND_LIMIT 个)作为单选选项;
      danger {"type":"noul"}  :辅助问句"当前行棋方是否处于危险",与 move 在
              同一次前向传播中并行评估(近乎零成本),仅用于界面展示。

  输出 answers["move"]           —— 模型对候选点的输出:
      choice        最佳落子点标签(如 "H8")→ 解析回 (row, col) 即落子位置;
      probabilities 每个候选点的落子概率 → 作为策略先验,与启发式分融合;
      confidence    归一化熵置信度(0~1)  → 显示在状态栏。
  输出 answers["danger"]["noul"] —— 危险概率(威胁指数,展示用)。

【决策融合(System1Agent.decide,关键函数)】
  1) 己方存在"落子即成五"的点  → 直接落子(强制,不接受模型否决);
  2) 对方存在"落子即成五"的点  → 封堵(强制);
  3) 其余局面:总分 = (1-w) × 启发式相对分 + w × 模型先验概率,取最大者。
     其中 w = MODEL_WEIGHT。模型未加载 / 预测失败 / 设置 GOMOKU_OFFLINE=1
     时,自动退化为纯启发式,程序依旧可玩。

【GUI】tkinter 实现:人机对战(玩家执黑)/ AI 自对弈两种模式;绘制棋盘、
棋子与最后一手标记;胜负时高亮连珠并弹窗提示;支持随时重新开始。
AI 计算在工作线程执行,通过 root.after 回到主线程更新界面,不阻塞 GUI。

运行:uv run gomoku.py   (需要本地 Python 带 tkinter;模型首次运行自动下载)
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import laya

# ---------------------------------------------------------------------------
# 1. 棋盘规则:15×15,五子连珠
# ---------------------------------------------------------------------------
BOARD_SIZE = 15          # 棋盘大小:15 行 × 15 列
WIN_COUNT = 5            # 获胜条件:同色五子连珠(横/竖/斜,长连亦胜)
EMPTY, BLACK, WHITE = 0, 1, 2
COLOR_NAME = {BLACK: "black", WHITE: "white"}
COLOR_TEXT = {BLACK: "黑方", WHITE: "白方"}
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))   # 横、竖、两条斜线


def point_label(r: int, c: int) -> str:
    """棋盘坐标 -> 棋谱标签:(7, 7) -> 'H8'(列字母 A-O + 行号 1-15)。

    标签同时用作 laya choice 问句的选项文本(短,节省模型输入预算),
    以及模型输出(answers["move"]["choice"])解析回坐标的钥匙。
    """
    return f"{chr(ord('A') + c)}{r + 1}"


class Board:
    """五子棋规则引擎:只负责棋盘状态与规则,不关心界面和 AI。"""

    def __init__(self, size: int = BOARD_SIZE) -> None:
        self.size = size
        self.reset()

    def reset(self) -> None:
        """清空棋盘,开始新对局(重新开始功能的基础)。"""
        self.grid = [[EMPTY] * self.size for _ in range(self.size)]
        self.history: list[tuple[int, int, int]] = []   # (row, col, color),按落子顺序

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def place(self, r: int, c: int, color: int) -> list | None:
        """在 (r, c) 落子;若形成五连,返回获胜连线的坐标列表,否则返回 None。"""
        if not self.in_bounds(r, c) or self.grid[r][c] != EMPTY:
            raise ValueError(f"非法落子位置:({r}, {c})")
        self.grid[r][c] = color
        self.history.append((r, c, color))
        return self._win_line(r, c, color)

    def to_move(self) -> int:
        """当前行棋方:黑先白后,由落子手数奇偶决定。"""
        return BLACK if len(self.history) % 2 == 0 else WHITE

    def _win_line(self, r: int, c: int, color: int) -> list | None:
        """检查以 (r, c) 为最后一子是否连成 WIN_COUNT 子;返回连线坐标。"""
        for dr, dc in DIRS:
            cells = [(r, c)]
            for step in (1, -1):
                rr, cc = r + dr * step, c + dc * step
                while self.in_bounds(rr, cc) and self.grid[rr][cc] == color:
                    cells.append((rr, cc))
                    rr += dr * step
                    cc += dc * step
            if len(cells) >= WIN_COUNT:
                cells.sort()
                return cells
        return None

    def neighbourhood_candidates(self, radius: int = 2) -> list[tuple[int, int]]:
        """候选落子点:空点、且与任意已有棋子的切比雪夫距离 ≤ radius。

        五子棋的好点总在已有棋子附近,借此把 225 个点压缩到几十个;
        空盘时返回天元(中心点)。
        """
        if not self.history:
            m = self.size // 2
            return [(m, m)]
        cand: set[tuple[int, int]] = set()
        for r, c, _ in self.history:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
                    if self.in_bounds(rr, cc) and self.grid[rr][cc] == EMPTY:
                        cand.add((rr, cc))
        return sorted(cand)

    def to_state(self) -> dict:
        """生成 laya 模型的 state 输入(模型输入映射,见模块 docstring)。"""
        stones = [f"{'b' if color == BLACK else 'w'}:{point_label(r, c)}"
                  for r, c, color in reversed(self.history[-48:])]   # 新着在前,截断友好
        return {
            "game": "gomoku 15x15, five in a row wins",
            "to_move": COLOR_NAME[self.to_move()],
            "stones_newest_first": stones,
        }

    def clone(self) -> "Board":
        """深拷贝,供 AI 工作线程只读快照使用,避免与界面线程竞争。"""
        b = Board(self.size)
        b.grid = [row[:] for row in self.grid]
        b.history = list(self.history)
        return b


# ---------------------------------------------------------------------------
# 2. 快速启发式评估(System 1 的"直觉"兜底,亦用于候选点预筛)
# ---------------------------------------------------------------------------
WIN_SCORE = 10_000_000
# (连子数, 开放端数) -> 棋型分。开放端越多越有价值;连五即胜。
PATTERN_SCORE = {
    (5, 2): WIN_SCORE, (5, 1): WIN_SCORE, (5, 0): WIN_SCORE,
    (4, 2): 500_000,    # 活四:两头开放,下一手必成五
    (4, 1): 60_000,     # 冲四:一头开放
    (4, 0): 0,          # 死四:无法成五
    (3, 2): 9_000,      # 活三
    (3, 1): 1_000,
    (3, 0): 0,
    (2, 2): 350,
    (2, 1): 60,
    (2, 0): 0,
    (1, 2): 25,
    (1, 1): 5,
    (1, 0): 0,
}


def _line_stats(grid: list[list[int]], r: int, c: int, dr: int, dc: int, color: int):
    """假设 color 已落在 (r,c),统计该方向上的连子数与两端开放数(不修改棋盘)。"""
    cnt, open_ends = 1, 0
    size = len(grid)
    for step in (1, -1):
        rr, cc = r + dr * step, c + dc * step
        while 0 <= rr < size and 0 <= cc < size and grid[rr][cc] == color:
            cnt += 1
            rr += dr * step
            cc += dc * step
        if 0 <= rr < size and 0 <= cc < size and grid[rr][cc] == EMPTY:
            open_ends += 1
    return min(cnt, 5), open_ends


def point_score(grid: list[list[int]], r: int, c: int, color: int) -> int:
    """(r, c) 若由 color 落子,四个方向棋型分之和(纯函数)。"""
    return sum(PATTERN_SCORE[_line_stats(grid, r, c, dr, dc, color)] for dr, dc in DIRS)


def evaluate_point(board: Board, r: int, c: int, color: int) -> float:
    """启发式总分 = 己方进攻分 + 0.85 × 对方在此点的进攻分(防守权重略低)。"""
    attack = point_score(board.grid, r, c, color)
    defend = point_score(board.grid, r, c, 3 - color)
    return attack + 0.85 * defend


# ---------------------------------------------------------------------------
# 3. System 1 落子决策:laya 模型 + 启发式融合
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("LAYA_GOMOKU_MODEL", "convaiinnovations/laya-multilingual")
MODEL_WEIGHT = 0.40    # 非强制局面下,模型先验在总分中的权重 w(启发式占 0.6)
CAND_LIMIT = 20        # 送入模型的候选点上限(受模型 head_max_len 选项预算约束)
CAND_RADIUS = 2        # 候选点与已有棋子的最大切比雪夫距离


class System1Agent:
    """将 laya System 1 决策模型适配为五子棋落子决策(映射关系见模块 docstring)。"""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None
        self._lock = threading.Lock()
        self.loaded = False
        self.load_error: str | None = None
        self.offline = os.environ.get("GOMOKU_OFFLINE") == "1"   # 跳过模型,纯启发式

    @property
    def available(self) -> bool:
        return self.loaded and self._model is not None

    def ensure_loaded(self) -> None:
        """懒加载 laya 模型(首次调用耗时,需网络下载);失败则记录并降级。"""
        if self.offline or self.loaded or self.load_error:
            return
        with self._lock:
            if self.loaded or self.load_error:
                return
            try:
                self._model = laya.load(self.model_id)
                self.loaded = True
            except Exception as exc:      # 网络不通、依赖缺失等 → 纯启发式兜底
                self.load_error = f"{type(exc).__name__}: {exc}"

    def decide(self, board: Board, color: int) -> tuple[tuple[int, int], dict]:
        """决策入口:返回 ((row, col), info);info 含决策方式与模型指标供界面展示。"""
        cands = board.neighbourhood_candidates(CAND_RADIUS)
        if not cands:
            raise ValueError("棋盘已满")
        if not board.history:                       # 空盘:直接占天元
            return cands[0], {"mode": "opening"}

        scored = {p: evaluate_point(board, p[0], p[1], color) for p in cands}

        # ① 强制:己方"落子即成五"
        wins = [p for p, s in scored.items() if s >= WIN_SCORE]
        if wins:
            return wins[0], {"mode": "win"}

        # ② 强制:封堵对方"落子即成五"的点
        opp = 3 - color
        blocks = [p for p in cands if point_score(board.grid, p[0], p[1], opp) >= WIN_SCORE]
        if blocks:
            return max(blocks, key=lambda p: scored[p]), {"mode": "block"}

        # ③ 常规局面:启发式相对分 + 模型先验融合,取总分最大者
        model_info = self._model_prior(board, color, cands, scored)
        hmax = max(scored.values()) or 1.0
        weight = MODEL_WEIGHT if model_info else 0.0
        best, best_total = cands[0], -1.0
        for p in cands:
            h = scored[p] / hmax                    # 启发式相对分 ∈ [0, 1]
            prior = model_info["probs"].get(point_label(*p), 0.0) if model_info else 0.0
            total = (1 - weight) * h + weight * prior
            if total > best_total:
                best, best_total = p, total

        info = {"mode": "model+heuristic" if model_info else "heuristic"}
        if model_info:
            info.update(confidence=model_info["confidence"],
                        threat=model_info["threat"],
                        move_prob=model_info["probs"].get(point_label(*best), 0.0))
        return best, info

    def _model_prior(self, board, color, cands, scored) -> dict | None:
        """调用 laya 模型,获取候选点的落子概率先验;任何失败返回 None(兜底启发式)。

        模型集成方式:一次 predict 调用同时提交两个问句(choice + noul),
        laya 在内部将它们批进同一次前向传播 —— 这正是 System 1 的"快"。
        """
        try:
            self.ensure_loaded()
            if not self.available:
                return None
            # 候选点过多会超出模型选项长度预算,按启发式预分取前 CAND_LIMIT 个
            top = sorted(cands, key=lambda p: scored[p], reverse=True)[:CAND_LIMIT]
            labels = {point_label(r, c): (r, c) for r, c in top}
            questions = {
                "move": {                                   # 输出:落子位置 + 各点概率
                    "type": "choice",
                    "instructions": ("Gomoku on a 15x15 board, five in a row wins. "
                                     "It is %s's turn. Pick the best empty point for %s."
                                     % (COLOR_NAME[color], COLOR_NAME[color])),
                    "criteria": {lab: None for lab in labels},
                },
                "danger": {                                 # 输出:威胁指数(展示用)
                    "type": "noul",
                    "instructions": ("In this position, is the side to move in immediate "
                                     "danger of the opponent completing five in a row "
                                     "with their next move?"),
                },
            }
            result = self._model.predict(board.to_state(), questions)

            move_ans = result["answers"]["move"]            # 输出映射:choice -> 坐标
            probs = {k: float(v) for k, v in move_ans["probabilities"].items() if k in labels}
            s = sum(probs.values()) or 1.0
            probs = {k: v / s for k, v in probs.items()}    # 重新归一化,防数值误差
            threat = float(result["answers"]["danger"]["noul"])
            return {"probs": probs,
                    "confidence": move_ans.get("confidence"),
                    "threat": threat}
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 4. 图形界面(tkinter):人机对战 / AI 自对弈 / 重新开始
# ---------------------------------------------------------------------------
class GomokuGUI:
    """棋盘绘制、对局流程控制与胜负提示。"""

    CELL = 38      # 交叉点间距(像素)
    MARGIN = 32    # 棋盘边距(像素)
    MODE_PVE = "人机对战(玩家执黑)"
    MODE_EVE = "AI 自对弈"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.agent = System1Agent()
        self.board = Board()
        self.game_over = False
        self.win_line: list | None = None
        self.gen = 0               # 对局代数:重新开始后,旧 AI 线程的结果自动作废
        root.title("五子棋 · laya System 1 决策模型")
        root.resizable(False, False)
        self._build_widgets()
        self.restart()

    # ---------- 界面构建与绘制 ----------
    def _build_widgets(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)
        self.mode_var = tk.StringVar(value=self.MODE_PVE)
        ttk.Label(top, text="模式:").pack(side=tk.LEFT)
        mode_box = ttk.Combobox(top, textvariable=self.mode_var, state="readonly",
                                values=[self.MODE_PVE, self.MODE_EVE], width=20)
        mode_box.pack(side=tk.LEFT, padx=4)
        mode_box.bind("<<ComboboxSelected>>", lambda _e: self.restart())
        ttk.Button(top, text="重新开始", command=self.restart).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="退出", command=self.root.destroy).pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="准备中…")
        ttk.Label(self.root, textvariable=self.status_var, padding=(8, 2)).pack(fill=tk.X)

        side = self.MARGIN * 2 + self.CELL * (BOARD_SIZE - 1)
        self.canvas = tk.Canvas(self.root, width=side, height=side,
                                bg="#e8c88f", highlightthickness=0)
        self.canvas.pack(padx=8, pady=(0, 8))
        self.canvas.bind("<Button-1>", self._on_click)

    def _rc_to_xy(self, r: int, c: int) -> tuple[int, int]:
        return self.MARGIN + c * self.CELL, self.MARGIN + r * self.CELL

    def _redraw(self) -> None:
        """重绘整个棋盘:网格、星位、坐标、棋子、最后一手标记、获胜连线。"""
        cv, n = self.canvas, BOARD_SIZE
        cv.delete("all")
        for i in range(n):                       # 网格线
            cv.create_line(self.MARGIN + i * self.CELL, self.MARGIN,
                           self.MARGIN + i * self.CELL, self.MARGIN + (n - 1) * self.CELL,
                           fill="#7a5230")
            cv.create_line(self.MARGIN, self.MARGIN + i * self.CELL,
                           self.MARGIN + (n - 1) * self.CELL, self.MARGIN + i * self.CELL,
                           fill="#7a5230")
        for r, c in ((3, 3), (3, 11), (7, 7), (11, 3), (11, 11)):   # 星位
            x, y = self._rc_to_xy(r, c)
            cv.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#7a5230", outline="")
        for i in range(n):                       # 坐标标注
            x, _ = self._rc_to_xy(0, i)
            cv.create_text(x, 12, text=chr(ord("A") + i), font=("Arial", 8), fill="#555")
            _, y = self._rc_to_xy(i, 0)
            cv.create_text(12, y, text=str(i + 1), font=("Arial", 8), fill="#555")
        for r in range(n):                       # 棋子
            for c in range(n):
                color = self.board.grid[r][c]
                if color != EMPTY:
                    x, y = self._rc_to_xy(r, c)
                    rad = self.CELL * 0.42
                    cv.create_oval(x - rad, y - rad, x + rad, y + rad,
                                   fill="#111" if color == BLACK else "#f5f5f5",
                                   outline="#333", width=1)
        if self.board.history and not self.game_over:               # 最后一手标记
            lr, lc, _ = self.board.history[-1]
            x, y = self._rc_to_xy(lr, lc)
            rad = self.CELL * 0.18
            cv.create_oval(x - rad, y - rad, x + rad, y + rad, outline="#e33", width=2)
        if self.win_line:                                            # 获胜连线高亮
            (r0, c0), (r1, c1) = self.win_line[0], self.win_line[-1]
            cv.create_line(*self._rc_to_xy(r0, c0), *self._rc_to_xy(r1, c1),
                           fill="#e33", width=3)

    # ---------- 对局流程 ----------
    def restart(self) -> None:
        """重新开始:清空棋盘;自对弈模式则让黑方 AI 先行。"""
        self.gen += 1                             # 使仍在运行的旧 AI 线程结果失效
        self.board.reset()
        self.game_over = False
        self.win_line = None
        self._redraw()
        if self.mode_var.get() == self.MODE_EVE:
            self.status_var.set("AI 自对弈开始,黑方思考中…")
            self.root.after(400, self._schedule_ai)
        else:
            self.status_var.set("你执黑先行,请点击棋盘交叉点落子")

    def _on_click(self, event) -> None:
        """人类玩家点击落子(仅人机模式、己方回合、位置有效时)。"""
        if self.game_over or self.mode_var.get() == self.MODE_EVE:
            return
        if self.board.to_move() != BLACK:         # AI 回合中
            self.status_var.set("AI 思考中,请稍候…")
            return
        c = round((event.x - self.MARGIN) / self.CELL)
        r = round((event.y - self.MARGIN) / self.CELL)
        if not self.board.in_bounds(r, c) or self.board.grid[r][c] != EMPTY:
            return
        x, y = self._rc_to_xy(r, c)               # 距交叉点太远视为误触
        if abs(event.x - x) > self.CELL * 0.45 or abs(event.y - y) > self.CELL * 0.45:
            return
        if not self._apply_move(r, c, BLACK):     # 未分胜负 → 轮到 AI
            self._schedule_ai()

    def _apply_move(self, r: int, c: int, color: int) -> bool:
        """落子并处理胜负;返回 True 表示对局已结束。"""
        line = self.board.place(r, c, color)
        self._redraw()
        if line:                                  # 五子连珠 → 胜负提示
            self.game_over = True
            self.win_line = line
            self._redraw()
            if self.mode_var.get() == self.MODE_EVE:
                text = f"{'黑方' if color == BLACK else '白方'} AI 获胜!"
            else:
                text = "玩家(黑方)获胜!" if color == BLACK else "AI(白方)获胜!"
            self.status_var.set(f"对局结束 —— {text}")
            messagebox.showinfo("五子棋", f"五子连珠!{text}")
            return True
        if len(self.board.history) == self.board.size ** 2:         # 平局
            self.game_over = True
            self.status_var.set("对局结束 —— 棋盘已满,平局")
            messagebox.showinfo("五子棋", "棋盘已满,平局!")
            return True
        return False

    def _schedule_ai(self) -> None:
        """在后台线程执行 AI 决策,完成后经 root.after 回到主线程落子。"""
        if self.game_over:
            return
        self.gen += 1
        gen, color = self.gen, self.board.to_move()
        snapshot = self.board.clone()             # 工作线程只读快照,避免线程竞争
        self.status_var.set(f"{COLOR_TEXT[color]}(AI)思考中…")
        threading.Thread(target=self._ai_worker, args=(gen, snapshot, color),
                         daemon=True).start()

    def _ai_worker(self, gen: int, snapshot: Board, color: int) -> None:
        """工作线程:模型加载与决策都可能耗时,绝不能在此触碰界面控件。"""
        try:
            move, info = self.agent.decide(snapshot, color)
        except Exception as exc:
            move, info = None, {"error": str(exc)}
        self.root.after(0, self._apply_ai, gen, move, info, color)

    def _apply_ai(self, gen: int, move, info: dict, color: int) -> None:
        """主线程:应用 AI 的落子结果(若期间已重新开始则丢弃)。"""
        if gen != self.gen or self.game_over:
            return
        if move is None:
            self.status_var.set(f"AI 决策出错:{info.get('error', '未知错误')}")
            return
        r, c = move
        if self._apply_move(r, c, color):
            return
        describe = {"win": "一步成五", "block": "封堵威胁", "opening": "开局天元",
                    "model+heuristic": "模型+启发式", "heuristic": "纯启发式"}
        extra = f"({describe.get(info.get('mode'), info.get('mode'))}"
        if "confidence" in info:                  # 展示模型输出的置信度与威胁指数
            extra += f",模型置信 {info['confidence']:.2f},威胁指数 {info['threat']:.2f}"
        elif info.get("mode") == "heuristic" and self.agent.load_error:
            extra += " | 模型加载失败,已用纯启发式"
        extra += ")"
        self.status_var.set(f"{COLOR_TEXT[color]} AI 落子 {point_label(r, c)} {extra}")
        if self.mode_var.get() == self.MODE_EVE and not self.game_over:
            self.root.after(500, self._schedule_ai)   # 自对弈:稍作停顿再下一手


def main() -> None:
    root = tk.Tk()
    GomokuGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
