# 果蠅走迷宮 v2.1：連接組模型自動控制

預設由隨附的 FlyWire 連接組子網路模型自動控制迷宮。
視窗右側顯示 512 個節點狀態與實際動作輸出，並可匯出完整感知／動作紀錄。
先看 [Windows／macOS 完整安裝與執行方式](RUNNING.md)，模型操作見 [V2_1_GUIDE.md](V2_1_GUIDE.md)。
可下載 [v2.1.0 發行包](https://github.com/Sloth0528/fly-maze/releases/tag/v2.1.0)，或複製本儲存庫。

| 平台 | Python | 安裝後啟動 |
|---|---|---|
| Windows x64 | 64 位元 Python 3.12 | 雙擊 `START_V2_1.cmd` |
| macOS Apple Silicon（M 系列） | 原生 arm64 Python 3.12 | `bash START_V2_1.command` |
| macOS Intel／Rosetta x86_64 | 目前相依套件不支援 | 請勿直接套用 M 系列指令 |

第一次使用，在完整解壓後的專案資料夾執行：

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
& ".\.venv\Scripts\python.exe" run_v2_1.py
```

**macOS Apple Silicon Terminal**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_v2_1.py
```

macOS 請使用 [Python.org 的 macOS 安裝程式](https://www.python.org/downloads/macos/) 安裝 Python 3.12，
並以 `.venv/bin/python -m tkinter` 檢查圖形環境。macOS 不使用 Windows 的 CPU wheel 索引。
Windows 已做本機圖形及回歸測試；跨平台無視窗測試結果請見 [GitHub Actions](https://github.com/Sloth0528/fly-maze/actions)。
遠端測試與 macOS 桌面人工驗收的範圍分別記錄於 [執行指南](RUNNING.md) 與 [驗證紀錄](V2_1_VALIDATION.md)。

這版已接通模型控制迴路；附帶的 120 回合試跑模型尚未學會可靠導航。
模型使用真實連線子圖與人工感覺／動作介面，不是完整真實果蠅大腦重建。

---

# v2.0：連續環境與模型比較

**v2.0 已加入連續座標、任意角度轉向，以及 GRU／真實 FlyWire 連接組子網路的比較流程。**
操作與設計請先讀 [V2_GUIDE.md](V2_GUIDE.md)。原本的離散格子版本與模型介面仍可使用。

```powershell
.\.venv\Scripts\python.exe -m flymaze_v2 preview --difficulty easy --seed 7
```

新版連接組模型使用隨附的成蟲實測連線子圖；感覺輸入、動作讀出和學習方法由本專案設計，
並非完整真實果蠅大腦重建。短訓練已跑通，尚未證實可靠導航或優於 GRU。

---

# v1：六個 Python 模組入門範本

這是以導航概念設計的 AI 教學範本，不是真實果蠅神經連接組重建。
採用 PyTorch 的感覺層、GRU 記憶、三動作輸出，以及按回合更新的 Actor–Critic。
為了讓核心程式容易閱讀，這裡沒有實作 PPO 的 clipping、多輪更新或平行環境。

## 新增：可重現的迷宮實驗環境

目前已加入三種難度的隨機迷宮、每回合換圖、以身體朝向對齊且有牆壁遮擋的局部視野，
以及不需模型的手動預覽視窗。完整規格與訓練方式見 [MAZE_DESIGN.md](MAZE_DESIGN.md)。

```powershell
.\.venv\Scripts\python.exe preview.py --difficulty easy --seed 7
.\.venv\Scripts\python.exe train.py --difficulty easy --vary-maze --random-start --episodes 1000 --output outputs/maze_easy_v2.pt
.\.venv\Scripts\python.exe evaluate.py --model outputs/maze_easy_v2.pt --difficulty easy --vary-maze --random-start --seed 10007 --episodes 50
```

方向鍵 ↑ 前進、← 左轉、→ 右轉，R 重設。右側顯示模型視野；左側全圖只供人觀察。
新環境仍提供 83 維觀察與三種動作，新增設定會隨模型儲存。
未指定新選項時，保留下方原始六模組範本的預設行為；新模式的感知與獎勵請以設計文件為準。

## 檔案分工與閱讀順序

| 檔案 | 主要入口 | 責任 |
|---|---|---|
| `fly.py` | `Fly.propose()` | 保存座標與朝向；提出前進位置或原地轉彎 |
| `maze.py` | `Maze.reset/observe/step()` | 地圖、局部觀察、碰撞、獎勵與成功判定 |
| `brain.py` | `Brain.forward/act()` | 83 維觀察 → 64 維感覺層 → 64 維 GRU → 動作與價值 |
| `train.py` | `train_episode()` | 收集一回合經驗、計算回報、反向傳播與更新權重 |
| `evaluate.py` | `run_episode()` | 載入模型，統計成功率、動作步數與撞牆次數 |
| `render.py` | `play()` | 顯示路徑、動作機率、價值與前四個 GRU 狀態值 |

## Windows PowerShell 執行方式

先在終端機切換到放有這六個檔案的資料夾。建議 Python 3.12。
如果已經有 `.venv`，不需要重建；安裝也只需要做一次。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

1. 訓練並儲存模型：

```powershell
.\.venv\Scripts\python.exe train.py --episodes 1000
```

產生 `outputs/fly_brain.pt` 模型與 `outputs/fly_brain.csv` 訓練紀錄。
為保留舊實驗，不會覆寫既有輸出。再次訓練請指定新名稱：

```powershell
.\.venv\Scripts\python.exe train.py --episodes 1000 --output outputs/run2.pt
```

2. 評估（不進行權重更新）：

```powershell
.\.venv\Scripts\python.exe evaluate.py --episodes 20
.\.venv\Scripts\python.exe evaluate.py --episodes 50 --random-start
```

預設每次選最高機率動作；`--sample` 改為按機率抽樣。
固定起點、固定朝向和 greedy 策略下，各回合會相同；重複成功不是泛化證據。
`--random-start` 會在同一張地圖的可達道路中抽取起點與朝向，也不等同於陌生地圖測試。

3. 觀看終端機動畫或圖形視窗：

```powershell
.\.venv\Scripts\python.exe render.py
.\.venv\Scripts\python.exe render.py --gui
```

圖形視窗使用 Python 的 Tkinter；地圖中的紅色圓點為果蠅，箭頭為朝向，
藍色格子為已走過路徑，綠色 G 是食物。關閉視窗即可結束。
若缺少 Tkinter，可先用終端機模式；觀看時不訓練。
其他模型可用 `evaluate.py --model outputs/run2.pt` 或 `render.py --model outputs/run2.pt --gui`。

## 環境與觀察

- 地圖符號：`#` 牆、`.` 道路、`S` 起點、`G` 食物。各列須等長且 S/G 各一個。
- 動作：0 前進一格、1 原地左轉 90 度、2 原地右轉 90 度。
- 一步指一次動作，包含轉彎；不是只計算移動格數。
- 5×5 局部地圖有牆、道路、食物三通道，共 75 個值。
- 加上精確朝向 one-hot 4 個、上一動作 one-hot 3 個與撞牆 1 個，共 83 個值。
- 視野採世界座標對齊，且提供理想朝向；這是方便教學的感覺假設，尚未模擬視覺或有誤差的神經羅盤。
- AI 看不到完整地圖、絕對座標、出口座標或全局食物距離。畫面可以顯示全圖。
- 每回合重設 GRU。GRU 提供歷史線索，並不保證建立了可解任意迷宮的地圖。
- 每動作 -0.01、撞牆額外 -0.1、本回合首次進入道路格額外 +0.05、抵達食物 +10。
- 探索獎勵是教學用的人為設定，不是已證實的果蠅獎勵機制；轉彎和重訪不會重複領取。
- 食物同時在動作前後的 5×5 視野內時，每接近一格額外 +0.2、遠離一格額外 -0.2。
  這是幫助入門模型學習的人為提示，不計算穿牆最短路徑；有障礙時接近目標不一定是正確路線。
- 訓練預設最多 40 動作，可用 `--max-steps 100` 調整；直接建立 `Maze()` 的上限是 100。
- 預設是容易接觸到食物的小型岔路地圖，供驗證流程；不代表已經學會複雜迷宮導航。

## 學習發生在哪裡？

```text
Maze.observe() → Brain.act() → Maze.step() → 收集動作、價值與獎勵
                                             ↓
                           計算折扣回報與 advantage
                                             ↓
                      loss.backward() → optimizer.step()
```

actor 輸出三個動作的 logits，再轉成機率；critic 估計折扣後的未來回報。
GRU 隱藏狀態保存本回合的歷史資訊。`optimizer.step()` 才是改變模型參數的位置。
訓練包含少量 entropy 獎勵以鼓勵探索；成功結束與時間截斷分開處理。
模型檔儲存權重、網路尺寸與訓練地圖；不含 optimizer 狀態，這版不提供續訓。

## 自訂地圖與循序練習

把下面內容存成 UTF-8 文字檔，例如 `new_maze.txt`：

```text
#########
#S..#..G#
#.#.#.#.#
#.#...#.#
#.#####.#
#.......#
#########
```

```powershell
.\.venv\Scripts\python.exe train.py --maze new_maze.txt --episodes 2000 --output outputs/custom.pt
.\.venv\Scripts\python.exe evaluate.py --model outputs/fly_brain.pt --maze new_maze.txt --episodes 20 --random-start
```

第二行才是在替換地圖測試既有模型；不保證成功。

建議依序練習：修改獎勵 → 修改地圖 → 隨機起點訓練 → 測試未見地圖。
原始預設只訓練一張地圖；若要抽取多張迷宮，使用新增的 `--difficulty easy --vary-maze`。
訓練結果受種子、回合數與地圖難度影響。短跑成功僅驗證流程，不代表已收斂。

## 原始六模組範本的實際驗證

使用 Python 3.12、PyTorch 2.14.1+cpu 與 NumPy 2.5.2。
已檢查局部觀察、碰撞、獎勵、成功與時間截斷、無效地圖拒絕、GRU 梯度、
實際權重更新、模型儲存／讀取，以及評估和觀看時不更新權重。
Tkinter 視窗的建立、地圖繪製、狀態文字與自動更新也完成程式化檢查。

最終設定以 seed=7 訓練 600 回合，之後固定起點、greedy 評估 5 回合成功 0 回合。
因此目前交付的是**可執行且可修改的學習流程範本，尚未訓練出穩定的迷宮解題策略**。
它適合用來練習調整探索、獎勵與訓練流程；若目標是展示可靠解題，仍需改進並重新評估。
壓縮檔只包含原始碼與說明，不附帶這些試跑模型。

## API 參考

- [PyTorch GRUCell](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html)
- [PyTorch Categorical](https://docs.pytorch.org/docs/stable/distributions.html#categorical)
- [PyTorch 序列化](https://docs.pytorch.org/docs/stable/notes/serialization.html)
