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

## 檔案分工與閱讀順序

| 檔案 | 主要入口 | 責任 |
|---|---|---|
| `fly.py` | `Fly.propose()` | 保存座標與朝向；提出前進位置或原地轉彎 |
| `maze.py` | `Maze.reset/observe/step()` | 地圖、局部觀察、碰撞、獎勵與成功判定 |
| `brain.py` | `Brain.forward/act()` | 83 維觀察 → 64 維感覺層 → 64 維 GRU → 動作與價值 |
| `train.py` | `train_episode()` | 收集一回合經驗、計算回報、反向傳播與更新權重 |
| `evaluate.py` | `run_episode()` | 載入模型，統計成功率、動作步數與撞牆次數 |
| `render.py` | `play()` | 顯示路徑、動作機率、價值與前四個 GRU 狀態值 |

## API 參考

- [PyTorch GRUCell](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html)
- [PyTorch Categorical](https://docs.pytorch.org/docs/stable/distributions.html#categorical)
- [PyTorch 序列化](https://docs.pytorch.org/docs/stable/notes/serialization.html)
