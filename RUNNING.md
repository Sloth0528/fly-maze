# Windows 與 macOS 執行方式

先完整解壓縮 [v2.1.1 發行包](https://github.com/Sloth0528/fly-maze/releases/tag/v2.1.1)，
或使用 `git clone https://github.com/Sloth0528/fly-maze.git` 取得程式與內附模型。
不要在壓縮檔內直接啟動，也不要把另一台電腦的 `.venv` 搬過來使用。

| 系統 | 執行方式 | 驗證範圍 |
|---|---|---|
| Windows x64 | Python 3.12、PyTorch CPU、Tkinter | 已有本機 GUI／訓練／評估測試；另有 CI |
| macOS Apple Silicon（M1、M2、M3、M4 等） | 原生 arm64 Python 3.12、PyPI 套件、Tkinter | 已提供 macOS 啟動器；CI 狀態以 Actions 為準，未做實機桌面人工驗收 |
| Intel Mac／Rosetta 下的 x86_64 Python | 本發行版不支援 | `torch>=2.6` 無對應官方 macOS x86_64 wheel |

本專案使用 CPU 推論／訓練，不要求 CUDA 或 MPS。下載模型後，播放不需連線。
預設引導模型已通過原始小迷宮的獨立起點測試；隨機生成或陌生迷宮尚未建立可靠表現。
請完整保留 `models/default_connectome.json` 與其中指定的模型，才能使用已驗證的新預設。

## Windows x64

1. 安裝 **64 位元 Python 3.12**，包含 pip 與 `py` 啟動器。可從 [Python Windows 官方下載頁](https://www.python.org/downloads/windows/) 取得。
2. 解壓到例如 `C:\FlyMaze`，在 PowerShell 執行：

```powershell
cd "C:\FlyMaze"
py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
& ".\.venv\Scripts\python.exe" run_v2_1.py --check
& ".\.venv\Scripts\python.exe" run_v2_1.py
```

之後可以雙擊 `START_V2_1.cmd`，不必重新安裝。
`--check` 會顯示系統、Python 位元、套件版本及載入的模型；它不會開啟 GUI。
若找不到 `py`，請確認已安裝 Python Launcher，或用已知 Python 3.12 執行檔的完整路徑建立 `.venv`。

## macOS Apple Silicon

1. 從 [Python macOS 官方下載頁](https://www.python.org/downloads/macos/) 安裝 **Python 3.12 universal2**。
   本版選擇 Python 3.12 作為共同比較環境，無須改用最新的 Python 大版本。
2. 在原生 Terminal 開啟解壓後的資料夾；下例假設名稱是 `fly-maze`，位於 Downloads。
3. 建立這台 Mac 專用的虛擬環境並安裝套件：

```bash
cd "$HOME/Downloads/fly-maze"
python3.12 -c 'import platform; print(platform.machine())'
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_v2_1.py --check
.venv/bin/python -m tkinter
.venv/bin/python run_v2_1.py
```

第一個檢查應輸出 `arm64`。`python -m tkinter` 會開啟測試視窗，關閉後再執行最後一行。
若解壓後資料夾名稱不同，請修改 `cd` 的路徑。
**Mac 使用一般 PyPI 安裝，不加 Windows 範例中的 `--index-url .../cpu`。**

安裝後可以直接在 Terminal 執行：

```bash
bash START_V2_1.command
```

若希望從 Finder 雙擊，可先在專案目錄執行一次：

```bash
chmod +x START_V2_1.command
```

啟動器會優先使用同資料夾的 `.venv/bin/python`。不必修改系統安全設定；若 Finder 不方便執行，
直接使用 Terminal 上述指令即可。

### Mac 常見問題

- **顯示 `x86_64`**：Intel Mac 不支援此套件組合；如果是 M 系列 Mac，請使用原生 Terminal 與 arm64 Python，避免 Rosetta。
- **`No module named _tkinter`**：這個 Python 沒有可用的 Tcl/Tk。優先使用 Python.org 安裝程式，重新建立此專案的環境。
  不要用 `pip install tkinter`；Tkinter 不是由這個 pip 套件安裝。
- **`No matching distribution found for torch`**：先檢查 Python 3.12、64 位元與 `arm64`；不要在 Intel Mac 上安裝舊版 torch 來繞過本版需求。
- **沒有桌面環境／GUI 測試失敗**：仍可使用下方無視窗指令，確認模型控制迴路；這不代表 GUI 已通過驗證。
- **M 系列 Mac 也沒有 GPU 加速**：本版刻意使用 CPU，方便跨平台比較與重現。

## 兩個平台的共同操作

Windows 在下列 Python 指令前使用 `& ".\.venv\Scripts\python.exe"`；Mac 使用 `.venv/bin/python`。

| 目的 | Python 後面的參數 |
|---|---|
| 連接組模型自動控制 | `run_v2_1.py` |
| 換 easy 迷宮 | `run_v2_1.py --difficulty easy --seed 7` |
| GRU 比較基準 | `run_v2_1.py --baseline` |
| 手動操作 | `run_v2_1.py --manual --difficulty easy` |
| 不開視窗執行 | `run_v2_1.py --headless --max-steps 80` |
| 使用自己的模型 | `run_v2_1.py --model outputs/my_connectome.pt` |
| 從頭訓練連接組模型 | `-m flymaze_v2 train --brain connectome --episodes 1000 --seed 22 --output outputs/my_connectome.pt` |
| 建議先練習的小迷宮引導訓練 | `-m flymaze_v2 train-guided --brain connectome --updates 300 --seed 42 --max-steps 160 --output outputs/my_guided.pt` |

模型與報告輸出不覆寫；重跑請換檔名。詳細模型、紀錄及評估說明見 [V2_1_GUIDE.md](V2_1_GUIDE.md)。
原始純強化學習指令仍保留作實驗，但已觀察到 1000 回合後的轉圈退化，不能以增加回合數當作可靠解法。
引導訓練使用訓練專用的全圖示範，播放時只有模型的局部感知；方法與限制見 [修復紀錄](FIX_STATIONARY.md)。

## 驗證範圍與官方參考

完整本機測試指令：`python -m unittest test_maze test_v2 test_v2_1 test_platform_support test_guided -v`。
[GitHub Actions](https://github.com/Sloth0528/fly-maze/actions) 會在 Windows 與 Apple Silicon macOS 執行無視窗的感知、幾何、模型、接線及啟動測試。
CI 不操作桌面視窗，因此 CI 成功也不等於 macOS GUI 人工驗收完成。

- [PyTorch 官方安裝說明](https://pytorch.org/get-started/locally/)
- [PyTorch 官方 macOS x86_64 停止建置公告](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690)
- [Python 官方 Tkinter 說明與測試方式](https://docs.python.org/3.12/library/tkinter.html)
- [GitHub 官方 runner 架構清單](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
