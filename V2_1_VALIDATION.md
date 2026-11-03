# v2.1.0 實際驗證

日期：2026-11-03（Asia/Taipei）。Windows、Python 3.12、PyTorch 2.14.1+cpu、NumPy 2.5.2。

## 接線與回歸測試

```powershell
& ".\.venv\Scripts\python.exe" -m unittest test_maze test_v2 test_v2_1 -v
```

**44 項通過**：原有 33 項，加上 11 項 v2.1 測試。

- 內附模型的 512 節點矩陣與 FlyWire 衍生檔逐值相符，版本與來源 SHA256 可追溯。
- `ControlSession` 的動作逐值等於同一步模型 `brain.act` 的輸出，沒有手動覆寫或額外規劃器。
- 下一步確實使用移動後的觀察及前一步神經記憶；重設會清除記憶及回合紀錄。
- 熱圖和紀錄中的 root ID／狀態值來自實際模型隱藏狀態，播放不更新權重。
- 測試中將連線矩陣歸零，下一次動作會改變，確認連接組遞迴路徑確實影響控制輸出。
  此消融只供接線測試，不代表所有神經元均必要或符合生物因果關係。
- 類型錯誤或遺失的模型不會觸發其他控制器；無效神經狀態會在移動前報錯。
- JSON 匯出可追溯且不覆寫；啟動程式在其他工作目錄仍能找到內附模型。
- 圖形模式共用同一 session；方向鍵不改寫自動動作，畫面顯示 512 個狀態格及實際動作。

圖形介面進行了 Tkinter 建立、繪圖、按鍵、暫停及數值對應的程式化檢查，沒有宣稱長時間人工驗收。

## 本次實際訓練

```powershell
& ".\.venv\Scripts\python.exe" -m flymaze_v2 train --brain connectome --episodes 120 --max-steps 160 --seed 21 --output models/connectome_v2_1.pt
& ".\.venv\Scripts\python.exe" -m flymaze_v2 train --brain gru --episodes 120 --max-steps 160 --seed 21 --output models/gru_v2_1.pt
```

完整曲線位於 `models/connectome_v2_1.csv` 與 `models/gru_v2_1.csv`。
使用原始小地圖、固定起點；連接組模型可訓練參數 17,925，GRU 為 27,205，容量未匹配。

## 獨立評估結果

兩次測試均為 seed=10007，使用確定性策略、每回合上限 160 步：

| 測試 | GRU | 連接組 |
|---|---:|---:|
| 相同小地圖、固定起點，5 回合 | 0/5 成功 | 0/5 成功 |
| easy 換圖、隨機起始位置與角度，10 回合 | 0/10 成功 | 0/10 成功 |

兩者都未成功，不能因碰撞少或移動慢就解讀為較好的導航能力。
固定地圖／固定起點／確定性策略的五次結果相同，並非五份獨立泛化證據。
逐回合資料見 `outputs/v2_1_fixed_evaluation.json`、`outputs/v2_1_paired_evaluation.json`。

另實際透過 `START_V2_1.cmd --headless` 跑過 80 步，得到 `outputs/v2_1_brain_trace.json`。
該回合由模型輸出控制，模擬 8 秒、移動約 0.329 格，未到達食物。
這驗證控制迴路確實接通，也顯示目前策略有慢走／旋轉問題，尚需改進訓練。

**v2.1 已完成模型接線與可啟動封裝，尚未完成可靠導航訓練。**
模型仍只是 FlyWire 實測結構子圖配合工程感覺／動作介面，非完整真實果蠅大腦。

資料限制與授權見 `data/DATA_LICENSE.md`。

## GitHub 跨平台發行補充

新增 Windows／Apple Silicon macOS 操作方式 `RUNNING.md`、Mac 啟動器 `START_V2_1.command`，
以及 `--check` 套件／模型檢查。平台條件另有 5 項單元測試；以模擬 macOS 名稱測試條件不等於在 Mac 上執行。
完整本機測試指令新增 `test_platform_support`，共 49 項。

Windows 與 macOS 的 [GitHub Actions](https://github.com/Sloth0528/fly-maze/actions) 執行無視窗測試，
包含舊版環境、連續幾何與感知、兩種模型、brain/body 接線、平台條件及啟動器。
本機結果與遠端 workflow 的實際執行狀態應分開判讀；macOS 桌面仍未進行人工驗收。
發行下載入口：[v2.1.0](https://github.com/Sloth0528/fly-maze/releases/tag/v2.1.0)。
