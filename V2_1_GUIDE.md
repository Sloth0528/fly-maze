# v2.1：由果蠅連接組模型自動控制

**目前修正版為 v2.1.1。** 預設從 `models/default_connectome.json` 載入通過小迷宮測試的新模型。
原地轉圈原因、200 個獨立起始條件結果，以及引導訓練方法見 [FIX_STATIONARY.md](FIX_STATIONARY.md)。

v2.1 開啟後預設接上 **FlyWire FAFB v783 的 512 神經元連接組子網路模型**。
不需要按方向鍵；每一步都執行「局部感知 → 連接組遞迴狀態 → 速度／轉向 → 身體移動 → 新感知」。
這是實測連接組結構約束的 rate-RNN，感覺與動作介面由工程訓練得到，並非完整生物大腦模擬。

## 直接啟動

Windows／macOS 第一次安裝方式見 [RUNNING.md](RUNNING.md)。
以下 PowerShell 範例用於 Windows；macOS 的同等指令將 Python 執行路徑改成 `.venv/bin/python`，並移除 `&`。
Windows 安裝完成後，**雙擊根目錄的 `START_V2_1.cmd`** 即可；Mac 可執行 `bash START_V2_1.command`。
Windows 也可開啟 PowerShell：

```powershell
cd "C:\FlyMaze"
& ".\.venv\Scripts\python.exe" run_v2_1.py
```

若使用下載的壓縮檔，先完整解壓縮，包含 `models/`、`data/`、`flymaze_v2/`。
不要直接在 ZIP 內執行。首次使用的電腦可在解壓後的資料夾建立環境：

```powershell
py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
& ".\.venv\Scripts\python.exe" run_v2_1.py
```

啟動程式以自身路徑尋找內附模型，因此不必從特定工作目錄啟動。
自訂 `--maze`、`--model`、`--report` 的相對路徑則以目前終端機所在目錄解讀。

## 畫面與操作

| 區域／控制 | 說明 |
|---|---|
| 左側迷宮 | 模型控制的圓形身體、方向、軌跡與遮擋射線 |
| 右側彩色格 | 512 個模型隱藏狀態；紅／藍區分正負，深淺表示幅度 |
| 滑鼠移到彩色格上 | 顯示對應的 FlyWire root ID 與狀態值 |
| Motor source | `brain.act` 表示動作由模型輸出直接產生 |
| throttle／steering | 原始動作；同時顯示格／秒與度／秒 |
| Top nodes | 目前絕對狀態值最大的五個節點；不是生物重要性排序 |
| Space／暫停按鈕 | 暫停或繼續 |
| R／重新開始 | 重設目前地圖，同時清除模型記憶與本回合紀錄 |
| N／下一張地圖 | 搭配 `--difficulty` 時使用 |
| 匯出模型紀錄 | 將目前回合寫入 `outputs/v2_1_sessions/`；不覆寫舊檔 |
| 載入模型 | 選取 `.pt`，切換至該模型的地圖並清除舊記憶 |
| 視窗底部檔名 | 確認實際載入的模型；不會自動選取最新訓練檔 |

彩色格是**抽象神經網路隱藏狀態，非生理放電或實測腦活動**。
自動模式下方向鍵不參與動作決策。播放不訓練，也沒有隱藏的路徑規劃器或避障控制器。
遇到無效神經狀態會暫停並顯示錯誤；模型遺失或類型錯誤時會報錯。

**v2.1.1 的預設引導模型已在原始小迷宮成功完成獨立起點測試。**
保留的舊 120 回合模型和原始純強化學習方法仍可能慢走、轉圈；新模型也未證明陌生迷宮能力。
舊版驗證見 [V2_1_VALIDATION.md](V2_1_VALIDATION.md)，新版結果見 [FIX_STATIONARY.md](FIX_STATIONARY.md)。

## 切換地圖、模型與手動控制

```powershell
# 連接組模型在隨機 easy 地圖中自動移動
& ".\.venv\Scripts\python.exe" run_v2_1.py --difficulty easy --seed 7

# 使用隨機起點與抽樣動作；抽樣不是可靠導航的保證
& ".\.venv\Scripts\python.exe" run_v2_1.py --difficulty easy --random-start --sample

# 切換為同條件訓練的 GRU 比較基準
& ".\.venv\Scripts\python.exe" run_v2_1.py --baseline

# 明確切回手動模式
& ".\.venv\Scripts\python.exe" run_v2_1.py --manual --difficulty easy

# 載入自己訓練的連接組 checkpoint
& ".\.venv\Scripts\python.exe" run_v2_1.py --model outputs/my_connectome.pt
```

模型類型有檢查：預設只能載入 `connectome`；GRU 檔案需要搭配 `--baseline`。
`--delay 0.05` 只調整視窗播放間隔，不改變模擬時間或模型的物理速度。
預設使用 checkpoint 的小型訓練地圖及 160 步上限；可用 `--max-steps` 明確覆寫。

## 不開視窗也能確認接線

```powershell
& ".\.venv\Scripts\python.exe" run_v2_1.py --headless --max-steps 80 --report outputs/my_brain_trace.json
```

預設會跑一個回合；`--episodes 5` 可跑五個。未指定 `--report` 時，自動寫入帶時間戳的輸出檔。
紀錄包含 checkpoint SHA256、資料來源與子圖規模、每一步 31 維感知、實際動作、移動前後姿態、
狀態 RMS、前五大節點、碰撞及獎勵。圖形模式與無視窗模式共用 `ControlSession`。
紀錄中的 `brain.act only` 是程式的控制來源標示；接線另以逐步動作比對和連線消融測試驗證。

## 模型來源與後續訓練

內附 `models/connectome_v2_1.pt` 與 `models/gru_v2_1.pt`，均以 seed=21、相同小地圖、固定起點，
各訓練 120 回合、每回合最多 160 步。兩者使用相同 recurrent Actor-Critic 與 TBPTT 設定。
連接組模型固定保留 16,212 條實測有向連線，輸入編碼器、動作／價值讀出則透過強化學習更新。
checkpoint 已內含矩陣與 root IDs，啟動不需網路下載。

這兩份是保留的原始試跑模型；目前預設為 `models/connectome_v2_1_retrained.pt`。
新預設使用示範與 DAgger 訓練，視窗顯示 `Guided updates`，並非額外的純強化學習回合數。
建議新手先依 [引導訓練步驟](FIX_STATIONARY.md#重新訓練) 練習小迷宮。

若要再做新實驗，請使用新的輸出檔名，例如：

```powershell
& ".\.venv\Scripts\python.exe" -m flymaze_v2 train --brain connectome --difficulty easy --vary-maze --random-start --episodes 1000 --seed 22 --output outputs/my_connectome.pt
& ".\.venv\Scripts\python.exe" run_v2_1.py --model outputs/my_connectome.pt --difficulty easy
```

這是重新訓練，不是接續 optimizer 狀態；回合數也不是成功保證。
資料選取、生物限制及授權分別見 [v2 設計指南](V2_GUIDE.md) 與 [資料授權](data/DATA_LICENSE.md)。
v2.0 指令 `python -m flymaze_v2 preview` 仍預設手動；需要預設自動控制時請使用本版啟動入口。
