# 果蠅走迷宮 v2.0 使用指南

這版把果蠅的身體改成連續位置與轉向，並在相同環境中比較 **GRU** 與 **FlyWire 連接組子網路**。
先用手動模式確認斜線與碰撞，再訓練和評估。實作版本為 `2.0.0`。

## 1. 先手動操作

在專案根目錄開啟 PowerShell，使用原本的 Python 環境：

```powershell
.\.venv\Scripts\python.exe -m flymaze_v2 --version
.\.venv\Scripts\python.exe -m flymaze_v2 preview --difficulty easy --seed 7
```

若是剛解壓縮，先建立環境；已存在 `.venv` 時可省略：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

| 操作 | 效果 |
|---|---|
| ↑ 或 W | 按住前進 |
| ←／A、→／D | 按住左／右轉；可與前進同時按 |
| 只按轉向鍵 | 原地轉向 |
| Space | 暫停／繼續 |
| R | 重設目前地圖；隨機起點模式會另抽起點 |
| N | 產生下一張地圖，需使用 `--difficulty` |

先轉到約 45° 再按前進，即可走出斜線；同時前進與轉向會形成弧線。
紅色圓形是會參與碰撞的身體，紫色虛線是感知射線，綠色射線表示看見食物，藍線是移動軌跡。
全地圖只供人觀看，沒有直接送入模型。手動模式沒有按鍵時不耗用步數。

## 2. 訓練兩個模型

發行版已附 `data/flywire_cx_512.npz`，連接組模型可直接訓練，無須帳號或重新下載。
以下兩次訓練使用相同種子、環境、起點抽樣、回合數、獎勵與更新規則：

```powershell
.\.venv\Scripts\python.exe -m flymaze_v2 train --brain gru --difficulty easy --vary-maze --random-start --seed 7 --episodes 1000 --output outputs/v2/gru_seed7.pt
.\.venv\Scripts\python.exe -m flymaze_v2 train --brain connectome --difficulty easy --vary-maze --random-start --seed 7 --episodes 1000 --output outputs/v2/connectome_seed7.pt
```

每次產生 `.pt` 模型與同名 `.csv` 訓練紀錄。**既有檔案不會覆寫**；重跑請換輸出名稱。
1000 回合是實驗起點，不是保證收斂所需的回合數。建議先用 `--episodes 2 --max-steps 20` 確認流程。

訓練預設使用 Adam、學習率 0.0003、gamma=0.99、每 64 步更新一次 recurrent Actor-Critic，
並截斷記憶的反向傳播；不是 PPO。兩個模型都輸出相同的 tanh-Gaussian 連續策略和 critic。
探索正則項使用轉換前 Gaussian 的 entropy。成功後 bootstrap=0；時間截斷仍估計下一狀態價值。
模型檔自含設定、權重及連接組矩陣，載入後不依賴原本的 NPZ 路徑；尚未提供 optimizer 續訓。

## 3. 配對評估

```powershell
.\.venv\Scripts\python.exe -m flymaze_v2 compare --models outputs/v2/gru_seed7.pt outputs/v2/connectome_seed7.pt --difficulty easy --vary-maze --random-start --seed 10007 --episodes 50 --output outputs/v2/comparison_seed7.json --trace outputs/v2/last_gru_episode.svg
```

相同測試種子會讓兩個模型取得**相同地圖序列和相同起點、初始角度**；評估不更新權重。
測試種子 10007 與範例訓練種子 7 分開，但有限的迷宮集合仍可能出現相同圖形，不能稱為絕對去重。
預設使用策略平均值經 tanh 轉換的確定性動作；`--sample` 可改成抽樣策略。

JSON 記錄每回合起始姿態、地圖 seed、成功、步數、模擬秒數、路徑長和碰撞次數，並報告參數量。
成功回合的平均時間與路長另算，沒有成功時為 `null`，避免將快速失敗誤認為有效率。
另有成功率 Wilson 95% 區間；它假設回合獨立，不能替代多訓練種子的誤差估計。
固定地圖／固定起點／確定性策略的重複回合不提供獨立證據。
`--trace` 只匯出第一個模型最後一回合的軌跡。

兩個模型的身體、感知、獎勵或時間預算不同時，compare 會拒絕。
不同訓練地圖可用共同的 `--difficulty` 或 `--maze` 測試；共同的 `--max-steps` 可覆寫評估預算。
兩模型容量未匹配：預設 GRU 有 **27,205** 個可訓練參數，512 節點連接組模型有 **17,925** 個。
因此任何表現差異都不能直接歸因於生物結構；更完整研究仍需容量控制、隨機結構對照與消融實驗。

要觀察訓練穩定性，可把上述訓練與比較分別以 seed 8、9 再跑一次，並用不同輸出名稱。
不要只挑最好的一次。難度變更、身體大小與訓練時間都應記錄。

單模型評估與觀看：

```powershell
.\.venv\Scripts\python.exe -m flymaze_v2 evaluate --model outputs/v2/gru_seed7.pt --difficulty medium --vary-maze --random-start --seed 20007 --episodes 20
.\.venv\Scripts\python.exe -m flymaze_v2 preview --model outputs/v2/connectome_seed7.pt --difficulty easy --seed 10007
```

播放時沿用模型檔內的身體與感知設定。`--delay` 只改畫面間隔，不改模擬時間或移動速度。

## 4. 連續環境規格

| 項目 | 預設設定 |
|---|---|
| 座標 | 浮點數；一格寬為 1，格中心為 `(x+0.5,y+0.5)` |
| 朝向 | 弧度；東方為 0，螢幕向下為正 y，因此正角度向右轉 |
| 身體 | 半徑 0.16 格的圓盤 |
| 動作 | `[throttle, steering]`，範圍分別為 `[0,1]`、`[-1,1]` |
| 速度 | 最大 1.5 格／模擬秒，無倒退 |
| 轉向 | 最大 180°／模擬秒 |
| 時間 | 每步 0.1 模擬秒，預設最多 800 步 |
| 感知 | 13 條身體相對射線，涵蓋前方 270°、最遠 4 格 |
| 食物 | 半徑 0.25 格的目標；身體中心進入該區即成功 |

例如直走 0.1 秒和朝 45° 走 0.1 秒都前進 0.15 格，不會因同時改變 x、y 而多獲得速度。
所有轉向都消耗模擬時間。撞牆時計算該步碰撞，停止受阻的平移，仍允許轉向。
更新使用 unicycle 圓弧公式及細分段；每段對整個身體做掃掠碰撞，並加入圓弧弦高的保守餘裕。
最後到達目標的時間按段內相交比例估計，路長依接受的弧長累加；這是數值模擬，不是連續碰撞的解析解。
高速也不能穿牆；斜穿封閉牆角和擦過身體半徑內的柱角均會受阻。

觀察共有 **31 維**：13 個正規化牆距、13 個可見食物命中、朝向 cos/sin、上一動作 2 維、撞牆 1 維。
第一面牆會遮蔽後方食物；射線可能漏掉細小目標，模型不會收到全域食物方向、絕對位置或最短路徑。
cos/sin 是理想羅盤，射線也不是果蠅的真實複眼輸入。

獎勵為每步 -0.01、該步撞牆再 -0.1、到達食物加 10；`shaped` 模式在動作結束時
首次落入新道路格再 +0.02，`sparse` 關閉此探索獎勵。沒有穿牆距離提示或全域最短路徑獎勵。
此設定是教學用工程選擇，不是果蠅的生物學習規則。

沿用舊版 `# . S G` 厚牆格子地圖與三種難度產生器；邊界必須封閉。
這不是電腦鼠競賽的 18 cm 格子／1.2 cm 薄牆幾何，也沒有六足關節、慣性或輪胎動力學。
`--maze` 可載入 UTF-8 自訂地圖；與 `--difficulty` 擇一。

## 5. 連接組模型究竟重建了什麼

資料採 **FlyWire FAFB v783 成蟲雌性果蠅**，使用 EB、FB、PB、NO 區域內連線。
按該區域內輸入加輸出突觸計數排名取前 512 個神經元，保留其誘導子圖與所選區域的連線：
**16,212 條有向邊、294,654 個突觸計數**。不是完整腦，也未納入腹神經索。

```text
共享的 31 維感知
    ├─ GRU：可訓練 encoder → 64 維 GRU 記憶 → 相同策略與價值輸出
    └─ 連接組：可訓練 encoder → 512 節點固定結構 rate-RNN → 相同策略與價值輸出
                                      ↓
                         共享的速度／轉向與連續迷宮
```

令 `C[post,pre]` 為原始突觸計數，`A` 為各 row 正規化矩陣，連接組的更新為：

```text
h_next = 0.7*h + 0.3*tanh(encoder(observation) + 0.8*A*h)
```

矩陣連線方向與計數來自資料，矩陣固定、不補造邊。輸入 encoder、策略 head、critic、探索標準差可學習。
本版採**非負結構權重**，不把傳遞物質標籤直接視為已知興奮／抑制效果；也未模擬神經元形態、
放電、受體、荷爾蒙或生物可塑性。感覺可輸入每個節點、動作可讀出每個節點，並非經驗證的感覺神經／下行神經介面。
節點活動可為負值，是抽象隱藏狀態，不是生理放電率。

因此正確名稱是「**真實連接組結構約束的模型**」。選擇高強度節點會產生取樣偏差，
略去外部連線也會改變原始迴路；不能據此宣稱已複製果蠅導航能力。

## 6. 重建或替換子圖

```powershell
.\.venv\Scripts\python.exe -m flymaze_v2 prepare-connectome --nodes 512 --output data/cx_512_rebuilt.npz
.\.venv\Scripts\python.exe -m flymaze_v2 prepare-connectome --nodes 256 --regions EB FB PB NO --output data/cx_256.npz
.\.venv\Scripts\python.exe -m flymaze_v2 train --brain connectome --connectome data/cx_256.npz --episodes 100 --output outputs/v2/cx256.pt
```

第一次從公開資料檔下載約 50.3 MB 至 `data/raw/connections.csv.gz`，之後使用快取。
預設原始檔有固定 SHA256 驗證，若來源改變會拒絕靜默使用不同資料；`--source` 可明確匯入其他本機官方 CSV(.gz)。
輸出 NPZ 內附 root IDs、計數矩陣與來源資訊；JSON 紀錄原始檔和衍生檔 SHA256。
若同名 JSON 存在，載入時會檢查衍生檔雜湊。沒有測量資料時會明確報錯，不會以隨機圖替代。
根據當前程式，每次選出的節點與計數可重現；封裝時間不同會讓 NPZ 檔案雜湊不同。
資料授權與引用見 [DATA_LICENSE.md](data/DATA_LICENSE.md)。

## 7. 驗證與檔案分工

```powershell
.\.venv\Scripts\python.exe -m unittest test_maze test_v2 -v
```

| 檔案 | 用途 |
|---|---|
| `flymaze_v2/environment.py` | 連續身體、感知、碰撞與獎勵 |
| `flymaze_v2/geometry.py` | 掃掠線段／牆壁／目標幾何 |
| `flymaze_v2/connectome.py` | 官方資料下載、子圖選取、來源與雜湊 |
| `flymaze_v2/models.py` | GRU、連接組模型、共同動作策略及 checkpoint |
| `flymaze_v2/experiment.py` | 共同訓練、評估、配對比較 |
| `flymaze_v2/viewer.py` | 手動與模型畫面、SVG 軌跡 |
| `flymaze_v2/__main__.py` | 所有 v2 指令入口 |
| `test_v2.py` | 新版行為與訓練／讀檔測試 |

本次實際驗證結果與限制另見 [V2_VALIDATION.md](V2_VALIDATION.md)。
舊版請繼續使用 `train.py`、`evaluate.py`、`render.py`、`preview.py`，舊模型不會自動轉成新版。
