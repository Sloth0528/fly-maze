# 果蠅大腦模型用的迷宮環境

本環境讓既有 GRU 模型練習「局部感知 → 記憶 → 行動 → 找到食物」。
採二維離散格子，適合先確認模型的感知、記憶與導航能力。
這裡的果蠅是教學用代理；格子感知、理想羅盤與獎勵是人為假設，沒有重建真實果蠅的視覺或神經連接組。

## 先操作看看

在專案資料夾的 PowerShell 執行：

```powershell
.\.venv\Scripts\python.exe preview.py --difficulty easy --seed 7
```

預覽只使用 Python 標準函式庫與 Tkinter，無須模型權重。
左側是給人看的完整迷宮，右側是模型可取得的 5×5 局部視野。
按 ↑ 前進、← 左轉、→ 右轉，或使用視窗按鈕；R 重設同一張地圖。
可切換難度、輸入 seed、產生下一張地圖，並將地圖匯出為 UTF-8 文字檔。

## 任務與難度

從 S 出發，碰到 G 食物成功；碰牆留在原地；用完動作額度則截斷。
一次轉彎也算一個動作。成功發生在最後一個允許動作時，仍算成功，不另算超時。

| 難度 | 地圖尺寸（寬×高） | 動作上限 | 路線設定 |
|---|---:|---:|---|
| easy | 9×9 | 160 | 樹狀通道與死巷，練習轉彎和回頭 |
| medium | 15×11 | 450 | 更長路徑，額外打通約 5% 的候選隔牆 |
| hard | 21×15 | 900 | 較大探索範圍，額外打通約 12% 的候選隔牆 |

這些是規模與拓樸的難度設定，尚未透過訓練結果證明嚴格的難度排序。
產生器先用隨機深度優先搜尋挖出連通通道，再打通少量隔牆形成環路。
食物放在距離起點最遠的房格之一，距離以通道步數計算。
完整邊界都是牆；同一難度和 seed 產生同一地圖。

`--vary-maze` 讓每回合換圖；`--random-start` 另行隨機抽取可達起點和朝向，且不會從食物格開始。
兩個選項可以分開使用。相同 seed 與相同 reset 呼叫順序可重現地圖和起點序列。

## 大腦收到什麼

新產生迷宮預設使用 `egocentric`，觀察長度維持 **83**：

| 區段 | 維度 | 編碼 |
|---|---:|---|
| 局部牆壁 | 25 | 5×5，依列展平 |
| 局部道路 | 25 | 可見的非牆格；包含食物格 |
| 局部食物 | 25 | 可見的 G 格 |
| 理想朝向 | 4 | 東、南、西、北 one-hot |
| 上一次動作 | 3 | 前進、左轉、右轉 one-hot；reset 後全 0 |
| 上一步是否撞牆 | 1 | 0 或 1 |

視野中央是果蠅自己，上方永遠是果蠅前方。
第一面牆可見，牆後不可見；也不允許沿對角線看穿封閉牆角。
未知格的三個通道都是 0，可見道路與可見牆壁則各有對應通道為 1。
這是格子射線遮擋模型，不是相機影像；局部通道也不是嗅覺濃度。
理想羅盤仍提供絕對朝向，讓入門實驗容易診斷。

觀察不包含完整地圖、絕對位置、全域食物方向、最短路徑或已探索格集合。
記住走過哪條路是大腦的工作。人類預覽畫面、環境內部狀態與評估指標都不直接輸入大腦。

動作維持原介面：`0` 前進一格、`1` 左轉 90°、`2` 右轉 90°。

## 獎勵

新迷宮預設 `shaped`：

| 事件 | 獎勵 |
|---|---:|
| 每個動作 | −0.01 |
| 撞牆 | 額外 −0.10 |
| 本回合第一次進入某格 | 額外 +0.02 |
| 抵達食物 | 該步總獎勵設為 +10 |

首次進入格的合計獎勵為 +0.01。原地轉彎與重訪不會領取探索獎勵。
新模式沒有「全域距離縮短」或「穿牆靠近食物」提示。
`--reward-mode sparse` 移除首次探索加分，保留動作成本、碰撞成本與食物獎勵，適合比較塑形獎勵的影響。
首次探索獎勵可能讓模型偏好探索而非直接取食，因此評估應同時看成功率與路徑效率。

## 接上模型

```python
from maze import make_maze

env = make_maze(difficulty="easy", seed=7, vary_maze=True)
observation = env.reset(random_start=True)  # 83 個 float
memory = brain.initial_memory()            # 每回合重設記憶

while True:
    action, _, _, _, memory, _ = brain.act(observation, memory)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

實際訓練仍由 `train.py` 收集 log-prob、value 與 reward 並更新權重；上面只示範互動介面。
這是既有教學 API：`reset` 回傳觀察，`step` 回傳五個值；沒有宣稱實作完整 Gymnasium Env。

訓練多張 easy 迷宮：

```powershell
.\.venv\Scripts\python.exe train.py --difficulty easy --vary-maze --random-start --episodes 1000 --output outputs/maze_easy_v2.pt
```

使用另一組 seed 評估多張地圖：

```powershell
.\.venv\Scripts\python.exe evaluate.py --model outputs/maze_easy_v2.pt --difficulty easy --vary-maze --random-start --seed 10007 --episodes 50
.\.venv\Scripts\python.exe render.py --model outputs/maze_easy_v2.pt --difficulty easy --seed 10007 --gui
```

不同 seed 只代表不同抽樣序列，小型地圖仍可能重複；嚴格的保留集實驗應記錄完整地圖並排除重複。
產生器的起點固定在左上房格，食物偏向遠端房格，因此未見地圖測試也仍屬這個生成分布。
評估替換地圖時預設沿用模型儲存的動作上限；跨難度請明確指定 `--max-steps`。

建議記錄成功率、成功回合的平均動作數、碰撞次數，以及路徑效率：
成功時為「該回合起點與朝向的最短動作數 ÷ 實際動作數」，失敗時為 0。
最短動作數包含轉彎，由評估用全圖搜尋計算，不提供給大腦。
僅檢查總獎勵不足以判斷是否學會導航。

## 相容性與實驗限制

舊模型缺少感知設定時，繼續使用原來的 `world` 世界對齊視野與 `legacy` 獎勵。
新模型會在 checkpoint 儲存感知模式、獎勵模式、當前地圖、難度與換圖設定。
預設評估與觀看載入 checkpoint 儲存的那張地圖；要抽取測試地圖，須明確加 `--difficulty` 或 `--vary-maze`。
同為 83 維不代表新舊感知語意相同，切換感知模式後應重新訓練。
只給 `--maze` 的訓練會沿用舊模式；匯出新環境地圖後，請額外指定
`--observation-mode egocentric --reward-mode shaped` 以使用本設計。

本次完成環境與流程驗證，沒有完成策略收斂訓練；短訓練的成功率不能作為可用模型的證明。
既有 Actor–Critic 沒有內建跨難度續訓；可先分別測試各難度，再另外設計課程式訓練。

驗證指令：

```powershell
.\.venv\Scripts\python.exe -m unittest test_maze -v
```

12 項測試包含 3 種難度 × 30 個 seed 的可解性及動作額度、種子重現、視野旋轉、遮擋、
探索獎勵不可重複領取、碰撞、成功／超時，以及新舊環境設定載入。
