# v2.0.0 實際驗證紀錄

日期：2026-11-03（Asia/Taipei）。環境：Windows、Python 3.12、PyTorch 2.14.1+cpu、NumPy 2.5.2。

## 已完成的檢查

```powershell
.\.venv\Scripts\python.exe -m unittest test_maze test_v2 -v
```

**33 項通過**：舊版 12 項、新版 21 項。

- 45° 斜走與直走的速度、路長及模擬時間一致；任意角度原地轉向與圓弧座標符合運動公式。
- 高速掃掠不穿牆；圓形身體受到牆角與自身半徑限制。
- 前方射線會隨身體旋轉；牆後食物不洩漏至觀察；觀察固定為 31 維。
- 沿移動段穿過食物也能成功；最後允許步成功不會同時算超時；成功時按相交比例記錄時間。
- 無效動作不改變回合狀態；隨機地圖與起始姿態可重現；環境設定可回存。
- 使用全圖的測試專用 waypoint 控制器可零碰撞完成原始小迷宮，確認身體與通道相容；這不是學習結果。
- CSV 匯入方向與跨區域計數合併正確；衍生資料被修改時雜湊檢查會報錯。
- 隨附真實子圖的節點、連線、突觸總數符合預期；資料缺失時不使用隨機圖替代。
- 兩種模型的動作有界、梯度有限、訓練會更新參數；連接組結構固定且確實影響遞迴狀態。
- checkpoint 可獨立載入；評估不更新權重；比較使用配對地圖與初始姿態並拒絕身體設定不同的模型。
- Tkinter 手動與模型視窗可建立、繪圖、處理按鍵與播放到回合結束；此為程式化 GUI 檢查，沒有做長時間人工操作驗收。

## 真實 512 節點模型的端到端短跑

以下檔案已在本機實際產生，發行壓縮檔亦附上，全部標記為 **smoke**，供檢查流程：

```powershell
.\.venv\Scripts\python.exe -m flymaze_v2 train --brain gru --episodes 2 --max-steps 20 --output outputs/v2_gru_smoke.pt
.\.venv\Scripts\python.exe -m flymaze_v2 train --brain connectome --episodes 2 --max-steps 20 --output outputs/v2_connectome_smoke.pt
.\.venv\Scripts\python.exe -m flymaze_v2 compare --models outputs/v2_gru_smoke.pt outputs/v2_connectome_smoke.pt --difficulty easy --vary-maze --random-start --seed 10007 --episodes 5 --max-steps 60 --output outputs/v2_smoke_comparison.json --trace outputs/v2_smoke_trace.svg
```

既有輸出不覆寫；重現時請替換檔名。
兩者訓練都用原始小地圖、seed=7、固定起點，各 2 回合、每回合最多 20 步，訓練成功率均為 0%。
測試改為 easy 隨機地圖、隨機起始位置與角度，seed=10007，共 5 回合，每回合最多 60 步。

| 模型 | 可訓練參數 | 測試成功 | 平均步數（所有回合） | 平均模擬秒數（所有回合） | 平均碰撞步數 |
|---|---:|---:|---:|---:|---:|
| GRU（64 維記憶） | 27,205 | 1/5 | 50.2 | 約 5.001 | 34.4 |
| FlyWire 子圖（512 節點） | 17,925 | 1/5 | 50.2 | 約 5.001 | 32.2 |

完整逐回合結果見 [v2_smoke_comparison.json](outputs/v2_smoke_comparison.json)。
其中成功回合由隨機起點產生，不能當作已學會完整起點到終點路徑的證據。
兩個成功率估計的 Wilson 95% 區間均約為 3.6%–62.4%，且只有一個訓練種子。

**這些結果僅驗證程式可執行，不支持穩定導航、收斂、連接組較佳或與真實果蠅行為等效的結論。**
較長訓練、多訓練種子、容量控制與生物介面驗證尚待進一步實驗。

## 資料與發行範圍

已下載官方公開 `connections.csv.gz`，驗證並記錄 SHA256；只封裝 512 節點衍生子圖與來源紀錄。
預設資料計 **16,212 條有向連線、294,654 個突觸計數**。資料使用條件見 [DATA_LICENSE.md](data/DATA_LICENSE.md)。

v2.0.0 是本機版本檔與可攜式 ZIP 發行包。此工作沒有建立 Git 提交、標籤或遠端 GitHub Release。
