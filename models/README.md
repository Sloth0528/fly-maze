# v2.1 隨附模型

| 檔案 | 模型 | 訓練設定 |
|---|---|---|
| `connectome_v2_1.pt` | FlyWire 512 節點結構約束 rate-RNN | seed=21，120 回合，每回合 160 步 |
| `gru_v2_1.pt` | GRU 64 維記憶比較基準 | 相同訓練設定 |
| `connectome_v2_1_retrained.pt` | v2.1.1 新預設，固定 FlyWire 512 節點 | 示範＋DAgger，300 次更新，依驗證選第 250 次 |

同名 CSV 是實際訓練紀錄。前兩者是原始試跑模型，結果見 `V2_1_VALIDATION.md`。
新預設在固定小迷宮的 200 個獨立起始條件中成功 200 次，結果見 `FIX_STATIONARY.md`。
`default_connectome.json` 指定目前預設與 SHA256；不會自動選取檔案時間最新的模型。
`--baseline` 仍使用原始 GRU 試跑模型，因此這不是同等資料／方法下的架構效能比較。

checkpoint 包含模型結構、環境、訓練設定、矩陣與權重，使用 PyTorch `weights_only=True` 載入。
連接組衍生矩陣的資料署名及非商業授權見 `../data/DATA_LICENSE.md`。
