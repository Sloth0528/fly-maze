# v2.1 隨附模型

| 檔案 | 模型 | 訓練設定 |
|---|---|---|
| `connectome_v2_1.pt` | FlyWire 512 節點結構約束 rate-RNN | seed=21，120 回合，每回合 160 步 |
| `gru_v2_1.pt` | GRU 64 維記憶比較基準 | 相同訓練設定 |

同名 CSV 是本次實際訓練紀錄。兩者使用固定小地圖與固定起點，尚未建立可靠導航，
詳見根目錄 `V2_1_VALIDATION.md`。`run_v2_1.py` 預設載入連接組模型；`--baseline` 改用 GRU。

checkpoint 包含模型結構、環境、訓練設定、矩陣與權重，使用 PyTorch `weights_only=True` 載入。
連接組衍生矩陣的資料署名及非商業授權見 `../data/DATA_LICENSE.md`。
