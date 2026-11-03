# FlyWire 資料來源與授權

`flywire_cx_512.npz` 和同名 JSON 為 FlyWire FAFB v783 成蟲雌性果蠅公開資料的衍生子圖。
資料權利屬於 FlyWire Consortium 及相關資料貢獻者。依 [FlyWire 公開資料指引](https://flywire.ai/guidelines)，
此資料按 **CC BY-NC 4.0** 使用，須署名、標示修改，並限非商業使用。
這項資料授權不代表本專案其他程式碼另行取得相同授權。

原始來源：[FlyWire Codex](https://codex.flywire.ai/)、[公開連線 CSV.gz](https://storage.googleapis.com/flywire-data/codex/data/fafb/783/connections.csv.gz)。
原始檔約 50.3 MB；發行壓縮檔只附約 55 KB 子圖，不附原始 CSV。

原始檔 SHA256：

```text
d49dd692e59e153aa3c83f5257bfc0eff51247b86d7bb183386c6d1622c70fc9
```

本專案的修改：只採 EB、FB、PB、NO 區域連線，按區域內輸入加輸出突觸計數挑選前 512 個神經元，
同分依 root ID 排序；只保留兩端均入選且位於所選區域的連線，並合併相同方向的突觸計數。
矩陣為 `counts[post, pre]`。模型再對每個 postsynaptic row 做總和正規化。
未補造連線，不使用神經傳遞物質類別推斷正負號。模型中的活動值並非生理放電率。

請引用並尊重原作者的[完整引用指引](https://flywire.ai/guidelines)：

- Dorkenwald et al. (2024), *Neuronal wiring diagram of an adult brain*, Nature. [DOI](https://doi.org/10.1038/s41586-024-07558-y)
- Schlegel et al. (2024), *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature. [DOI](https://doi.org/10.1038/s41586-024-07686-5)

資料版本、選取方式、原始檔與衍生檔雜湊另存於 `flywire_cx_512.json`。
