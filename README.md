# Anomaly Detection Using DINOv2

以 DINOv2 視覺特徵和 SSPTT（token reconstruction + segmentation）進行 MVTec AD 異常偵測與定位。專案包含完整的 PyTorch/Lightning 訓練與評估流程，以及可上傳單張圖片的 Flask Web 介面。

## 功能

- 使用預訓練 DINOv2 `dinov2_vitl14_reg` 擷取 patch-level features。
- 透過 Perlin、CutPaste Normal 和 CutPaste Scar 產生合成異常，訓練 SSPTT。
- 在 MVTec AD test split 計算 image-level 與 pixel-level AUROC。
- 產生原圖、異常熱圖、疊合圖和預測 mask。
- 以 Flask Web UI 上傳圖片並取得異常分數。

## 專案結構

```text
.
├── SSPTT.py                 # 訓練與 MVTec 評估入口
├── model.py                 # SSPTT 模型
├── mvtec.py                 # MVTec AD Dataset
├── anomaly_types/           # Perlin 與 CutPaste 異常生成器
├── web_app/                 # Flask Web 服務
├── requirements.txt         # 訓練與評估依賴
├── requirements-web.txt     # Web 服務依賴
└── WEB_USAGE.md             # Web 介面快速說明
```

## 環境需求

- Python 3.10 或更新版本
- PyTorch 與 torchvision
- NVIDIA GPU（建議，訓練和第一次載入 DINOv2 會需要較多資源）
- 可連線至網路的環境（第一次執行會透過 `torch.hub` 下載 DINOv2 權重）

建立虛擬環境並安裝訓練依賴：

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若只執行 Web 服務，也可以安裝：

```powershell
pip install -r requirements-web.txt
```

## 準備 MVTec AD 資料集

下載並解壓 MVTec AD，使資料夾結構符合以下格式：

```text
dataset/
└── mvtec_anomaly_detection/
	└── carpet/
		├── ground_truth/
		├── test/
		│   ├── good/
		│   └── <defect_type>/
		└── train/
			└── good/
```

`SSPTT.py` 預設使用 `carpet` 類別。要訓練其他類別，修改檔案中的 `CONFIG["class_name"]`，並確保對應資料夾存在。

訓練時若要使用外部 DTD 圖片作為 Perlin anomaly source，可放置於：

```text
datasets/dtd/dtd/images/
```

若該資料夾不存在，Perlin generator 仍可使用自身產生的 pattern。

## 訓練與評估

在專案根目錄執行：

```powershell
python SSPTT.py
```

程式會依序：

1. 建立合成異常資料並訓練 SSPTT。
2. 每 50 個 epoch 將模型儲存至 `checkpoints/<class>_ep<epoch>.pth`。
3. 使用 test split 執行推論。
4. 將 AUROC 寫入 `results/<class>_v22/metrics.txt`，並將預測視覺化圖存至同一個資料夾。

主要設定位於 `SSPTT.py` 頂端的 `CONFIG`，包括 `epochs`、`batch_size`、`lr`、`num_workers` 和 `class_name`。

## Web 介面

Web 服務需要一個已訓練的 SSPTT state dict。程式預設尋找：

```text
checkpoints/wood_ep300.pth
```

可使用環境變數指定其他 checkpoint：

```powershell
$env:SSPTT_CHECKPOINT="C:\path\to\carpet_ep300.pth"
$env:SSPTT_THRESHOLD="0.5"
python web_app\app.py
```

啟動後開啟 <http://127.0.0.1:8000>。模型會在第一次呼叫 `/api/predict` 時載入，因此第一次推論可能需要較久。

健康檢查：

```text
GET /api/health
```

圖片推論：

```text
POST /api/predict
Content-Type: multipart/form-data
欄位：file
```

成功回應包含：

- `score`：輸入圖片的最大異常分數。
- `label`：`normal` 或 `abnormal`。
- `threshold`：目前使用的判定門檻。
- `images.original`：預處理後的原圖 Data URL。
- `images.heatmap`：異常熱圖 Data URL。
- `images.overlay`：原圖與熱圖疊合結果 Data URL。
- `images.mask`：依門檻二值化的預測 mask Data URL。

## 注意事項

- Web 服務的 checkpoint 必須和目前模型設定相容，包含 DINOv2 backbone、patch size 和 embedding dimension。
- 輸入圖片會被轉成 RGB，縮放並 center crop 為 `224 x 224`。
- 異常分數會先經過 Gaussian smoothing，再以最高 patch 分數作為 image score。
- `checkpoints/`、`dataset/` 和 `datasets/` 已列入 `.gitignore`，不會被 Git 追蹤。

## 參考文件

- [Web upload inference](WEB_USAGE.md)
