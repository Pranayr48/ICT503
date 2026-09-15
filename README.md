# Sentinel Grid

A model-backed cybersecurity activity monitor built from `CybersecurityThreatDetection.ipynb`.

## Run it

```powershell
python train_model.py
streamlit run app.py
```

Open the local URL printed by Streamlit. The sidebar can scan the supplied CSV feed in batches, change the alert confidence threshold, and reset the window.

## Real network capture on Windows

The sidebar includes **Live network capture**. Install [Npcap](https://npcap.com/#download) with WinPcap API-compatible mode enabled, install the Python requirements, restart Streamlit, then select live capture and click **Start live capture**. Administrator privileges may be required by the capture driver. Click **Refresh live packets** to score newly captured IP traffic with the exported model.

The model can classify packet metadata such as protocol, ports, byte length, IP octets, and internal traffic. Raw packet capture does not provide reliable URL or user-agent values, so those fields are blank for live events. This is a model-assisted detector, not a replacement for a production IDS such as Suricata or Zeek.

## Model artifact

`train_model.py` saves the fitted preprocessing and ExtraTrees pipeline to `artifacts/cybersecurity_model.joblib`. The artifact is intentionally generated locally and is ignored by source control because joblib files are binary and can be retrained from the dataset.

The app is a demonstration monitor over the supplied CSV feed. For production, replace the scan action with a message queue, API, or network telemetry collector and route alerts into an incident response workflow.
