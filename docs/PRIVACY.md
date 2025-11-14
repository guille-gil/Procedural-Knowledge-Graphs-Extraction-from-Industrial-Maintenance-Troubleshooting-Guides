# Privacy & Security Guarantees

## 100% Local Processing

This flowchart extractor is designed to work **completely offline** with your proprietary data. Here's what you need to know:

### What Runs Locally

- **All PDF processing**: PDFs are read and processed entirely on your machine
- **All OCR processing**: EasyOCR runs locally using downloaded models
- **All shape detection**: OpenCV processes images locally
- **All graph construction**: NetworkX builds graphs in-memory
- **All output files**: Everything is saved to your local filesystem

### One-Time Model Download

**EasyOCR** requires model files for OCR. On first use, it will:
- Download ~500MB of model files (one-time only)
- Store them locally in `~/.EasyOCR/model/` (or similar)
- Never download again after initial setup

**This is the ONLY network activity**. After the initial download:
- No internet connection required
- No data sent anywhere
- 100% offline operation

### No External Services

The codebase has been verified to contain:
- No API calls to OpenAI, Anthropic, or other LLM services
- No cloud storage uploads
- No telemetry or analytics
- No external HTTP requests (except initial EasyOCR model download)

### Where Your Data Goes

- **Input PDFs**: Stay in `data/raw/TOCAPs/` (gitignored)
- **Output JSON**: Saved to `data/processed/` (gitignored)
- **Debug images**: Saved to `./debug/` (if debug mode enabled)
- **Nothing leaves your machine**

### How to Verify

You can verify no network activity by:

1. **Disconnect from internet** after initial EasyOCR setup
2. **Monitor network traffic** using tools like `netstat` or `lsof`
3. **Review the code**: All processing functions are in `src/flowchart_extractor.py` - no external calls

### For Maximum Security

If you want to be extra cautious:

1. **Download EasyOCR models manually** before processing sensitive data
2. **Run in an isolated environment** (Docker container, VM)
3. **Review the source code** before running on proprietary data
4. **Use firewall rules** to block network access if desired

## Summary

**Your proprietary PDFs are safe**. The extractor processes everything locally, and the only network activity is a one-time model download that happens before you process any sensitive documents.

