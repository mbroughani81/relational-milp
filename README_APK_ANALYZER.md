# APK VPN Link Analyzer

A bash script for analyzing Android APK files to extract VPN configuration links, endpoints, and related information.

## Features

The script extracts:
- **VPN Protocol URLs**: vless://, vmess://, trojan://, ss://, ssr://, hysteria://, tuic://, wireguard://
- **Configuration Endpoints**: HTTP/HTTPS URLs that might be subscription or API endpoints
- **Domain Names**: Hardcoded domains and IP addresses
- **Interesting Strings**: Filtered strings related to proxy, VPN, tunnel, V2Ray, Xray, SingBox, Clash, etc.

## Prerequisites

1. **Java JDK** (for jadx)
2. **jadx** - Automatically downloaded and installed in `/tmp/bin/jadx`
3. **unzip** - For extracting APK resources
4. **grep** - For pattern matching

## Installation

The script is ready to use. Just ensure you have the dependencies:

```bash
# Install Java and unzip if not already installed
sudo apt-get update && sudo apt-get install -y default-jdk unzip

# The script will automatically use jadx from /tmp/bin/jadx
# If needed, download jadx manually:
wget https://github.com/skylot/jadx/releases/download/v1.5.0/jadx-1.5.0.zip
unzip jadx-1.5.0.zip
```

## Usage

```bash
./apk_analyzer.sh <path_to_apk_file>
```

### Example

```bash
./apk_analyzer.sh v2ray_app.apk
```

## Output

The script creates an analysis directory (`./apk_analysis/<app_name>/`) containing:

- `vpn_urls.txt` - Direct VPN protocol URLs found in the code
- `endpoints.txt` - Configuration endpoints and domain names
- `all_strings.txt` - All extracted strings from the APK
- `interesting_strings.txt` - Filtered strings related to VPN/proxy functionality
- `src/` - Decompiled Java source code (for manual review)
- `resources/` - Extracted APK resources including strings.xml

## How It Works

1. **Decompilation**: Uses jadx to convert the APK to readable Java source code
2. **Resource Extraction**: Unzips the APK to access resources like strings.xml
3. **Pattern Matching**: Searches for VPN protocol URLs and configuration patterns
4. **Base64 Decoding**: Attempts to decode base64 strings that might contain VPN configs
5. **String Analysis**: Extracts and filters all strings for manual review

## Important Notes

⚠️ **Dynamic Configurations**: Many modern VPN apps don't hardcode VPN links directly in the APK. Instead, they:
- Fetch configurations from remote servers via API calls
- Use QR codes or user-provided subscription links
- Generate configurations dynamically

🔍 **What to Look For**:
- API endpoints that return subscription data
- URLs with keywords like "config", "subscribe", "api", "node", "server"
- Base64-encoded strings that decode to VPN URLs
- Hardcoded domain names used for configuration servers

## Example Output

```
==================================
APK VPN Link Analyzer
==================================
Analyzing: v2ray_app.apk

[1/5] Decompiling APK to Java source...
✓ Source code extracted to: ./apk_analysis/v2ray_app/src
[2/5] Extracting resources...
✓ Resources extracted to: ./apk_analysis/v2ray_app/resources
[3/5] Searching for VPN protocol URLs...
⚠ No direct VPN protocol URLs found

[4/5] Searching for configuration endpoints and domains...
✓ Found 15 potential endpoints/domains
https://api.example-vpn.com/config
https://subscribe.example.net/node
...

[5/5] Extracting all strings for manual review...
✓ Found 42 interesting strings
Sample interesting strings:
v2ray_config
proxy_enabled
subscription_url
...

==================================
Analysis Complete
==================================
Output directory: ./apk_analysis/v2ray_app
```

## License

This script is provided as-is for educational and security research purposes.
