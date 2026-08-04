#!/bin/bash

# APK VPN Link Analyzer Script
# This script analyzes APK files to extract VPN configuration links and URLs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
JADX_PATH="/tmp/bin/jadx"
OUTPUT_DIR="./apk_analysis"

# Function to display usage
usage() {
    echo -e "${BLUE}APK VPN Link Analyzer${NC}"
    echo "Usage: $0 <apk_file>"
    echo ""
    echo "This script analyzes APK files to extract:"
    echo "  - VPN protocol URLs (vless://, vmess://, trojan://, ss://, etc.)"
    echo "  - Configuration endpoints and API URLs"
    echo "  - Domain names and IP addresses"
    echo "  - Hardcoded strings that might be VPN-related"
    echo ""
    echo "Arguments:"
    echo "  apk_file    Path to the APK file to analyze"
    exit 1
}

# Check if jadx is available
check_dependencies() {
    if [ ! -x "$JADX_PATH" ]; then
        echo -e "${RED}Error: jadx not found at $JADX_PATH${NC}"
        echo "Please ensure jadx is installed and accessible."
        exit 1
    fi
    
    if ! command -v unzip &> /dev/null; then
        echo -e "${RED}Error: unzip is required but not installed.${NC}"
        exit 1
    fi
    
    if ! command -v grep &> /dev/null; then
        echo -e "${RED}Error: grep is required but not installed.${NC}"
        exit 1
    fi
}

# Main analysis function
analyze_apk() {
    local apk_file="$1"
    
    if [ ! -f "$apk_file" ]; then
        echo -e "${RED}Error: APK file not found: $apk_file${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}==================================${NC}"
    echo -e "${BLUE}APK VPN Link Analyzer${NC}"
    echo -e "${BLUE}==================================${NC}"
    echo -e "${GREEN}Analyzing: ${NC}$apk_file"
    echo ""
    
    # Create output directory
    local app_name=$(basename "$apk_file" .apk)
    local work_dir="$OUTPUT_DIR/$app_name"
    mkdir -p "$work_dir"
    
    # Step 1: Decompile APK to Java source
    echo -e "${YELLOW}[1/5] Decompiling APK to Java source...${NC}"
    "$JADX_PATH" -d "$work_dir/src" "$apk_file" > /dev/null 2>&1 || true
    echo -e "${GREEN}✓ Source code extracted to: $work_dir/src${NC}"
    
    # Step 2: Extract resources (strings.xml)
    echo -e "${YELLOW}[2/5] Extracting resources...${NC}"
    unzip -q -o "$apk_file" -d "$work_dir/resources" > /dev/null 2>&1 || true
    echo -e "${GREEN}✓ Resources extracted to: $work_dir/resources${NC}"
    
    # Step 3: Search for VPN protocol URLs
    echo -e "${YELLOW}[3/5] Searching for VPN protocol URLs...${NC}"
    local vpn_urls_file="$work_dir/vpn_urls.txt"
    
    # Common VPN protocols patterns
    grep -rhoE "(vless|vmess|trojan|ss|ssr|hysteria|tuic|wireguard)://[^\"'<>\s\)]+" \
        "$work_dir/src" "$work_dir/resources" 2>/dev/null | sort -u > "$vpn_urls_file" || true
    
    # Also search for base64 encoded configs (common in V2Ray apps)
    grep -rhoE "[A-Za-z0-9+/]{20,}={0,2}" "$work_dir/src" 2>/dev/null | \
        while read -r potential_b64; do
            decoded=$(echo "$potential_b64" | base64 -d 2>/dev/null || true)
            if echo "$decoded" | grep -qE "^(vless|vmess|trojan|ss|ssr)://"; then
                echo "$decoded" >> "$vpn_urls_file"
            fi
        done || true
    
    sort -u "$vpn_urls_file" -o "$vpn_urls_file" 2>/dev/null || true
    
    if [ -s "$vpn_urls_file" ]; then
        echo -e "${GREEN}✓ Found $(wc -l < "$vpn_urls_file") potential VPN URLs${NC}"
        cat "$vpn_urls_file"
    else
        echo -e "${YELLOW}⚠ No direct VPN protocol URLs found${NC}"
    fi
    echo ""
    
    # Step 4: Search for configuration endpoints and domains
    echo -e "${YELLOW}[4/5] Searching for configuration endpoints and domains...${NC}"
    local endpoints_file="$work_dir/endpoints.txt"
    
    # Search for HTTP/HTTPS URLs that might be config endpoints
    grep -rhoE "https?://[a-zA-Z0-9./_-]+\.[a-zA-Z]{2,}[^\"'<>\s\)]*" \
        "$work_dir/src" "$work_dir/resources" 2>/dev/null | \
        grep -iE "(config|subscribe|api|node|server|proxy|vpn|link)" | \
        sort -u > "$endpoints_file" || true
    
    # Search for domain patterns
    grep -rhoE "[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}\.[a-zA-Z]{2,}" \
        "$work_dir/src" 2>/dev/null | sort -u >> "$endpoints_file" || true
    
    if [ -s "$endpoints_file" ]; then
        echo -e "${GREEN}✓ Found $(wc -l < "$endpoints_file") potential endpoints/domains${NC}"
        head -20 "$endpoints_file"
        if [ $(wc -l < "$endpoints_file") -gt 20 ]; then
            echo "... and $(($(wc -l < "$endpoints_file") - 20)) more. See: $endpoints_file"
        fi
    else
        echo -e "${YELLOW}⚠ No obvious configuration endpoints found${NC}"
    fi
    echo ""
    
    # Step 5: Extract all strings for manual review
    echo -e "${YELLOW}[5/5] Extracting all strings for manual review...${NC}"
    strings "$apk_file" > "$work_dir/all_strings.txt" 2>/dev/null || true
    
    # Filter for interesting patterns
    local interesting_file="$work_dir/interesting_strings.txt"
    grep -iE "(proxy|vpn|tunnel|shadowsocks|v2ray|xray|singbox|clash|subscription|endpoint)" \
        "$work_dir/all_strings.txt" 2>/dev/null | \
        head -100 > "$interesting_file" || true
    
    if [ -s "$interesting_file" ]; then
        echo -e "${GREEN}✓ Found $(wc -l < "$interesting_file") interesting strings${NC}"
        echo "Sample interesting strings:"
        head -10 "$interesting_file"
    fi
    echo ""
    
    # Summary
    echo -e "${BLUE}==================================${NC}"
    echo -e "${BLUE}Analysis Complete${NC}"
    echo -e "${BLUE}==================================${NC}"
    echo -e "${GREEN}Output directory: ${NC}$work_dir"
    echo ""
    echo "Files generated:"
    echo "  - vpn_urls.txt          : Direct VPN protocol URLs"
    echo "  - endpoints.txt         : Configuration endpoints and domains"
    echo "  - all_strings.txt       : All extracted strings"
    echo "  - interesting_strings.txt : Filtered interesting strings"
    echo "  - src/                  : Decompiled Java source code"
    echo "  - resources/            : Extracted resources"
    echo ""
    echo -e "${YELLOW}Note:${NC} Many VPN apps fetch configurations dynamically from remote servers."
    echo "Look for API endpoints in the source code that might return subscription links."
}

# Main execution
if [ $# -eq 0 ]; then
    usage
fi

check_dependencies
analyze_apk "$1"
