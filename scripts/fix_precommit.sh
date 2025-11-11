#!/usr/bin/env bash
# ==============================================================================
# KEPLER PRE-COMMIT FIX SCRIPT
# ==============================================================================
# Purpose: Fix all pre-commit issues automatically
# Usage: bash scripts/fix_precommit.sh
# Date: 2025-11-11 22:24:03 UTC
# Author: sulegogh
# ==============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

echo "🔧 KEPLER PRE-COMMIT FIX"
echo "========================"
echo ""

# Fix 1: Executables
echo "📝 Fix 1/3: Setting executable permissions..."
chmod +x scripts/analyze_model.py
chmod +x scripts/create_baseline_reference.py
git update-index --chmod=+x scripts/analyze_model.py 2>/dev/null || true
git update-index --chmod=+x scripts/create_baseline_reference.py 2>/dev/null || true
echo "   ✅ Executables fixed"
echo ""

# Fix 2: Logger bug
echo "📝 Fix 2/3: Fixing undefined logger bug..."
if [[ -f "scripts/analyze_model_v2.py" ]]; then
    # macOS compatible sed
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' '473s/logger\.warning/print/' "scripts/analyze_model_v2.py"
    else
        sed -i '473s/logger\.warning/print/' "scripts/analyze_model_v2.py"
    fi
    echo "   ✅ Bug fixed (analyze_model_v2.py:473)"
else
    echo "   ⚠️  File not found (skipping)"
fi
echo ""

# Fix 3: Test hooks
echo "📝 Fix 3/3: Testing pre-commit hooks..."
pre-commit clean
pre-commit install
echo ""

if pre-commit run --all-files; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ SUCCESS! All hooks passed!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Next steps:"
    echo "  git add -A"
    echo "  git commit -m 'config: Autonomous pre-commit v4.1.0'"
    echo "  git push origin main"
    echo ""
else
    echo ""
    echo "⚠️  Some hooks failed (check output above)"
    echo "   This might be expected if config just updated."
    echo ""
    exit 1
fi
