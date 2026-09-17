#!/usr/bin/env bash
# Hit every legacy-bridge endpoint through the user's real runserver and
# dump the response body.  Pair with the DEBUG=True traceback surfacing
# in apps.legacy.api.views._with_legacy to get a full stack trace in
# each 500 body — no need to scroll through runserver stdout.
#
# Usage:
#   export ACCESS_TOKEN="eyJ..."        # access JWT (NOT the refresh)
#   bash scripts/diagnose_legacy.sh     # defaults to http://localhost:8000
#   BASE=http://localhost:8000 bash scripts/diagnose_legacy.sh
set -u

BASE="${BASE:-http://localhost:8000}"
TOK="${ACCESS_TOKEN:?set ACCESS_TOKEN to your access JWT}"

paths=(
    "/api/v1/legacy/portfolio/"
    "/api/v1/legacy/positions/"
    "/api/v1/legacy/trades/"
    "/api/v1/legacy/straddles/"
    "/api/v1/legacy/audit/?limit=8"
    "/api/v1/legacy/risk/"
    "/api/v1/legacy/alerts/"
    "/api/v1/legacy/analytics/"
    "/api/v1/legacy/exposure/"
    "/api/v1/legacy/system/"
    "/api/v1/legacy/strategies/"
    "/api/v1/legacy/watchlist/"
)

fail=0
pass=0
for p in "${paths[@]}"; do
    body=$(curl -sS -m 15 \
        -H "Authorization: Bearer $TOK" \
        -H "Accept: application/json" \
        -w $'\n__STATUS__=%{http_code}' \
        "$BASE$p")
    code=$(printf '%s' "$body" | awk -F= '/__STATUS__=/ {print $2}' | tail -1)
    payload=$(printf '%s' "$body" | sed '$d')
    if [[ "$code" =~ ^2 ]]; then
        printf 'OK  %s  %s\n' "$code" "$p"
        pass=$((pass+1))
    else
        printf '!!  %s  %s\n' "$code" "$p"
        # pretty-print the error body for 4xx/5xx — detail + exc_type + tb
        if command -v jq >/dev/null 2>&1; then
            printf '%s' "$payload" | jq '{error, exc_type, detail, traceback}' 2>/dev/null \
                || printf '    %s\n' "$payload" | head -c 600
        else
            printf '    %s\n' "$payload" | head -c 600
            echo
        fi
        fail=$((fail+1))
    fi
done

echo
echo "================================================================"
echo "  Passed: $pass    Failed: $fail    Base: $BASE"
echo "================================================================"
exit $fail
