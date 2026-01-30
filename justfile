set shell := ["bash", "-cu"]

input := "examples/basic.zigsm"
ziggysm := "zig-out/bin/ziggsm"

build:
    set -euo pipefail
    zig build install

help: build
    set -euo pipefail
    tmp="$(mktemp)"
    {{ziggysm}} --help > "$tmp"
    rg -q "^Usage: " "$tmp"
    rm -f "$tmp"

missing: build
    set -euo pipefail
    tmp_out="$(mktemp)"
    tmp_err="$(mktemp)"
    if {{ziggysm}} >"$tmp_out" 2>"$tmp_err"; then
        echo "expected failure for missing input" >&2
        exit 1
    fi
    rg -q "missing input" "$tmp_err"
    rg -q "^Usage: " "$tmp_err"
    rm -f "$tmp_out" "$tmp_err"

stdout: build
    set -euo pipefail
    tmp_out="$(mktemp)"
    tmp_err="$(mktemp)"
    {{ziggysm}} "{{input}}" >"$tmp_out" 2>"$tmp_err"
    test -s "$tmp_out"
    ! rg -q "AST:|IR:" "$tmp_out"
    rm -f "$tmp_out" "$tmp_err"

dump: build
    set -euo pipefail
    tmp_out="$(mktemp)"
    tmp_err="$(mktemp)"
    {{ziggysm}} --dump-ast --dump-ir "{{input}}" >"$tmp_out" 2>"$tmp_err"
    test -s "$tmp_out"
    ! rg -q "AST:|IR:" "$tmp_out"
    rg -q "AST:" "$tmp_err"
    rg -q "IR:" "$tmp_err"
    rm -f "$tmp_out" "$tmp_err"

output: build
    set -euo pipefail
    tmp_out="$(mktemp)"
    tmp_err="$(mktemp)"
    tmp_file="$(mktemp --suffix=.zig)"
    {{ziggysm}} -o "$tmp_file" "{{input}}" >"$tmp_out" 2>"$tmp_err"
    test -s "$tmp_file"
    test ! -s "$tmp_out"
    test ! -s "$tmp_err"
    rm -f "$tmp_out" "$tmp_err" "$tmp_file"

test: help missing stdout dump output
    set -euo pipefail
    zig build install test
