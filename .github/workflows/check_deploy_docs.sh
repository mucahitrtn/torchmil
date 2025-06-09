#!/bin/bash
set -e

echo "Checking for warnings in MkDocs build..."

output=$(mkdocs build 2>&1)

echo "$output"

if echo "$output" | grep -q "WARNING"; then
  echo "❌ Documentation build contains warnings! Aborting deploy."
  exit 1
fi

echo "✅ No warnings found. Proceeding with deployment..."
mkdocs gh-deploy --force --clean

