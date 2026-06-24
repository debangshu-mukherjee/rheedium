#!/usr/bin/env bash
# populate_wiki.sh
# Populates the Rheedium GitLab wiki from the existing repo docs.
# If a page already exists, it updates it instead of failing.
#
# Requirements: bash, curl, git (to clone the repo locally first)
#
# Usage:
#   export GITLAB_TOKEN="<your_personal_access_token>"
#   bash populate_wiki.sh
#
# Run from the root of the cloned rheedium repository.

set -euo pipefail

GITLAB_URL="https://code.ornl.gov"
PROJECT_ID="17162"
API="${GITLAB_URL}/api/v4/projects/${PROJECT_ID}/wikis"
TOKEN="${GITLAB_TOKEN:?Set GITLAB_TOKEN env var before running}"

# Convert a wiki page title to its URL slug (spaces -> hyphens, slashes kept)
slugify() {
  echo "$1" | sed 's/ /-/g'
}

create_or_update_page() {
  local title="$1"
  local file="$2"

  if [[ ! -f "$file" ]]; then
    echo "SKIP (file not found): $file"
    return
  fi

  local slug
  slug=$(slugify "$title")

  echo "Upserting wiki page: '$title' from $file ..."

  # Try POST first
  http_code=$(curl --silent --output /dev/null --write-out "%{http_code}" \
    --request POST \
    --header "PRIVATE-TOKEN: ${TOKEN}" \
    --form "title=${title}" \
    --form "content=<${file}" \
    "${API}")

  if [[ "$http_code" == "201" ]]; then
    echo "  Created: $title"
  elif [[ "$http_code" == "422" ]]; then
    # Page already exists — update it with PUT
    http_code=$(curl --silent --output /dev/null --write-out "%{http_code}" \
      --request PUT \
      --header "PRIVATE-TOKEN: ${TOKEN}" \
      --form "title=${title}" \
      --form "content=<${file}" \
      "${API}/${slug}")

    if [[ "$http_code" == "200" ]]; then
      echo "  Updated: $title"
    else
      echo "  ERROR updating '$title' (HTTP $http_code)"
    fi
  else
    echo "  ERROR creating '$title' (HTTP $http_code)"
  fi
}

# ── Top-level pages ──────────────────────────────────────────────────────────

create_or_update_page "Home"         "README.md"
create_or_update_page "Contributing" "CONTRIBUTING.md"
create_or_update_page "Changelog"    "CHANGELOG.md"

# ── Guides (already Markdown) ────────────────────────────────────────────────

create_or_update_page "Guides/Ewald Sphere"           "docs/source/guides/ewald-sphere.md"
create_or_update_page "Guides/Ewald CTR Tutorial"     "docs/source/guides/ewald-ctr-tutorial.md"
create_or_update_page "Guides/Data Wrangling"         "docs/source/guides/data-wrangling.md"
create_or_update_page "Guides/Arbitrary Directions"   "docs/source/guides/arbitrary-directions.md"
create_or_update_page "Guides/Numerical Entry Points" "docs/source/guides/checked-numerical-entry-points.md"

# ── API Reference index page (hand-written summary) ──────────────────────────

read -r -d '' API_REF_CONTENT << 'EOF' || true
# API Reference

The full API reference is published on **Read the Docs**:
https://rheedium.readthedocs.io/en/latest/api/index.html

## Modules

| Module | Description |
|--------|-------------|
| `rh.types`  | PyTree data structures and physical constants |
| `rh.ucell`  | Unit-cell and reciprocal-space crystallography |
| `rh.inout`  | Parsing and I/O (CIF, XYZ, POSCAR, TIFF, HDF5) |
| `rh.simul`  | The RHEED forward simulator |
| `rh.procs`  | Differentiable surface models |
| `rh.recon`  | Inverse problems and reconstruction |
| `rh.plots`  | Visualization |
| `rh.tools`  | Shared numerical kernels |
| `rh.audit`  | Physics-invariant checks and benchmarking |
EOF

local_slug=$(slugify "API Reference")

echo "Upserting wiki page: 'API Reference' (inline content) ..."
http_code=$(curl --silent --output /dev/null --write-out "%{http_code}" \
  --request POST \
  --header "PRIVATE-TOKEN: ${TOKEN}" \
  --form "title=API Reference" \
  --form "content=${API_REF_CONTENT}" \
  "${API}")

if [[ "$http_code" == "201" ]]; then
  echo "  Created: API Reference"
elif [[ "$http_code" == "422" ]]; then
  http_code=$(curl --silent --output /dev/null --write-out "%{http_code}" \
    --request PUT \
    --header "PRIVATE-TOKEN: ${TOKEN}" \
    --form "title=API Reference" \
    --form "content=${API_REF_CONTENT}" \
    "${API}/${local_slug}")
  [[ "$http_code" == "200" ]] && echo "  Updated: API Reference" || echo "  ERROR updating API Reference (HTTP $http_code)"
else
  echo "  ERROR creating API Reference (HTTP $http_code)"
fi

echo ""
echo "All wiki pages processed."
echo "View at: ${GITLAB_URL}/lotf-pilot/rheedium/-/wikis"