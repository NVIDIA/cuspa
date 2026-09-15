# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

upload="${1:?upload is required}"
project_id="${2:?project ID is required}"
project_name="${3:?project name is required}"
pic="${4:?project owner email is required}"
pattern="${5:?artifact pattern is required}"
expected_count="${6:?artifact count is required}"
manifest="${7:?manifest is required}"

: "${KITMAKER_API_TOKEN:?KITMAKER_API_TOKEN is required}"
[[ "$upload" == true || "$upload" == false ]]
[[ "$project_id" =~ ^[0-9]+$ ]]
[[ "$expected_count" =~ ^[0-9]+$ ]]
[[ -f "$manifest" ]]

portal="https://kitmaker-portal.nvidia.com/api/v0"
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
printf '::add-mask::%s\n' "$KITMAKER_API_TOKEN"

mapfile -t artifacts < <(
  jq -r --arg pattern "$pattern" '
    .artifacts[] |
    select(.name | test($pattern)) |
    [.name, .sourcePointer, .sha256] | @tsv
  ' "$manifest"
)
[[ "${#artifacts[@]}" -eq "$expected_count" ]]

fetch_index() {
  local url="$1"
  local output="$2"
  local status

  status="$(
    curl --silent --show-error --retry 3 --retry-all-errors \
      --connect-timeout 10 --max-time 60 \
      -H 'Accept: application/vnd.pypi.simple.v1+json' \
      -H 'Cache-Control: no-cache' \
      -o "$output" -w '%{http_code}' "$url"
  )"
  [[ "$status" == 200 || "$status" == 404 ]]
  printf '%s' "$status"
}

load_indexes() {
  pypi_status="$(
    fetch_index "https://pypi.org/simple/${project_name}/" "$tmp_dir/pypi"
  )"
  devzone_status="$(
    fetch_index "https://pypi.nvidia.com/${project_name}/" "$tmp_dir/devzone"
  )"
}

pypi_digest() {
  local name="$1"
  if [[ "$pypi_status" == 404 ]]; then
    return
  fi
  jq -er --arg name "$name" '
    [.files[] | select(.filename == $name) | .hashes.sha256] | unique |
    if length == 0 then ""
    elif length == 1 then .[0]
    else error("duplicate filename")
    end
  ' "$tmp_dir/pypi"
}

devzone_digest() {
  local name="$1"
  local digest=""
  local href
  local candidate

  if [[ "$devzone_status" == 404 ]]; then
    return
  fi
  while IFS= read -r href; do
    href="${href#href=\"}"
    href="${href%\"}"
    if [[ "$href" == "${name}#sha256="* ]]; then
      candidate="${href#*#sha256=}"
      [[ "$candidate" =~ ^[0-9a-f]{64}$ ]]
      if [[ -n "$digest" && "$digest" != "$candidate" ]]; then
        printf 'Multiple hashes found for %s on pypi.nvidia.com.\n' "$name" >&2
        return 1
      fi
      digest="$candidate"
    fi
  done < <(grep -o 'href="[^"]*"' "$tmp_dir/devzone" || true)
  printf '%s' "$digest"
}

select_missing() {
  local allow_partial="${1:-false}"
  missing=()
  load_indexes
  local artifact
  local name
  local url
  local expected
  local public
  local devzone

  for artifact in "${artifacts[@]}"; do
    IFS=$'\t' read -r name url expected <<< "$artifact"
    [[ "$name" == "$(basename "$url")" ]]
    [[ "$url" == https://*/artifactory/* ]]
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]]
    public="$(pypi_digest "$name")"
    devzone="$(devzone_digest "$name")"

    if [[ "$public" == "$expected" && "$devzone" == "$expected" ]]; then
      continue
    fi
    if [[ -n "$public" && "$public" != "$expected" ]] || \
      [[ -n "$devzone" && "$devzone" != "$expected" ]]; then
      printf 'Published hash mismatch for %s.\n' "$name" >&2
      return 1
    fi
    if [[ -n "$public" || -n "$devzone" ]]; then
      if [[ "$allow_partial" == true ]]; then
        missing+=("$url")
        continue
      fi
      printf 'Partial Kitmaker publication detected for %s.\n' "$name" >&2
      return 1
    fi
    missing+=("$url")
  done
}

select_missing false
if [[ "${#missing[@]}" -eq 0 ]]; then
  printf '%s is already published on both indexes.\n' "$project_name"
  exit 0
fi

payload='[]'
for url in "${missing[@]}"; do
  payload="$(
    jq -c \
      --arg pic "$pic" \
      --arg url "$url" \
      --argjson upload "$upload" '
        . + [{
          pic: $pic,
          job_type: "wheel-release-job",
          url: $url,
          upload: $upload,
          mirroring_strategy: "upload-both"
        }]
      ' <<< "$payload"
  )"
done
body="$(
  jq -cn --arg project_name "$project_name" --argjson payload "$payload" \
    '{project_name: $project_name, payload: $payload}'
)"

response="$tmp_dir/response"
curl_status=0
http_code="$(
  curl --silent --show-error --connect-timeout 10 --max-time 120 \
    -H "Authorization: Bearer ${KITMAKER_API_TOKEN}" \
    -H 'Content-Type: application/json' \
    -o "$response" -w '%{http_code}' \
    --data "$body" \
    "${portal}/projects/${project_id}/releases"
)" || curl_status=$?
if [[ "$curl_status" -ne 0 || "$http_code" != 202 ]]; then
  printf 'Kitmaker submission failed (curl %s, HTTP %s).\n' \
    "$curl_status" "$http_code" >&2
  cat "$response" >&2
  exit 1
fi
release_uuid="$(jq -er '.release_uuid | strings | select(length > 0)' "$response")"

completed=false
for _ in {1..180}; do
  curl_status=0
  http_code="$(
    curl --silent --show-error --connect-timeout 10 --max-time 60 \
      -H "Authorization: Bearer ${KITMAKER_API_TOKEN}" \
      -o "$response" -w '%{http_code}' \
      "${portal}/status/${release_uuid}"
  )" || curl_status=$?

  if [[ "$curl_status" -ne 0 || "$http_code" == 429 || "$http_code" =~ ^5[0-9]{2}$ ]]; then
    sleep 30
    continue
  fi
  if [[ "$http_code" != 200 ]]; then
    printf 'Kitmaker status failed (HTTP %s).\n' "$http_code" >&2
    cat "$response" >&2
    exit 1
  fi

  status="$(jq -er '.status | strings | select(length > 0)' "$response")"
  printf 'Kitmaker %s status: %s\n' "$project_name" "$status"
  case "$status" in
    completed)
      completed=true
      break
      ;;
    failed)
      cat "$response" >&2
      exit 1
      ;;
    *)
      sleep 30
      ;;
  esac
done
[[ "$completed" == true ]]

if [[ "$upload" == true ]]; then
  published=false
  for _ in {1..24}; do
    select_missing true
    if [[ "${#missing[@]}" -eq 0 ]]; then
      published=true
      break
    fi
    sleep 30
  done
  [[ "$published" == true ]]
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  printf "Kitmaker \`%s\` completed for \`%s\` (%s).\n" \
    "$release_uuid" "$project_name" \
    "$(if [[ "$upload" == true ]]; then printf published; else printf validated; fi)" \
    >> "$GITHUB_STEP_SUMMARY"
fi
