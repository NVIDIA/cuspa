#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Publish one project's files from a published GitHub release:
#
#   kitmaker_release.sh UPLOAD PROJECT_ID PROJECT_NAME PIC PATTERN COUNT MANIFEST
#
# The release service fetches each file from its release-asset URL and
# publishes it to pypi.org and pypi.nvidia.com. A publication cannot be
# recalled, so nothing is uploaded until every check has passed: the service's
# project carries the expected name and owner, each file belongs to it and is
# attached to the release with the manifest's SHA-256, and a dry run completes.
# Right before the upload the release must still hold the same assets, by ID as
# well as digest. Afterwards both indexes must serve each file with that
# SHA-256. UPLOAD=false stops after the dry run.
#
# Requests go through the tunnel at TUNNEL_URL unless KITMAKER_ROUTE=direct,
# which sends them to KITMAKER_PORTAL_URL. Release assets are read with gh.

set -euo pipefail

upload="${1:?upload is required}"
project_id="${2:?project ID is required}"
project_name="${3:?project name is required}"
pic="${4:?project owner email is required}"
pattern="${5:?artifact pattern is required}"
expected_count="${6:?artifact count is required}"
manifest="${7:?manifest is required}"

: "${KITMAKER_API_TOKEN:?KITMAKER_API_TOKEN is required}"
: "${RELEASE_SHA:?release SHA is required}"
: "${RELEASE_TAG:?release tag is required}"
route="${KITMAKER_ROUTE:-tunnel}"
[[ "$upload" == true || "$upload" == false ]]
[[ "$project_id" =~ ^[0-9]+$ ]]
[[ "$expected_count" =~ ^[0-9]+$ ]]
[[ -f "$manifest" ]]

case "$route" in
  tunnel)
    : "${ACTIONS_ID_TOKEN_REQUEST_TOKEN:?GitHub OIDC permission is required}"
    : "${ACTIONS_ID_TOKEN_REQUEST_URL:?GitHub OIDC permission is required}"
    TUNNEL_URL="${TUNNEL_URL:-http://127.0.0.1:8888}"
    portal="${TUNNEL_URL%/}/kitmaker-portal/api/v0"
    ;;
  direct)
    portal="${KITMAKER_PORTAL_URL:?KITMAKER_PORTAL_URL is required for the direct route}"
    portal="${portal%/}"
    ;;
  *)
    printf 'KITMAKER_ROUTE must be tunnel or direct, not %s.\n' "$route" >&2
    exit 2
    ;;
esac

tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
if [[ "${GITHUB_ACTIONS:-}" == true ]]; then
  printf '::add-mask::%s\n' "$KITMAKER_API_TOKEN"
fi

# curl reads credentials from private config files so they stay out of argv.
portal_auth="${tmp_dir}/portal.curlrc"
tunnel_auth="${tmp_dir}/tunnel.curlrc"
(umask 077 && printf 'header = "Authorization: Bearer %s"\n' "$KITMAKER_API_TOKEN" > "$portal_auth")
response="${tmp_dir}/response"

# Service responses can carry internal details, so GitHub Actions logs get only
# the HTTP status. Local runs print the whole response.
show_response() {
  if [[ "${GITHUB_ACTIONS:-}" != true ]]; then
    cat "$response" >&2
  fi
}

tunnel_token_time=0
portal_http_code=""

refresh_tunnel_token() {
  local now
  local oidc_url
  local token_response
  local token

  now="$(date +%s)"
  if ((tunnel_token_time > 0 && now - tunnel_token_time < 240)); then
    return
  fi
  oidc_url="$ACTIONS_ID_TOKEN_REQUEST_URL"
  if [[ "$oidc_url" == *\?* ]]; then
    oidc_url+='&audience=charon.nvidia.com'
  else
    oidc_url+='?audience=charon.nvidia.com'
  fi
  token_response="$(
    curl --fail-with-body --silent --show-error \
      --connect-timeout 10 --max-time 60 \
      -H "Authorization: Bearer ${ACTIONS_ID_TOKEN_REQUEST_TOKEN}" \
      "$oidc_url"
  )" || return
  token="$(
    jq -er '.value | strings | select(length > 0)' <<< "$token_response"
  )" || return
  printf '::add-mask::%s\n' "$token"
  (umask 077 && printf 'header = "X-Charon-GHA-Token: %s"\n' "$token" > "$tunnel_auth")
  tunnel_token_time="$now"
}

# portal_request METHOD PATH [BODY]: the body goes to $response and the HTTP
# status to $portal_http_code. Never call it inside a command substitution: a
# tunnel token refresh prints a mask command for the runner.
portal_request() {
  local method="$1"
  local path="$2"
  local body="${3:-}"
  local -a args=(
    --config "$portal_auth" --silent --show-error
    --connect-timeout 10 --max-time 120
    -X "$method" -o "$response" -w '%{http_code}'
  )

  portal_http_code=""
  if [[ "$route" == tunnel ]]; then
    refresh_tunnel_token || return
    args+=(--config "$tunnel_auth")
  fi
  if [[ -n "$body" ]]; then
    args+=(-H 'Content-Type: application/json' --data "$body")
  fi
  portal_http_code="$(curl "${args[@]}" "${portal}/${path}")"
}

mapfile -t artifacts < <(
  jq -r --arg pattern "$pattern" '
    .artifacts[] |
    select(.name | test($pattern)) |
    [.name, .sourcePointer, .sha256] | @tsv
  ' "$manifest"
)
[[ "${#artifacts[@]}" -eq "$expected_count" ]]
repository="$(jq -er '.repository | strings | select(length > 0)' "$manifest")"
tag="$(jq -er '.tag | strings | select(length > 0)' "$manifest")"
commit="$(jq -er '.commit | strings | select(length > 0)' "$manifest")"
[[ "$repository" == NVIDIA/cuspa ]]
[[ "$tag" == "$RELEASE_TAG" ]]
[[ "$commit" == "$RELEASE_SHA" ]]
version="${tag#v}"
source_base="https://github.com/${repository}/releases/download/${tag}"

canonical() {
  sed -E 's/[-_.]+/-/g' <<< "${1,,}"
}

# Every file must be this project's distribution at the release version.
for artifact in "${artifacts[@]}"; do
  IFS=$'\t' read -r name url expected <<< "$artifact"
  [[ "$name" == "$(basename "$name")" ]]
  [[ "$url" == "${source_base}/${name}" ]]
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]]
  case "$name" in
    *.whl)
      distribution="${name%%-*}"
      file_version="${name#*-}"
      file_version="${file_version%%-*}"
      ;;
    *.tar.gz)
      distribution="${name%-"${version}".tar.gz}"
      file_version="$version"
      [[ "$distribution" != "$name" ]]
      ;;
    *)
      false
      ;;
  esac
  if [[ "$(canonical "$distribution")" != "$(canonical "$project_name")" ]] ||
    [[ "$file_version" != "$version" ]]; then
    printf '%s is not a %s %s distribution.\n' "$name" "$project_name" "$version" >&2
    exit 1
  fi
done

# The service's project must be the expected one, owned by the release PIC.
curl_status=0
portal_request GET "projects/${project_id}" || curl_status=$?
if [[ "$curl_status" -ne 0 || "$portal_http_code" != 200 ]]; then
  printf 'Project lookup failed (curl %s, HTTP %s).\n' \
    "$curl_status" "$portal_http_code" >&2
  show_response
  exit 1
fi
if ! jq -e --arg name "$project_name" --arg pic "$pic" '
  .name == $name and ((.owner_email // "") | ascii_downcase) == ($pic | ascii_downcase)
' "$response" > /dev/null; then
  printf 'Project %s is not %s owned by the release PIC.\n' "$project_id" "$project_name" >&2
  show_response
  exit 1
fi

# Pin each file's release asset by ID and digest. Sets $pinned.
pin_assets() {
  local release="${tmp_dir}/release.json"
  local artifact
  local name
  local url
  local expected
  local asset
  local id
  local asset_url
  local digest

  gh api "repos/${repository}/releases/tags/${tag}" > "$release"
  jq -e --arg tag "$tag" '.draft == false and .tag_name == $tag' "$release" > /dev/null
  pinned=""
  for artifact in "${artifacts[@]}"; do
    IFS=$'\t' read -r name url expected <<< "$artifact"
    asset="$(
      jq -r --arg name "$name" '
        [.assets[] | select(.name == $name and .state == "uploaded")] |
        if length == 1 then .[0] | [.id, .browser_download_url, (.digest // "")] | @tsv
        else empty end
      ' "$release"
    )"
    if [[ -z "$asset" ]]; then
      printf '%s is not attached to release %s.\n' "$name" "$tag" >&2
      exit 1
    fi
    IFS=$'\t' read -r id asset_url digest <<< "$asset"
    if [[ "$asset_url" != "$url" || "$digest" != "sha256:${expected}" ]]; then
      printf 'The attached %s (%s) is not the manifest file.\n' \
        "$name" "${digest:-no digest}" >&2
      exit 1
    fi
    pinned+="${name} ${id} ${digest}"$'\n'
  done
}

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
      printf 'Partial publication detected for %s.\n' "$name" >&2
      return 1
    fi
    missing+=("$url")
  done
}

# submit UPLOAD: submit the missing files; sets $release_uuid.
submit() {
  local payload='[]'
  local body
  local url

  for url in "${missing[@]}"; do
    payload="$(
      jq -c \
        --arg pic "$pic" \
        --arg url "$url" \
        --argjson upload "$1" '
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

  curl_status=0
  portal_request POST "projects/${project_id}/releases" "$body" || curl_status=$?
  if [[ "$curl_status" -ne 0 || "$portal_http_code" != 202 ]]; then
    printf 'Submission failed (curl %s, HTTP %s).\n' \
      "$curl_status" "$portal_http_code" >&2
    show_response
    exit 1
  fi
  release_uuid="$(jq -er '.release_uuid | strings | select(length > 0)' "$response")"
}

# wait_for_release WHAT: poll $release_uuid until the service finishes it.
wait_for_release() {
  local status

  for _ in {1..180}; do
    curl_status=0
    portal_request GET "status/${release_uuid}" || curl_status=$?
    if [[ "$curl_status" -ne 0 || "$portal_http_code" == 429 ||
      "$portal_http_code" =~ ^5[0-9]{2}$ ]]; then
      sleep 30
      continue
    fi
    if [[ "$portal_http_code" != 200 ]]; then
      printf 'Status request failed (HTTP %s).\n' "$portal_http_code" >&2
      show_response
      exit 1
    fi

    status="$(jq -er '.status | strings | select(length > 0)' "$response")"
    printf '%s %s: %s\n' "$project_name" "$1" "$status"
    case "${status,,}" in
      completed | success | succeeded)
        return 0
        ;;
      failed | error | timeout | cancelled | canceled)
        show_response
        exit 1
        ;;
      *)
        sleep 30
        ;;
    esac
  done
  printf '%s %s did not finish: %s\n' "$project_name" "$1" "$release_uuid" >&2
  exit 1
}

pin_assets
select_missing false
if [[ "${#missing[@]}" -eq 0 ]]; then
  printf '%s is already published on both indexes.\n' "$project_name"
  exit 0
fi

submit false
wait_for_release 'dry run'
if [[ "$upload" == true ]]; then
  dry_run_pins="$pinned"
  pin_assets
  if [[ "$pinned" != "$dry_run_pins" ]]; then
    printf 'The release assets changed during the dry run.\n' >&2
    exit 1
  fi
  submit true
  wait_for_release publication

  published=false
  for _ in {1..40}; do
    select_missing true
    if [[ "${#missing[@]}" -eq 0 ]]; then
      published=true
      break
    fi
    sleep 30
  done
  if [[ "$published" != true ]]; then
    printf 'Both indexes do not yet serve every %s file.\n' "$project_name" >&2
    exit 1
  fi
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  printf "Release \`%s\` completed for \`%s\` (%s).\n" \
    "$release_uuid" "$project_name" \
    "$(if [[ "$upload" == true ]]; then printf published; else printf validated; fi)" \
    >> "$GITHUB_STEP_SUMMARY"
fi
