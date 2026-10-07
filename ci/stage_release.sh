#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stage a Prepare Release bundle for nSpect review, or attest its launch readiness.
#
# GitHub runners cannot reach Artifactory or the nSpect v2 API, so a maintainer
# runs this on the NVIDIA network between the Prepare Release and Publish Release
# workflows:
#
#   ci/stage_release.sh stage vX.Y.Z  upload the bundle to Artifactory and
#                                     register it with nSpect
#   ci/stage_release.sh check vX.Y.Z  evaluate nSpect launch readiness and, on
#                                     pass, set the nspect/launch commit status
#
# Requires an authenticated gh, curl, jq, and sha256sum. Both commands read
# NSPECT_SSA_CLIENT_ID and NSPECT_SSA_CLIENT_SECRET; stage also reads
# ARTIFACTORY_TOKEN, which must be able to deploy to sw-cuspa-generic-local.
# Other settings come from the repository's GitHub variables, as in the workflows.

set -Eeuo pipefail
trap '((BASH_SUBSHELL)) || printf "%s: line %s failed: %s\n" "$0" "$LINENO" "$BASH_COMMAND" >&2' ERR

repository=NVIDIA/cuspa
nspect_api=https://nspect.nvidia.com/api/v2
ssa_token_url=https://4ubglassowmtsi7ogqwarmut7msn1q5ynts62fwnr1i.ssa.nvidia.com/token
status_context=nspect/launch

usage() {
  printf 'usage: %s stage|check vX.Y.Z[rcN]\n' "$0" >&2
  exit 2
}

# curl reads credentials from private config files so they stay out of argv.
auth_config() {
  (umask 077 && printf 'header = "%s"\n' "$2" > "$1")
}

repo_variable() {
  gh api "repos/${repository}/actions/variables/$1" --jq .value
}

staging_variable() {
  gh api "repos/${repository}/environments/release-staging/variables/$1" --jq .value
}

show_response() {
  jq . "$response" >&2 2> /dev/null || cat "$response" >&2
}

expect_status() {
  local got="$1"
  shift
  local want
  for want in "$@"; do
    [[ "$got" == "$want" ]] && return 0
  done
  printf 'nSpect returned HTTP %s.\n' "$got" >&2
  show_response
  return 1
}

# nspect METHOD PATH [BODY [CONTENT_TYPE]]: response body goes to $response,
# the HTTP status to stdout.
nspect() {
  local -a args=(
    --config "$nspect_auth" --silent --show-error
    --connect-timeout 10 --max-time 120
    -X "$1" -o "$response" -w '%{http_code}'
  )
  if [[ -n "${3:-}" ]]; then
    args+=(-H "Content-Type: ${4:-application/json}" --data "$3")
  fi
  curl "${args[@]}" "${nspect_api}$2"
}

retry_after_seconds() {
  local seconds
  seconds="$(
    awk 'tolower($1) == "retry-after:" {gsub("\\r", "", $2); print $2}' "$1"
  )"
  if [[ ! "$seconds" =~ ^[0-9]+$ ]] || ((seconds > 60)); then
    printf 'nSpect rate-limited the request with Retry-After %s.\n' "$seconds" >&2
    return 1
  fi
  printf '%s' "$seconds"
}

load_release() {
  local on_main
  local run_id

  artifactory_url="$(repo_variable ARTIFACTORY_URL)"
  artifactory_url="${artifactory_url%/}"
  artifactory_repository="$(repo_variable ARTIFACTORY_REPOSITORY)"
  nspect_id="$(staging_variable NSPECT_ID)"
  [[ "$artifactory_url" == https://artifactory.nvidia.com ]]
  [[ "$artifactory_repository" == sw-cuspa-generic-local ]]
  [[ "$nspect_id" =~ ^NSPECT-[A-Za-z0-9]+-[A-Za-z0-9]+$ ]]
  if [[ "$version" == *rc* ]]; then
    release_type="$(staging_variable NSPECT_PRERELEASE_TYPE)"
  else
    release_type="$(staging_variable NSPECT_FINAL_TYPE)"
  fi
  [[ -n "$release_type" ]]

  sha="$(gh api "repos/${repository}/commits/${tag}" --jq .sha)"
  [[ "$sha" =~ ^[0-9a-f]{40}$ ]]
  on_main="$(gh api "repos/${repository}/compare/${sha}...main" --jq .status)"
  [[ "$on_main" == ahead || "$on_main" == identical ]]
  source_base="${artifactory_url}/artifactory/${artifactory_repository}/cuspa/nspect/${nspect_id}/${tag}/${sha}"

  run_id="$(
    gh run list --repo "$repository" --workflow release.yml --event push \
      --branch "$tag" --status success --limit 20 --json databaseId,headSha |
      jq -r --arg sha "$sha" \
        '[.[] | select(.headSha == $sha)][0].databaseId // empty'
  )"
  if [[ ! "$run_id" =~ ^[0-9]+$ ]]; then
    printf 'No successful Prepare Release run for %s at %s.\n' "$tag" "$sha" >&2
    exit 1
  fi
  gh run download "$run_id" --repo "$repository" \
    --name "cuspa-release-${tag}" --dir "$bundle"
  verify_bundle
  printf 'Verified the bundle from Prepare Release run %s (manifest %s).\n' \
    "$run_id" "${manifest_digest:0:12}"
}

verify_bundle() {
  local expected_sums="${work}/SHA256SUMS"
  local name
  local size
  local digest
  local pointer

  jq -e \
    --arg repository "$repository" \
    --arg tag "$tag" \
    --arg version "$version" \
    --arg commit "$sha" '
      .schemaVersion == 1 and
      .repository == $repository and
      .tag == $tag and
      .version == $version and
      .commit == $commit and
      (.artifacts | length) == 5 and
      ([.artifacts[].name] | unique | length) == 5
    ' "$manifest" > /dev/null

  : > "$expected_sums"
  while IFS=$'\t' read -r name size digest pointer; do
    [[ "$name" == "$(basename "$name")" ]]
    [[ "$name" =~ ^cuspa(_cu(12|13))?-[A-Za-z0-9.+]+.*(\.whl|\.tar\.gz)$ ]]
    [[ "$pointer" == "${source_base}/${name}" ]]
    [[ "$(stat -c %s "${bundle}/${name}")" == "$size" ]]
    [[ "$(sha256sum "${bundle}/${name}" | cut -d ' ' -f 1)" == "$digest" ]]
    printf '%s  %s\n' "$digest" "$name" >> "$expected_sums"
  done < <(
    jq -r '.artifacts[] | [.name, .size, .sha256, .sourcePointer] | @tsv' \
      "$manifest"
  )
  cmp -s "$expected_sums" "${bundle}/SHA256SUMS"
  [[ "$(find "$bundle" -maxdepth 1 -type f | wc -l)" -eq 7 ]]
  manifest_digest="$(sha256sum "$manifest" | cut -d ' ' -f 1)"
}

stage_artifactory() {
  local artifactory_auth="${work}/artifactory.curlrc"
  local -a files
  local file
  local name
  local digest
  local size
  local url
  local remote
  local status

  auth_config "$artifactory_auth" "Authorization: Bearer ${ARTIFACTORY_TOKEN}"
  mapfile -t files < <(
    find "$bundle" -maxdepth 1 -type f ! -name manifest.json -print | sort
  )
  files+=("$manifest")
  [[ "${#files[@]}" -eq 7 ]]
  for file in "${files[@]}"; do
    name="$(basename "$file")"
    digest="$(sha256sum "$file" | cut -d ' ' -f 1)"
    size="$(stat -c %s "$file")"
    url="${source_base}/${name}"
    remote="${work}/remote-${name}"

    status="$(
      curl --config "$artifactory_auth" --silent --show-error --location \
        --retry 3 --retry-all-errors --connect-timeout 10 \
        --max-time 900 --proto '=https' --proto-redir '=https' \
        -o "$remote" -w '%{http_code}' "$url"
    )"
    case "$status" in
      200)
        printf 'Already staged: %s\n' "$name"
        ;;
      404)
        curl --config "$artifactory_auth" --fail-with-body --silent --show-error \
          --retry 3 --retry-all-errors --connect-timeout 10 \
          --max-time 900 --proto '=https' \
          -H "X-Checksum-Sha256: ${digest}" \
          --upload-file "$file" "$url" > /dev/null
        printf 'Uploaded: %s\n' "$name"
        ;;
      *)
        printf 'Artifactory returned HTTP %s for %s.\n' "$status" "$url" >&2
        exit 1
        ;;
    esac

    curl --config "$artifactory_auth" --fail-with-body --silent --show-error \
      --location --retry 3 --retry-all-errors --connect-timeout 10 \
      --max-time 900 --proto '=https' --proto-redir '=https' \
      -o "$remote" "$url"
    if [[ "$(stat -c %s "$remote")" != "$size" ]] ||
      [[ "$(sha256sum "$remote" | cut -d ' ' -f 1)" != "$digest" ]]; then
      printf 'The staged %s does not match the bundle.\n' "$url" >&2
      exit 1
    fi
  done
}

nspect_login() {
  local ssa_auth="${work}/ssa.curlrc"
  local basic
  local token

  basic="$(
    printf '%s:%s' "$NSPECT_SSA_CLIENT_ID" "$NSPECT_SSA_CLIENT_SECRET" |
      base64 -w 0
  )"
  auth_config "$ssa_auth" "Authorization: Basic ${basic}"
  token="$(
    curl --config "$ssa_auth" --fail-with-body --silent --show-error \
      --connect-timeout 10 --max-time 60 \
      --data-urlencode grant_type=client_credentials \
      --data-urlencode 'scope=public.api:read public.api:write' \
      "$ssa_token_url" |
      jq -er '.access_token | strings | select(length > 0)'
  )"
  auth_config "$nspect_auth" "Authorization: Bearer ${token}"
}

wait_for_job() {
  local job_url="https://nspect.nvidia.com$1"
  local headers="${work}/headers"
  local attempt
  local status
  local delay

  for attempt in {1..180}; do
    status="$(
      curl --config "$nspect_auth" --silent --show-error \
        --connect-timeout 10 --max-time 120 -D "$headers" \
        -o "$response" -w '%{http_code}' "$job_url"
    )"
    case "$status" in
      200)
        case "$(jq -er '.data.job.status' "$response")" in
          SUCCESS)
            return 0
            ;;
          PENDING | STARTED | PROCESSING | RETRY | PROGRESS)
            sleep 10
            ;;
          *)
            show_response
            return 1
            ;;
        esac
        ;;
      429)
        delay="$(retry_after_seconds "$headers")"
        sleep "$delay"
        ;;
      502 | 503 | 504)
        sleep "$((attempt < 6 ? attempt * 5 : 30))"
        ;;
      *)
        expect_status "$status" 200
        ;;
    esac
  done
  printf 'nSpect job %s did not finish.\n' "$1" >&2
  return 1
}

register_nspect() {
  local release="/public/programs/${nspect_id}/releases/${tag}"
  local owner_email
  local owner_name
  local release_date
  local status
  local body
  local patch
  local registration
  local job_uri

  owner_email="$(staging_variable NSPECT_RELEASE_OWNER_EMAIL)"
  owner_name="$(staging_variable NSPECT_RELEASE_OWNER_NAME 2> /dev/null)" ||
    owner_name=""
  release_date="$(gh api "repos/${repository}/commits/${sha}" --jq .commit.committer.date)"
  release_date="${release_date:0:10}"
  [[ "$release_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]

  status="$(nspect GET "$release")"
  if [[ "$status" == 404 ]]; then
    body="$(
      jq -nc \
        --arg name "$tag" \
        --arg description "Cuspa ${tag}" \
        --arg date "$release_date" \
        --arg email "$owner_email" \
        --arg owner "$owner_name" \
        '{name: $name, description: $description, releaseDate: $date, owner: ({email: $email} + (if $owner == "" then {} else {name: $owner} end)), labels: ["github-release"]}'
    )"
    status="$(nspect POST "/public/programs/${nspect_id}/releases" "$body")"
    if [[ "$status" == 409 ]]; then
      status="$(nspect GET "$release")"
      expect_status "$status" 200
    else
      expect_status "$status" 201
    fi
  else
    expect_status "$status" 200
  fi
  jq -e --arg name "$tag" \
    '.success == true and .data.release.name == $name' "$response" > /dev/null
  patch="$(
    jq -c '
      {
        deploymentTypes: {isDownloadable: true},
        releaseLocations: (
          ([.data.release.releaseLocations[]?.name] + ["GitHub"]) | unique
        )
      }
    ' "$response"
  )"

  status="$(nspect GET "${release}/release-types?pageSize=1000")"
  expect_status "$status" 200
  jq -e '
    [.data.releaseTypes[] | select(.regulation != null)] | length == 0
  ' "$response" > /dev/null

  status="$(nspect GET "/public/release-types?pageSize=1000")"
  expect_status "$status" 200
  jq -e --arg type "$release_type" '
    .page.total <= 1000 and
    ([.data.releaseTypes[] |
      select(.name == $type and .regulation == null)] | length) == 1
  ' "$response" > /dev/null

  body="$(jq -nc --arg type "$release_type" '{releaseTypes: [{releaseType: $type}]}')"
  status="$(nspect PUT "${release}/release-types" "$body")"
  expect_status "$status" 200
  jq -e '.success == true' "$response" > /dev/null

  status="$(nspect PATCH "$release" "$patch" application/merge-patch+json)"
  expect_status "$status" 200
  jq -e '
    .success == true and
    .data.release.deploymentTypes.isDownloadable == true and
    ([.data.release.releaseLocations[].name] | index("GitHub")) != null
  ' "$response" > /dev/null

  status="$(nspect GET "${release}/binary-artifacts?pageSize=1000")"
  expect_status "$status" 200
  jq -e --slurpfile manifest "$manifest" '
    ($manifest[0].artifacts | map(.sourcePointer)) as $expected |
    (.page.total <= 1000) and
    ([.data.binaryArtifacts[] |
      select(.sourcePointer as $url | ($expected | index($url)) == null)] |
      length == 0) and
    ([.data.binaryArtifacts[] as $artifact |
      select(($expected | index($artifact.sourcePointer)) != null) |
      select($artifact.includedInRelease != true)] | length == 0)
  ' "$response" > /dev/null

  registration="$(
    jq -n \
      --slurpfile manifest "$manifest" \
      --slurpfile current "$response" '
        ($current[0].data.binaryArtifacts | map(.sourcePointer)) as $existing |
        {binaryArtifacts: [
          $manifest[0].artifacts[] |
          select(.sourcePointer as $url | ($existing | index($url)) == null) |
          {sourcePointer: .sourcePointer, includedInRelease: true}
        ]}
      '
  )"
  if [[ "$(jq '.binaryArtifacts | length' <<< "$registration")" -gt 0 ]]; then
    status="$(nspect POST "${release}/binary-artifacts" "$registration")"
    expect_status "$status" 202
    jq -e '
      .success == true and
      (.data.jobURI | startswith("/api/v2/public/jobs/"))
    ' "$response" > /dev/null
    job_uri="$(jq -er '.data.jobURI' "$response")"
    wait_for_job "$job_uri"
  fi

  status="$(nspect GET "${release}/binary-artifacts?pageSize=1000")"
  expect_status "$status" 200
  jq -e --slurpfile manifest "$manifest" '
    ($manifest[0].artifacts | map(.sourcePointer) | sort) as $expected |
    ([.data.binaryArtifacts[].sourcePointer] | sort) == $expected and
    ([.data.binaryArtifacts[] | select(.includedInRelease != true)] |
      length == 0)
  ' "$response" > /dev/null
  printf 'Registered %s with nSpect %s as %s.\n' "$tag" "$nspect_id" "$release_type"
  printf 'After the nSpect scans finish, run: %s check %s\n' "$0" "$tag"
}

check_launch() {
  local release="/public/programs/${nspect_id}/releases/${tag}"
  local headers="${work}/headers"
  local encoded_type
  local launch
  local attempt
  local status
  local delay
  local description

  status="$(nspect GET "${release}/binary-artifacts?pageSize=1000")"
  expect_status "$status" 200
  jq -e --slurpfile manifest "$manifest" '
    ($manifest[0].artifacts | map(.sourcePointer) | sort) as $expected |
    (.page.total <= 1000) and
    ([.data.binaryArtifacts[].sourcePointer] | sort) == $expected and
    ([.data.binaryArtifacts[] | select(.includedInRelease != true)] |
      length == 0)
  ' "$response" > /dev/null

  status="$(nspect GET "${release}/release-types?pageSize=1000")"
  expect_status "$status" 200
  jq -e --arg type "$release_type" '
    .page.total == 1 and
    ([.data.releaseTypes[] | select(.regulation == null) | .name] == [$type])
  ' "$response" > /dev/null

  encoded_type="$(jq -rn --arg value "$release_type" '$value | @uri')"
  launch="/launch/programs/${nspect_id}/releases/${tag}?releaseType=${encoded_type}"
  for attempt in {1..20}; do
    status="$(
      curl --config "$nspect_auth" --silent --show-error \
        --connect-timeout 10 --max-time 180 -D "$headers" \
        -o "$response" -w '%{http_code}' "${nspect_api}${launch}"
    )"
    case "$status" in
      200)
        break
        ;;
      429)
        delay="$(retry_after_seconds "$headers")"
        sleep "$delay"
        ;;
      500 | 504)
        sleep "$((attempt < 6 ? attempt * 5 : 30))"
        ;;
      *)
        expect_status "$status" 200
        ;;
    esac
  done
  expect_status "$status" 200
  jq -e --arg id "$nspect_id" --arg release "$tag" '
    .success == true and
    .data.nspectId == $id and
    .data.releaseVersion == $release
  ' "$response" > /dev/null

  if ! jq -e '
    .data.launchResult == "pass" and (.data.partial // false) == false
  ' "$response" > /dev/null; then
    printf 'nSpect launch readiness for %s is not passing:\n' "$tag" >&2
    jq -r '
      "  launchResult: \(.data.launchResult)\(if .data.partial then " (partial)" else "" end)",
      (.data.requirements // {} | to_entries[] | .key as $group | .value[]? |
        select(.status != "pass") | "  \($group): \(.name) [\(.id)]")
    ' "$response" >&2
    exit 1
  fi

  description="nSpect launch pass: ${release_type}; manifest ${manifest_digest:0:12}"
  ((${#description} <= 140))
  gh api --method POST "repos/${repository}/statuses/${sha}" \
    -f state=success \
    -f context="$status_context" \
    -f description="$description" \
    -f target_url="https://nspect.nvidia.com/beta/program/${nspect_id}/release-versions/${tag}/requirements" \
    > /dev/null
  printf 'Set %s on %s: %s\n' "$status_context" "$sha" "$description"
}

main() {
  command="${1:-}"
  tag="${2:-}"
  if [[ "$command" != stage && "$command" != check ]] ||
    [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?$ ]]; then
    usage
  fi
  version="${tag#v}"
  : "${NSPECT_SSA_CLIENT_ID:?NSPECT_SSA_CLIENT_ID is required}"
  : "${NSPECT_SSA_CLIENT_SECRET:?NSPECT_SSA_CLIENT_SECRET is required}"
  if [[ "$command" == stage ]]; then
    : "${ARTIFACTORY_TOKEN:?ARTIFACTORY_TOKEN is required}"
  fi

  work="$(mktemp -d)"
  trap 'rm -rf -- "$work"' EXIT
  bundle="${work}/bundle"
  manifest="${bundle}/manifest.json"
  response="${work}/response.json"
  nspect_auth="${work}/nspect.curlrc"

  load_release
  nspect_login
  case "$command" in
    stage)
      stage_artifactory
      register_nspect
      ;;
    check)
      check_launch
      ;;
  esac
}

main "$@"
