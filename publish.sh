#!/usr/bin/env bash
# Publish a sanitised snapshot of this repo to the public mirror
# (JuniperSP/DT-Bridge-OpenAI-ComfyUI).
#
# Why a snapshot and not a normal push: this repository's history still contains
# an early commit with personal model/LoRA names, so publishing our history
# would leak them. Instead we build ONE commit whose tree is the current HEAD
# and whose parent is the public repo's current tip, then push that.
#
#   ./publish.sh [--dry-run] [--message "text"]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

PUBLIC_REPO="${DT_PUBLIC_REPO:-JuniperSP/DT-Bridge-OpenAI-ComfyUI}"
PUBLIC_URL="https://github.com/${PUBLIC_REPO}.git"
PUBLIC_ACCOUNT="${DT_PUBLIC_ACCOUNT:-JuniperSP}"

c_reset=$'\033[0m'; c_dim=$'\033[2m'; c_red=$'\033[31m'; c_green=$'\033[32m'; c_yellow=$'\033[33m'
ok() { printf '%s✔%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s!%s %s\n' "$c_yellow" "$c_reset" "$*"; }
err() { printf '%s✘%s %s\n' "$c_red" "$c_reset" "$*" >&2; }
die() { err "$*"; exit 1; }

DRY_RUN=0
MESSAGE=""
args=("$@")
i=0
while [ "$i" -lt "${#args[@]}" ]; do
  case "${args[$i]}" in
    --dry-run|-n) DRY_RUN=1 ;;
    --message|-m) i=$((i + 1)); MESSAGE="${args[$i]:-}" ;;
    -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac
  i=$((i + 1))
done

# ------------------------------------------------------------------ preflight
[ "$(git rev-parse --is-inside-work-tree 2>/dev/null)" = "true" ] || die "not a git repository"

if [ -n "$(git status --porcelain)" ]; then
  err "working tree is dirty - commit or stash first so the snapshot is reproducible"
  git --no-pager status --short | head -10
  exit 1
fi

# Scan the exact tree we are about to publish.
TREE="$(git rev-parse 'HEAD^{tree}')"
# The word list lives in .publish-ignore (git-ignored) so the words themselves
# are not published either. Without it we can still catch paths and keys.
DENY_FILE="$ROOT/.publish-ignore"
if [ -f "$DENY_FILE" ]; then
  SENSITIVE="$(grep -vE '^[[:space:]]*(#|$)' "$DENY_FILE" | paste -sd'|' -)"
else
  warn "no .publish-ignore - only generic patterns are checked"
  SENSITIVE='/Users/'
fi
if [ -n "$SENSITIVE" ] && git grep -I -n -E "$SENSITIVE" "$TREE" -- . ':(exclude)publish.sh' >/tmp/dt-publish-scan 2>/dev/null; then
  err "refusing to publish: sensitive strings found in the snapshot"
  sed -n '1,10p' /tmp/dt-publish-scan | sed 's/^/    /'
  exit 1
fi
SECRETS='sk-[A-Za-z0-9]{12,}|gho_[A-Za-z0-9]{12,}|github_pat_[A-Za-z0-9_]{12,}|api[_-]?key["'"']?\s*[:=]\s*["'"'][^"'"']{12,}'
if git grep -I -n -E "$SECRETS" "$TREE" -- . ':(exclude)publish.sh' >/tmp/dt-publish-secrets 2>/dev/null; then
  err "refusing to publish: something that looks like a secret was found"
  sed -n '1,10p' /tmp/dt-publish-secrets | sed 's/^/    /'
  exit 1
fi
ok "snapshot tree is clean (no personal strings, no secrets)"

# ------------------------------------------------------------------- fetch
if ! git remote get-url public >/dev/null 2>&1; then
  git remote add public "$PUBLIC_URL"
  ok "added remote 'public' → $PUBLIC_REPO"
fi
git fetch -q public main || die "cannot fetch $PUBLIC_REPO (is it still there?)"
PARENT="$(git rev-parse public/main)"

if [ "$(git rev-parse "$PARENT^{tree}")" = "$TREE" ]; then
  ok "nothing to publish - public repo already matches this tree"
  exit 0
fi

TOKEN="$(gh auth token --user "$PUBLIC_ACCOUNT" 2>/dev/null)" || true
if [ -z "$TOKEN" ]; then
  die "no gh token for $PUBLIC_ACCOUNT (run: gh auth login --user $PUBLIC_ACCOUNT)"
fi

# ---------------------------------------------------------------- commit
AUTHOR_NAME="$(git log -1 --format=%an "$PARENT" 2>/dev/null || echo "$PUBLIC_ACCOUNT")"
AUTHOR_EMAIL="$(git log -1 --format=%ae "$PARENT" 2>/dev/null || echo "${PUBLIC_ACCOUNT}@users.noreply.github.com")"
[ -n "$MESSAGE" ] || MESSAGE="Snapshot $(date '+%Y-%m-%d %H:%M') from the development repo"

COMMIT="$(
  GIT_AUTHOR_NAME="$AUTHOR_NAME" GIT_AUTHOR_EMAIL="$AUTHOR_EMAIL" \
  GIT_COMMITTER_NAME="$AUTHOR_NAME" GIT_COMMITTER_EMAIL="$AUTHOR_EMAIL" \
  git commit-tree "$TREE" -p "$PARENT" -m "$MESSAGE"
)"

printf '\n%sabout to publish%s\n' "$c_dim" "$c_reset"
git --no-pager diff --stat "$PARENT" "$COMMIT" | tail -12

if [ "$DRY_RUN" = "1" ]; then
  warn "dry run - nothing pushed"
  printf '  would push %s → %s:main\n' "${COMMIT:0:7}" "$PUBLIC_REPO"
  exit 0
fi

AUTH_HEADER="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$TOKEN" | base64)"
if git -c "http.https://github.com/.extraheader=$AUTH_HEADER" push -q public "$COMMIT:refs/heads/main"; then
  ok "published ${COMMIT:0:7} → https://github.com/$PUBLIC_REPO"
  git fetch -q public main
else
  die "push failed (check that $PUBLIC_ACCOUNT still has write access)"
fi
