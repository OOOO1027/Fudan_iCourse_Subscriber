#!/usr/bin/env bash
# Mirror the board store (src/pipeline/board_store.py) to boards/ on the
# `boards` branch.
#
#   board_branch.sh fetch DIR   put the branch's index.enc into DIR, mark DIR ready
#   board_branch.sh push DIR    commit DIR's *.enc and *.rev on top of the branch
#
# BOARD_REMOTE is the repository URL (with a token for `push`).  The branch only
# grows: every push adds one commit on top of the commit that `fetch` saw, and
# is refused if the branch has moved since.  Work happens in a scratch
# repository that fetches no file contents except the index, so the lecture
# bundles already on the branch are never downloaded.
set -euo pipefail

cmd=$1
dir=$2
repo=${BOARD_GIT_DIR:-${RUNNER_TEMP:-/tmp}/boards.git}
mkdir -p "$dir"

case "$cmd" in
fetch)
  rm -f "$dir/.ready"
  rm -rf "$repo"
  git init --quiet --bare "$repo"
  git -C "$repo" remote add origin "$BOARD_REMOTE"
  set +e
  git -C "$repo" ls-remote --exit-code --heads origin boards >/dev/null 2>&1
  code=$?
  set -e
  if [ "$code" -eq 2 ]; then
    echo "No boards branch yet."
    echo none > "$dir/.ready"
  elif [ "$code" -eq 0 ]; then
    git -C "$repo" fetch --quiet --depth=1 --filter=blob:none origin "+refs/heads/boards:refs/heads/boards"
    if git -C "$repo" ls-tree --name-only boards boards/ | grep -qx boards/index.enc; then
      git -C "$repo" show boards:boards/index.enc > "$dir/index.enc"
    fi
    git -C "$repo" rev-parse boards > "$dir/.ready"
    echo "Boards branch holds $(git -C "$repo" ls-tree --name-only boards boards/ | grep -c '\.enc$' || true) encrypted file(s)."
  else
    echo "::warning::Could not reach the boards branch; board pages are off for this run."
    exit 1
  fi
  ;;
push)
  if [ ! -f "$dir/.ready" ]; then
    echo "The board store was not fetched; nothing to push."
    exit 0
  fi
  parent=$(cat "$dir/.ready")
  export GIT_DIR="$repo" GIT_INDEX_FILE="$repo/boards.index"
  rm -f "$GIT_INDEX_FILE"
  if [ "$parent" != none ]; then
    git read-tree "$parent"
  fi
  count=0
  for path in "$dir"/*.enc "$dir"/*.rev; do
    [ -f "$path" ] || continue
    blob=$(git hash-object -w "$path")
    git update-index --add --cacheinfo "100644,$blob,boards/$(basename "$path")"
    count=$((count + 1))
  done
  if [ "$count" -eq 0 ]; then
    echo "No board files to push."
    exit 0
  fi
  tree=$(git write-tree --missing-ok)
  if [ "$parent" != none ] && [ "$tree" = "$(git rev-parse "$parent^{tree}")" ]; then
    echo "Board pages unchanged."
    exit 0
  fi
  export GIT_AUTHOR_NAME="github-actions[bot]" GIT_COMMITTER_NAME="github-actions[bot]"
  export GIT_AUTHOR_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"
  export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
  if [ "$parent" != none ]; then
    commit=$(git commit-tree "$tree" -p "$parent" -m "chore: update encrypted board pages")
  else
    commit=$(git commit-tree "$tree" -m "chore: add encrypted board pages")
  fi
  git push --quiet origin "$commit:refs/heads/boards"
  echo "$commit" > "$dir/.ready"
  echo "Pushed board pages."
  ;;
*)
  echo "usage: board_branch.sh fetch|push DIR" >&2
  exit 2
  ;;
esac
