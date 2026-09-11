#!/usr/bin/env bash
# Repo-native task registry. Requires Bash 4+ and standard Linux utilities only.
set -euo pipefail
export LC_ALL=C
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TODOS="$ROOT/todos"
KEYS=(ID Title Status Priority Assignee Created Updated 'Related Files / Commits / PRs')
SECTIONS=('Context & Problem Statement' 'Acceptance Criteria' 'Ideas & Candidate Solutions' 'Work Log & Progress' 'Verification & Findings')
declare -A META=()
declare -a FILES=() IDS=() TITLES=() STATUSES=() PRIORITIES=() ASSIGNEES=() UPDATED=()

fail() { printf 'todo: %s\n' "$*" >&2; exit 1; }
usage() { printf 'Usage: %s [list | new "Task Title" | view <id> | check]\n' "$0"; }

metadata() {
    local file=$1 template=${2:-false} rows key value normalized section
    META=()
    # Only the opening metadata table is machine-readable; discussion tables are free-form.
    rows=$(awk -F '|' '
        function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }
        /^## / { exit }
        /^\|/ {
            k=trim($2)
            if (k ~ /^(ID|Title|Status|Priority|Assignee|Created|Updated|Related Files \/ Commits \/ PRs)$/) {
                if (NF != 4 || seen[k]++) exit 1
                print k "\t" trim($3)
            }
        }
    ' "$file") || fail "$file: duplicate metadata key or malformed table row"
    while IFS=$'\t' read -r key value; do
        [[ -z "$key" ]] || META["$key"]=$value
    done <<< "$rows"
    for key in "${KEYS[@]}"; do
        [[ -n ${META[$key]:-} ]] || fail "$file: missing/empty $key"
    done
    if [[ $template == true ]]; then
        [[ ${META[ID]} == '{{ID}}' && ${META[Title]} == '{{TITLE}}' &&
           ${META[Created]} == '{{DATE}}' && ${META[Updated]} == '{{DATE}}' ]] ||
            fail "$file: expected ID, TITLE and DATE placeholders"
        META[ID]=001 META[Title]='Template task' META[Created]=2000-01-01 META[Updated]=2000-01-01
    elif grep -Eq '\{\{[^}]+\}\}' "$file"; then
        fail "$file: unresolved template placeholder"
    fi
    [[ ${META[ID]} =~ ^[0-9]{3}$ && ${META[ID]} != 000 ]] || fail "$file: ID must be 001..999"
    [[ ${META[Status]} =~ ^(BACKLOG|ACTIVE|IN_PROGRESS|BLOCKED|COMPLETED)$ ]] || fail "$file: invalid Status"
    [[ ${META[Priority]} =~ ^P[0-3]$ ]] || fail "$file: invalid Priority"
    [[ ${META[Assignee]} =~ ^(unassigned|human|antigravity|claude|pi)$ ]] || fail "$file: invalid Assignee"
    for key in Created Updated; do
        value=${META[$key]}
        [[ $value =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || fail "$file: invalid $key date"
        normalized=$(date -u -d "$value" +%F 2>/dev/null) || fail "$file: invalid $key date"
        [[ $normalized == "$value" ]] || fail "$file: invalid $key date"
    done
    [[ ${META[Updated]} < ${META[Created]} ]] && fail "$file: Updated precedes Created"
    for section in "${SECTIONS[@]}"; do
        grep -Fxq "## $section" "$file" || fail "$file: missing section '$section'"
    done
}

load_tasks() {
    local file name id
    local -A seen=()
    FILES=() IDS=() TITLES=() STATUSES=() PRIORITIES=() ASSIGNEES=() UPDATED=()
    [[ -d $TODOS ]] || fail "missing $TODOS"
    shopt -s nullglob dotglob
    for file in "$TODOS"/*; do
        name=${file##*/}
        case "$name" in README.md|template.md|.todo-new.lock) continue ;; esac
        [[ -f $file && ! -L $file && $name =~ ^([0-9]{3})_[a-z0-9]+(_[a-z0-9]+)*\.md$ ]] ||
            fail "unexpected task path: $file (expected NNN_lowercase_slug.md)"
        id=${BASH_REMATCH[1]}
        [[ ! ${seen[$id]+present} ]] || fail "duplicate task ID $id"
        seen[$id]=1
        metadata "$file"
        [[ ${META[ID]} == "$id" ]] || fail "$file: filename/metadata ID mismatch"
        FILES+=("$file") IDS+=("$id") TITLES+=("${META[Title]}") STATUSES+=("${META[Status]}")
        PRIORITIES+=("${META[Priority]}") ASSIGNEES+=("${META[Assignee]}") UPDATED+=("${META[Updated]}")
    done
    shopt -u nullglob dotglob
}

registry() {
    local i
    printf '| ID | Title | Status | Priority | Assignee | Last Updated |\n'
    printf '|---|---|---|---|---|---|\n'
    for ((i=0; i<${#FILES[@]}; i++)); do
        printf '| [%s](%s) | %s | %s | %s | %s | %s |\n' "${IDS[i]}" "${FILES[i]##*/}" \
            "${TITLES[i]}" "${STATUSES[i]}" "${PRIORITIES[i]}" "${ASSIGNEES[i]}" "${UPDATED[i]}"
    done
}

check_registry() {
    [[ -f $TODOS/README.md && -f $TODOS/template.md ]] || fail 'README.md and template.md are required'
    metadata "$TODOS/template.md" true
    awk '
        $0 == "<!-- TASKS:START -->" { if (++starts != 1 || ends) exit 1; next }
        $0 == "<!-- TASKS:END -->" { if (++ends != 1 || starts != 1) exit 1 }
        END { if (starts != 1 || ends != 1) exit 1 }
    ' "$TODOS/README.md" || fail 'README.md: expected one ordered TASKS:START/END marker pair'
    diff -u <(registry) <(awk '
        $0 == "<!-- TASKS:END -->" { inside=0 }
        inside { print }
        $0 == "<!-- TASKS:START -->" { inside=1 }
    ' "$TODOS/README.md") || fail 'README.md registry differs from task metadata; update its rows (expected shown as -)'
}

list_tasks() {
    local i status priority reset='' color='' pcolor=''
    [[ ! -t 1 || ${NO_COLOR+x} ]] || reset=$'\033[0m'
    printf '%-3s  %-64s  %-11s  %-8s  %-11s  %s\n' ID TITLE STATUS PRIORITY ASSIGNEE UPDATED
    for ((i=0; i<${#FILES[@]}; i++)); do
        status=${STATUSES[i]} priority=${PRIORITIES[i]}
        if [[ -n $reset ]]; then
            case $status in
                BACKLOG) color=$'\033[90m';; ACTIVE) color=$'\033[36m';;
                IN_PROGRESS) color=$'\033[33m';; BLOCKED) color=$'\033[31m';;
                COMPLETED) color=$'\033[32m';;
            esac
            case $priority in
                P0) pcolor=$'\033[1;31m';; P1) pcolor=$'\033[33m';;
                P2) pcolor=$'\033[36m';; P3) pcolor=$'\033[90m';;
            esac
        fi
        printf '%-3s  %-64s  %s%-11s%s  %s%-8s%s  %-11s  %s\n' "${IDS[i]}" "${TITLES[i]}" \
            "$color" "$status" "$reset" "$pcolor" "$priority" "$reset" "${ASSIGNEES[i]}" "${UPDATED[i]}"
    done
}

new_task() {
    local title=$1 slug id max=0 existing today path lock="$TODOS/.todo-new.lock"
    local created='' committed=false
    [[ -n ${title//[[:space:]]/} && ! $title =~ [[:cntrl:]\|] && $title != *'{{'* && $title != *'}}'* ]] ||
        fail 'title must be nonblank, single-line, and contain no pipe, control characters or template delimiters'
    title=$(printf '%s' "$title" | awk '{ sub(/^ +/, ""); sub(/ +$/, ""); print }')
    slug=$(printf '%s' "$title" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g')
    slug=${slug:0:80}; slug=${slug%_}
    [[ -n $slug ]] || fail 'title must contain at least one ASCII letter or digit for the filename'
    mkdir -- "$lock" 2>/dev/null || fail "new-task lock exists (or todos is unwritable): $lock; do not remove a live lock"
    # Only cooperating `new` calls in this worktree are serialized, not git clones or manual edits.
    trap 'if [[ $committed == false && -n $created ]]; then rm -f -- "$created"; fi; rm -f -- "$lock/task" "$lock/index" "$lock/readme"; rmdir -- "$lock"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    load_tasks
    check_registry
    for existing in "${IDS[@]}"; do
        (( 10#$existing <= max )) || max=$((10#$existing))
    done
    (( max < 999 )) || fail 'ID space exhausted (999); do not recycle completed task IDs'
    printf -v id '%03d' "$((max + 1))"
    today=$(date -u +%F)
    path="$TODOS/${id}_${slug}.md"
    # ENVIRON + literal replacement avoids sed replacement metacharacters in human titles.
    TODO_ID=$id TODO_TITLE=$title TODO_DATE=$today awk '
        function replace(s, token, value, p, out) {
            out=""
            while ((p=index(s, token))) { out=out substr(s,1,p-1) value; s=substr(s,p+length(token)) }
            return out s
        }
        { s=replace($0,"{{ID}}",ENVIRON["TODO_ID"])
          s=replace(s,"{{DATE}}",ENVIRON["TODO_DATE"])
          print replace(s,"{{TITLE}}",ENVIRON["TODO_TITLE"]) }
    ' "$TODOS/template.md" > "$lock/task"
    metadata "$lock/task"
    [[ ${META[Status]} == BACKLOG && ${META[Priority]} == P2 && ${META[Assignee]} == unassigned ]] ||
        fail 'template defaults must be BACKLOG / P2 / unassigned'
    FILES+=("$path") IDS+=("$id") TITLES+=("$title") STATUSES+=(BACKLOG)
    PRIORITIES+=(P2) ASSIGNEES+=(unassigned) UPDATED+=("$today")
    registry > "$lock/index"
    awk '
        FNR == NR { rows=rows $0 "\n"; next }
        $0 == "<!-- TASKS:START -->" { print; printf "%s", rows; inside=1; next }
        $0 == "<!-- TASKS:END -->" { inside=0 }
        !inside { print }
    ' "$lock/index" "$TODOS/README.md" > "$lock/readme"
    # Hard-link creation refuses to overwrite a task created by a non-cooperating writer.
    ln -- "$lock/task" "$path" || fail "cannot create $path"
    created=$path
    mv -- "$lock/readme" "$TODOS/README.md"
    committed=true
    printf '%s\n' "$path"
    # Run cleanup while these local variables are still in scope.
    rm -f -- "$lock/task" "$lock/index"
    rmdir -- "$lock"
    trap - EXIT INT TERM
}

command=${1:-list}
case "$command" in
    list) (( $# <= 1 )) || { usage >&2; exit 1; }; load_tasks; list_tasks ;;
    check) (( $# == 1 )) || { usage >&2; exit 1; }; load_tasks; check_registry
        printf 'OK: %d task(s); metadata, template and registry agree.\n' "${#FILES[@]}" ;;
    new) (( $# == 2 )) || { usage >&2; exit 1; }; new_task "$2" ;;
    view)
        (( $# == 2 )) || { usage >&2; exit 1; }
        [[ $2 =~ ^[0-9]{1,3}$ && $2 != 0 && $2 != 00 && $2 != 000 ]] || fail 'ID must be 1..999'
        printf -v wanted '%03d' "$((10#$2))"
        load_tasks
        for ((i=0; i<${#FILES[@]}; i++)); do
            if [[ ${IDS[i]} == "$wanted" ]]; then cat -- "${FILES[i]}"; exit 0; fi
        done
        fail "task $wanted not found" ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 1 ;;
esac
