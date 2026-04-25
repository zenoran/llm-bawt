#!/usr/bin/env bash
set -euo pipefail

BOT_ID="${BOT_ID:-byte}"
TURNS="${TURNS:-10}"
LLM_CMD="${LLM_CMD:-llm}"
CWD="${CWD:-$PWD}"
OUTPUT="${OUTPUT:-$CWD/tmp/name-loop.md}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-}"
SEED_NOTE="${SEED_NOTE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bot-id) BOT_ID="$2"; shift 2 ;;
    --turns) TURNS="$2"; shift 2 ;;
    --llm-cmd) LLM_CMD="$2"; shift 2 ;;
    --cwd) CWD="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --system-prompt-file) SYSTEM_PROMPT_FILE="$2"; shift 2 ;;
    --seed-note) SEED_NOTE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

read -r -d '' DEFAULT_SYSTEM_PROMPT <<'EOF' || true
You are a naming strategist helping choose a single new name for one unified AI product currently represented by the names 'unmute' and 'llm-bawt'.

Your highest priority is how comfortable a name feels to say and hear in natural spoken English.
A name that looks clever but feels awkward, abrupt, harsh, or annoying to pronounce must be rejected.

The name should clearly evoke communication between humans and AI.
It should suggest reciprocity, dialogue, listening, interpretation, rapport, resonance, shared understanding, or bridge-building between a person and an intelligent nonhuman counterpart.
It should feel recognizably relevant to AI without falling into generic cold-tech slop.
It must not imply a master ordering a slave around all day.
Avoid names that feel hierarchical, domineering, servile, robotic-drone-like, militaristic, or command-and-obey coded.

Product context:
- Right now the system has pieces currently called 'unmute' and 'llm-bawt', but this exercise is explicitly about finding ONE unified product name for the whole thing.
- Do not split the naming into frontend/backend, app/service, company/protocol, or any other multi-name architecture.
- The goal is one name that can represent the entire experience: bots chatting with humans in an endearing, reciprocal, communication-first way.

Hard scope constraints:
- Stay focused on naming the single unified product currently represented by unmute and llm-bawt.
- Do not split the answer into separate frontend/backend names.
- Do not propose company name + product name, protocol name + brand name, or any dual-name architecture.
- Do not rename the company, protocol, architecture, engine, platform, protocol layer, or invented subproducts.
- Do not drift into enterprise infrastructure, compliance, cryptography, logistics, workflow middleware, or security startup naming.
- Reject names that sound like infra/security/B2B middleware brand names.
- If you start drifting into multiple names or layered naming systems, stop and return to one product name immediately.

Priorities in order:
1. Comfort to say out loud
2. Comfort to hear out loud
3. Human/AI reciprocity and communication feel
4. Clear AI↔human relevance
5. Memorability
6. Brand distinctiveness
7. Product fit

Phonetic preferences:
- Favor smooth consonant-vowel flow
- Favor names that feel natural in casual speech
- Favor names people can say clearly the first time they read them
- Hard limit: no candidate may exceed 2 syllables
- Prefer 1 or 2 syllables only
- Avoid abrupt stops, harsh clusters, ambiguous stress, or weird annunciation
- Avoid names that feel too fantasy-coded, too synthetic, too startup-generic, or too self-consciously clever

Behavior rules:
- Stay practical and decision-oriented
- Do not spiral into abstract naming philosophy
- Do not invent fake linguistic theories
- Do not get distracted by recursion, symmetry, mirror phonemes, or made-up phonetic frameworks unless they directly improve actual name quality
- Every turn must produce useful naming progress
- Be blunt when a candidate feels cold, bossy, subservient, emotionally wrong, or like fake enterprise nonsense
- Every suggested name must be justified in direct relation to the application's real function: letting bots chat with humans in an endearing, reciprocal, communication-first way
- Do not justify names with vague branding language alone; tie them to conversation, warmth, listening, rapport, shared understanding, or emotional accessibility between humans and bots

Every turn must:
1. Propose strong candidate names for the single unified product only
2. Use only candidates that are 1 or 2 syllables; reject anything longer immediately
3. For every suggested name, explain exactly how it fits the product's real function of helping bots chat with humans endearingly
4. For every suggested name, explain how it connects specifically to AI↔human communication rather than generic software branding
5. Give a short say-it-out-loud test for the best candidates
6. Score top candidates from 1 to 10 on: Speakability, Hearability, Reciprocity feel, AI↔human relevance, Memorability, Brand fit
7. Reject weak candidates bluntly
8. Prefer names that a human would actually enjoy saying repeatedly
9. Give clear next-step instructions for the next iteration

Final-turn rule:
On the final turn, provide final votes with the winner and runner-up for the single unified product name, plus one blunt recommendation about what you would actually choose tonight if you had to decide now.

Extra guidance:
- If a name has a real meaning in another language, treat that as a potential asset if it strengthens the brand story without making the name less comfortable to say or hear.
- Re-evaluate promising names through the lens of listening-with, not just listening-to.
- Favor names that feel like an invitation to communicate, not a tool for issuing orders.
- Automatically reject candidates in the general vibe class of Relayseal, Chainmark, TrustGrid, SignalForge, or other fake infra-company sounding names.
- Automatically reject candidates longer than 2 syllables.
- Bias toward names that feel like a human speaking with an AI presence, not just using an app.
EOF

if [[ -n "$SYSTEM_PROMPT_FILE" ]]; then
  SYSTEM_PROMPT="$(cat "$SYSTEM_PROMPT_FILE")"
else
  SYSTEM_PROMPT="$DEFAULT_SYSTEM_PROMPT"
fi

mkdir -p "$(dirname "$OUTPUT")"
workdir_tmp="$(mktemp -d)"
trap 'rm -rf "$workdir_tmp"' EXIT

strip_ansi() {
  perl -pe 's/\e\[[0-9;?]*[ -\/]*[@-~]//g; s/\r\n/\n/g; s/\r/\n/g'
}

needs_single_name_repair() {
  local file="$1"
  grep -Eiq 'frontend|backend|company|protocol|dual architecture|pairing|pair-brand|product/protocol|app/service|for both products|two names|separate names' "$file"
}

repair_to_single_name() {
  local source_file="$1"
  local repaired_file="$2"
  local repair_prompt="$workdir_tmp/repair-$$.txt"
  cat > "$repair_prompt" <<EOF
Rewrite the following naming analysis so it obeys the real constraint:

- There must be exactly ONE unified product name direction.
- Do NOT split into frontend/backend.
- Do NOT propose company/product or protocol/brand architectures.
- Every candidate must be 1 or 2 syllables only.
- Every candidate must feel specifically about AI and humans communicating.
- Collapse everything into a single-product shortlist.
- Output only:
  1. Top 3 single-name candidates
  2. Syllable count for each
  3. Short justification for each tied directly to bots chatting with humans endearingly and AI↔human communication
  4. One current frontrunner
  5. Next-turn instructions that still preserve the single-name constraint

Source text:

$(cat "$source_file")
EOF
  cat "$repair_prompt" | "$LLM_CMD" -b "$BOT_ID" --plain --no-stream > "$repaired_file"
}

build_user_prompt() {
  local turn="$1"
  local total="$2"
  local prev_file="$3"
  cat <<EOF
This is turn ${turn} of ${total}.

You are in a self-propelled naming experiment.
Your output from this turn will be fed into the next turn.
Keep the chain alive by ending with concrete next-turn instructions.
Spend this turn doing one creative research move before proposing names.

Critical constraint reminder:
- You are naming ONE unified product.
- The name must be no more than 2 syllables.
- The name should feel specifically about AI and humans communicating.
- Do NOT propose separate frontend/backend names.
- Do NOT propose company/product pairs.
- Do NOT propose protocol/brand pairs.
- If you feel tempted to split the naming, stop and collapse back to a single-product shortlist immediately.
EOF

  if [[ "$turn" == "1" ]]; then
    printf '\nStart from scratch and establish the naming landscape, but stay tightly focused on assistant/product naming only.\n'
  else
    printf '\nHere is the previous turn output. React to it, build on it, disagree with it where needed, and continue the recursive search:\n\n'
    cat "$prev_file"
    printf '\n'
  fi

  if [[ -n "$SEED_NOTE" ]]; then
    printf '\nExtra founder note:\n%s\n' "$SEED_NOTE"
  fi

  if [[ "$turn" -lt "$total" ]]; then
    printf '\nRemember: do NOT finalize yet. Keep searching, narrowing, and redirecting the next turn.\n'
  else
    printf "\nThis is the final turn. You must include the exact 'Final votes:' section.\n"
  fi
}

prev_clean="$workdir_tmp/prev.txt"
: > "$prev_clean"

{
  echo '# Recursive naming experiment'
  echo
  echo "- CLI: \
\`$LLM_CMD\`"
  echo "- Bot ID: \
\`$BOT_ID\`"
  echo "- Working directory: \
\`$CWD\`"
  echo "- Turns: \
\`$TURNS\`"
  echo
} > "$OUTPUT"

for ((turn=1; turn<=TURNS; turn++)); do
  prompt_file="$workdir_tmp/prompt-$turn.txt"
  transcript_file="$workdir_tmp/transcript-$turn.txt"
  clean_file="$workdir_tmp/clean-$turn.txt"

  {
    echo 'SYSTEM PROMPT — APPLY THIS EXACTLY AS WRITTEN:'
    echo "$SYSTEM_PROMPT"
    echo
    echo 'USER PROMPT:'
    build_user_prompt "$turn" "$TURNS" "$prev_clean"
  } > "$prompt_file"

  printf '\n======================== TURN %s/%s ========================\n\n' "$turn" "$TURNS"

  cd "$CWD"
  wrapper="$workdir_tmp/run-$turn.sh"
  cat > "$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cat "$prompt_file" | "$LLM_CMD" -b "$BOT_ID"
EOF
  chmod +x "$wrapper"

  # Pass 1: live terminal rendering for the human.
  script -qefc "$wrapper" "$transcript_file"

  # Pass 2: clean canonical text for recursion. Do NOT feed terminal transcript
  # back into the next turn, because Rich redraw/progress output causes loops.
  cat "$prompt_file" | "$LLM_CMD" -b "$BOT_ID" --plain --no-stream > "$clean_file"

  if needs_single_name_repair "$clean_file"; then
    echo
    echo '[repair] Model drifted into multi-name architecture; collapsing back to one product name...'
    repair_to_single_name "$clean_file" "$clean_file.repaired"
    mv "$clean_file.repaired" "$clean_file"
  fi

  {
    echo "## Turn $turn"
    echo
    cat "$clean_file"
    echo
  } >> "$OUTPUT"

  cp "$clean_file" "$prev_clean"
done

echo
printf 'Saved markdown transcript to %s\n' "$OUTPUT"
