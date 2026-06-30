# OpenClaw/Hermes Agent Garble Checkpoint: Bounded Bootstrap

Date: 2026-06-30

This checkpoint documents the orchestration-side fix used after the DSpark
runtime was already healthy. The symptom was not a dead vLLM engine: direct
requests were clean, MTP5 was active, and the queue was empty. The failure was
agent-shaped traffic through OpenClaw/Hermes: prompt/tool leakage, session
pollution, retry loops, and occasional weird date/text drift.

## What "bounded bootstrap" means

Bounded bootstrap means the agent still receives startup identity/context, but
OpenClaw does not inject giant workspace files, tool catalogs, memory dumps, or
old tool results unbounded into every turn.

The bad extremes were:

- Too much bootstrap: agents can dump `AGENTS.md`, `TOOLS.md`, skill lists, XML,
  tool schemas, repeated text, Chinese drift, or random fragments.
- Almost no bootstrap: agents stop leaking, but become generic,
  under-contextualized, or weird on normal conversation.

The bounded profile is the middle: enough identity for normal conversation, hard
caps so long prompts do not poison generation.

## OpenClaw supervisor profile that passed

Applied to the DeepSeek supervisors only:

- `donnie`
- `sage`
- `draco`
- `oracle`
- `helios`

```json
{
  "bootstrapMaxChars": 12000,
  "bootstrapTotalMaxChars": 30000,
  "contextInjection": "continuation-skip",
  "experimental": {
    "localModelLean": true
  },
  "contextLimits": {
    "memoryGetMaxChars": 4000,
    "memoryGetDefaultLines": 80,
    "toolResultMaxChars": 12000,
    "postCompactionMaxChars": 1800
  },
  "skillsLimits": {
    "maxSkillsPromptChars": 4000
  }
}
```

Global startup context was bounded:

```json
{
  "agents": {
    "defaults": {
      "startupContext": {
        "enabled": true,
        "applyOn": ["new", "reset"],
        "dailyMemoryDays": 1,
        "maxFileBytes": 8192,
        "maxFileChars": 800,
        "maxTotalChars": 1200
      }
    }
  }
}
```

Global tool search was compacted:

```json
{
  "tools": {
    "toolSearch": {
      "enabled": true,
      "mode": "tools",
      "searchDefaultLimit": 5,
      "maxSearchLimit": 10
    }
  }
}
```

The DeepSeek supervisors kept:

```json
{
  "model": {
    "primary": "dgx-spark-dsv4-cluster/deepseek-v4-flash-spark",
    "fallbacks": []
  }
}
```

## Session cleanup matters

The old session indexes contained stale sessions from earlier configs, including
large bootstrap reports and prompt-leak artifacts. After changing prompt shape,
start fresh sessions or clear the affected supervisor session indexes so the
orchestrator does not keep replaying poisoned history.

Do not restore archived poison sessions into production validation.

## Verification that passed

Sage 15-turn OpenClaw regression:

- 15/15 turns returned.
- Every turn used `deepseek-v4-flash-spark`.
- `fallback=false` on every turn.
- No visible bootstrap/tool dump.
- Time/date/weather/news/session-memory style prompts worked.

Donnie and Draco concurrent OpenClaw regression:

- Donnie: 5/5 clean replies.
- Draco: 5/5 clean replies.
- Both ran at the same time.
- Both used local DeepSeek.
- `fallback=false` for both.

This was tested without switching to `max_num_seqs=1`. The existing MTP5
concurrency runtime profile stayed active.

## Symptoms by bootstrap size

Too large:

- Tool list dumps.
- `AGENTS.md`, `SOUL.md`, `TOOLS.md`, or skill prompt leakage.
- XML/prompt-looking output.
- Chinese drift, repeated characters, random fragments.
- OpenClaw prompt-leak guard retries that look like agent lockups.
- High latency from large prompts.

Too small:

- Generic assistant voice.
- Missing lane/persona.
- `OK`/`ping` style replies.
- Basic prompts pass, real conversation gets weird.
- The agent may hallucinate its role or workflow.

Bounded correctly:

- Agent knows its name and lane.
- Normal conversation works.
- Time/model/weather/news prompts work.
- No hidden prompt/tool dump.
- Fallback stays disabled.
- Prompt sizes stay controlled.

## Important note

This is an orchestration fix, not a replacement for the DSpark runtime patches.
Keep NVFP4 KV, MTP5, and Keys Patch 2b in the runtime. Bounded bootstrap fixes
the OpenClaw/Hermes prompt/session poison path that can remain even after direct
vLLM tests are clean.
