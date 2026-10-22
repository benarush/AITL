# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.8.0] - 2026-10-23

### Added
- LangChain and Strands: model/tool events now carry `duration_seconds` (wall time of
  the call, 3 decimals) and model events carry `input_tokens` / `output_tokens`
  (`None` if the provider reported none).
- The payload sent to the back-end now includes a top-level `total_duration_seconds`
  (time since the root run started; "so far" for a manual mid-run evaluation).

## [0.7.1] - 2026-10-18

### Fixed
- Strands: every Trellar hook handler is now wrapped in a `guarded` helper, so an
  error inside telemetry is logged and swallowed instead of raising into the host
  application (Strands does not catch hook exceptions itself).

## [0.7.0] - 2026-10-10

### Added
- Strands: registering the Trellar agent on the Graph/Swarm now binds every node's
  Agent automatically, so `hooks=[trellar_agent]` on each Agent is no longer needed
  (still harmless, and still required for a standalone Agent). Registration is
  idempotent: nothing is recorded twice.

### Changed
- `trellar_strands_single_call` and `trellar_langchain_single_call` now default to `ObservabilityMode.ALWAYS` (was `NONE`): a single call cannot be evaluated manually, so with no mode it used to evaluate nothing. Pass `ObservabilityMode.NONE` to keep the old behavior.
- The public factory functions are renamed to product-friendly names:
  `get_agent_guard` -> `trellar_langchain_agent`,
  `get_single_call_guard` -> `trellar_langchain_single_call`,
  `get_strands_guard` -> `trellar_strands_agent`,
  `get_strands_single_call_guard` -> `trellar_strands_single_call`.
  Behavior and the payload sent to the back-end are unchanged.
- Internal callback classes renamed accordingly (`_AgentGuardCallback` ->
  `_LangchainAgentCallback`, `_SingleCallGuardCallback` ->
  `_LangchainSingleCallCallback`, `_StrandsGuardCallback` ->
  `_StrandsAgentCallback`, `_StrandsSingleCallGuardCallback` ->
  `_StrandsSingleCallCallback`). These are not public API.

### Deprecated
- `get_agent_guard`, `get_single_call_guard`, `get_strands_guard` and
  `get_strands_single_call_guard` still work but emit a `DeprecationWarning`
  and will be removed in a future release.

## [0.6.1] - 2026-10-05

### Fixed
- The full conversation history was not sent to the back-end; only the last
  message was. LLM calls now send the earlier Human/AI turns together with the
  current message in `input.human`, in the format:

  ```
  Human: <turn 1>
  AI LLM: <turn 1 response>

  Current message - <last human text>
  ```

  With no prior history, `input.human` is still just the plain last message.
  - LangChain callback: `_extract_llm_input` now folds earlier human/AI turns
    into `human` (also accepts the OpenAI-style `assistant` role). Tool messages
    and tool-call-only AI messages are skipped.
  - Strands callback: added `_llm_human_input`, used instead of `_last_user_text`
    when building the LLM call input.

## [0.6.0] - 2026-10-04

### Added
- Strands agents support (`add_strands_support`).

[Unreleased]: https://github.com/benarush/AITL/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/benarush/AITL/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/benarush/AITL/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/benarush/AITL/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/benarush/AITL/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/benarush/AITL/compare/v0.5.0...v0.6.0
