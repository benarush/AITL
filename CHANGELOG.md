# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/benarush/AITL/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/benarush/AITL/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/benarush/AITL/compare/v0.5.0...v0.6.0
