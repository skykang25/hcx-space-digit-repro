# HyperCLOVAX-SEED-Think-14B Leading-Space Digit Token Sensitivity

This repository documents a reproducible tokenization-form sensitivity observed while reproducing GSM8K-style arithmetic evaluations with `naver-hyperclovax/HyperCLOVAX-SEED-Think-14B`.

The key observation is that visible prompt text can remain unchanged while the exact prompt token IDs affect generation stability for quantities encoded as leading-space digit single tokens, such as `" 1"`, `" 2"`, `" 4"`, and `" 5"`.

## Summary

In TinyGSM8K / GSM8K-style prompts, quantities such as:

```text
Rory orders 2 subs ... 2 bags ... 2 cookies ...
```

were preserved by tokenizer encode/decode and by the API payload, but generation sometimes reconstructed those quantity positions as placeholder-like or special-token-like strings.

Observed examples included:

```text
dog subs
dog bags of chips
dog cookies
to make<|stop|> liters
needs.webdriver kilograms
```

The issue was mitigated by changing only the prompt token IDs for leading-space digit tokens.

Example:

```text
Original tokenization:
" 2" -> [109896]

Rewritten tokenization:
" 2" -> [220, 17]

decode(original) == decode(rewritten) == " 2"
```

This is not a text rewrite such as `"2" -> "two"`. The visible prompt text stays the same.

## What Was Checked

The following checks were performed before narrowing the issue down to tokenization-form sensitivity:

- HF tokenizer encode/decode preserved visible quantities such as `2 subs`, `2 bags`, and `2 cookies`.
- The lm-eval / OpenAI-compatible API payload preserved the same quantities in `messages[0].content`.
- vLLM server-generated `prompt_text` and `prompt_token_ids` matched the HF tokenizer result.
- HF Transformers generation reproduced the same type of failure.
- vLLM offline generation with directly supplied HF `prompt_token_ids` produced the same generated token IDs as HF in the deterministic path check.

This suggests the issue is not simply caused by prompt text corruption, tokenizer decode corruption, the lm-eval payload, or the vLLM OpenAI wrapper.

## Environment

The issue was reproduced in the following setup:

- Model: `naver-hyperclovax/HyperCLOVAX-SEED-Think-14B`
- Evaluation: TinyGSM8K / GSM8K-style arithmetic prompts
- Inference paths tested:
  - Hugging Face Transformers generation
  - vLLM OpenAI-compatible server
  - vLLM offline generation with directly supplied HF `prompt_token_ids`
- vLLM version observed in response metadata: `vllm-0.22.0-tp4-85d53a4e`
- Tensor parallelism: TP=4
- Decoding for deterministic path checks: greedy / `temperature=0`
- Additional lm-eval runs also reproduced the issue under non-greedy settings
- Tokenizer check: HF tokenizer encode/decode preserved the visible prompt text

## Repository Contents

```text
artifacts/
  hcx_leading_space_digit_repro_summary.json  # concise reproduction summary
  space_digit_rewrite_map.json                # decode-equivalent token rewrite map
  symptom_screenshot.png                      # observed generation symptoms

src/
  rewrite_space_digit_tokens.py               # small utility for token-id rewrite experiments
```

## Rewrite Utility

The helper script can rewrite a prompt token ID list using the same decode-equivalent leading-space digit map.

Example:

```bash
python3 src/rewrite_space_digit_tokens.py --ids "109896,101709,110217"
```

Output:

```json
{
  "original_token_ids": [109896, 101709, 110217],
  "rewritten_token_ids": [220, 17, 220, 19, 220, 20],
  "rewrite_events": [...]
}
```

If `transformers` and the model tokenizer are available locally, the script can also check decode equivalence for the map:

```bash
python3 src/rewrite_space_digit_tokens.py \
  --check-map \
  --tokenizer naver-hyperclovax/HyperCLOVAX-SEED-Think-14B
```

## Scope

This repository is an unofficial reproduction note. It does not modify model weights, tokenizer files, or official benchmark numbers.

The goal is to document a reproducible evaluation caveat: arithmetic benchmark results may be sensitive not only to visible prompt text, but also to the exact prompt token IDs used for leading-space digit quantities.

