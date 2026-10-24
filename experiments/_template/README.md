# YYMMDD-slug

## Hypothesis

What you expect to happen, and why.

## Data

| Field             | Value |
| ----------------- | ----- |
| Dataset `repo_id` |       |
| Revision          |       |
| Episodes used     |       |

## Command

```bash
uv run lerobot-train --config_path experiments/YYMMDD-slug/train.yaml
uv run lerobot-eval --policy.path=<checkpoint> --env.type=nexarm --env.task=stack_bowls --eval.n_episodes=50
```

## Checkpoint

Hub id, step and W&B run link.

## Results

| Policy | Setting  | Episodes | Success rate | Notes |
| ------ | -------- | -------- | ------------ | ----- |
|        | sim      | 50       |              |       |
|        | real arm | 10       |              |       |

## Notes

What you learned and what to try next.
