---
status: proposed
---
# Every configuration knob is declared once

The four risk environment variables stop at the strategy manager's risk module while each strategy re-declares the same defaults twice more and reads sixty keys with `.get(key, literal)`, so a typo yields a default and six adaptive parameters are pinned by tests yet read by nothing. We decided one settings module is the only reader of `os.environ`, and each strategy declares its parameters as a dataclass whose defaults live once and whose constructor rejects unknown keys. No new dependency: standard-library dataclasses. Scripts import settings instead of hand-reading API keys, and the baseline scripts' constants become settings with the baseline values as defaults.
