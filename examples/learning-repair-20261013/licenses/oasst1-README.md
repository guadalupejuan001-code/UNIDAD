---
license: apache-2.0
dataset_info:
  features:
  - name: message_id
    dtype: string
  - name: parent_id
    dtype: string
  - name: user_id
    dtype: string
  - name: created_date
    dtype: string
  - name: text
    dtype: string
  - name: role
    dtype: string
  - name: lang
    dtype: string
  - name: review_count
    dtype: int32
  - name: review_result
    dtype: bool
  - name: deleted
    dtype: bool
  - name: rank
    dtype: int32
  - name: synthetic
    dtype: bool
  - name: model_name
    dtype: string
  - name: detoxify
    struct:
    - name: toxicity
      dtype: float64
    - name: severe_toxicity
      dtype: float64
    - name: obscene
      dtype: float64
    - name: identity_attack
      dtype: float64
    - name: insult
      dtype: float64
    - name: threat
      dtype: float64
    - name: sexual_explicit
      dtype: float64
  - name: message_tree_id
    dtype: string
  - name: tree_state
    dtype: string
  - name: emojis
    sequence:
    - name: name
      dtype: string
    - name: count
      dtype: int32
  - name: labels
    sequence:
    - name: name
      dtype: string
    - name: value
      dtype: float64
    - name: count
      dtype: int32
  splits:
  - name: train
    num_bytes: 100367999
    num_examples: 84437
  - name: validation
    num_bytes: 5243405
    num_examples: 4401
  download_size: 41596430
  dataset_size: 105611404
language:
- en
- es
- ru
- de
- pl
- th
- vi
- sv
- bn
- da
- he
- it
- fa
- sk
- id
- nb
- el
- nl
- hu
- eu
- zh
- eo
- ja
- ca
- cs
- bg
- fi
- pt
- tr
- ro
- ar
- uk
- gl
- fr
- ko
tags:
- human-feedback
size_categories:
- 100K<n<1M
pretty_name: OpenAssistant Conversations
---

# OpenAssistant Conversations Dataset (OASST1)

## Dataset Description

- **Homepage:** https://www.open-assistant.io/
- **Repository:** https://github.com/LAION-AI/Open-Assistant
- **Paper:** https://arxiv.org/abs/2304.07327

### Dataset Summary

In an effort to democratize research on large-scale alignment, we release OpenAssistant 
Conversations (OASST1), a human-generated, human-annotated assistant-style conversation 
corpus consisting of 161,443 messages in 35 different languages, annotated with 461,292 
quality ratings, resulting in over 10,000 fully annotated conversation trees. The corpus 
is a product of a worldwide crowd-sourcing effort involving over 13,500 volunteers.

Please refer to our [paper](https://arxiv.org/abs/2304.07327) for further details.

### Dataset Structure

This dataset contains message trees. Each message tree has an initial prompt message as the root node, 
which can have multiple child messages as replies, and these child messages can have multiple replies. 

All messages have a role property: this can either be "assistant" or "prompter". The roles in 
conversation threads from prompt to leaf node strictly alternate between "prompter" and "assistant".

This version of the dataset contains data collected on the [open-assistant.io](https://open-assistant.io/) website until April 12 2023.

### JSON Example: Message

For readability, the following JSON examples are shown formatted with indentation on multiple lines.
Objects are stored without indentation (on single lines) in the actual jsonl files.

```json
{
    "message_id": "218440fd-5317-4355-91dc-d001416df62b",
    "parent_id": "13592dfb-a6f9-4748-a92c-32b34e239bb4",
    "user_id": "8e95461f-5e94-4d8b-a2fb-d4717ce973e4",
    "text": "It was the winter of 2035, and artificial intelligence (..)",
    "role": "assistant",
    "lang": "en",
    "review_count": 3,
    "review_result": true,
    "deleted": false,
    "rank": 0,
    "synthetic": true,
    "model_name": "oasst-sft-0_3000,max_new_tokens=400 (..)",
    "labels": {
        "spam": { "value": 0.0, "count": 3 },
        "lang_mismatch": { "value": 0.0, "count": 3 },
        "pii": { "value": 0.0, "count": 3 },
        "not_appropriate": { "value": 0.0, "count": 3 },
        "hate_speech": { "value": 0.0, "count": 3 },
        "sexual_content": { "value": 0.0, "count": 3 },
        "quality": { "value": 0.416, "count": 3 },
        "toxicity": { "value": 0.16, "count": 3 },
        "humor": { "value": 0.0, "count": 3 },
        "creativity": { "value": 0.33, "count": 3 },
        "violence": { "value": 0.16, "count": 3 }
    }
}
```

### JSON Example: Conversation Tree

For readability, only a subset of the message properties is shown here.

```json
{
  "message_tree_id": "14fbb664-a620-45ce-bee4-7c519b16a793",
  "tree_state": "ready_for_export",
  "prompt": {
    "message_id": "14fbb664-a620-45ce-bee4-7c519b16a793",
    "text": "Why can't we divide by 0? (..)",
    "role": "prompter",
    "lang": "en",
    "replies": [
      {
        "message_id": "894d30b6-56b4-4605-a504-89dd15d4d1c8",
        "text": "The reason we cannot divide by zero is because (..)",
        "role": "assistant",
        "lang": "en",
        "replies": [
          // ...
        ]
      },
      {
        "message_id": "84d0913b-0fd9-4508-8ef5-205626a7039d",
        "text": "The reason that the result of a division by zero is (..)",
        "role": "assistant",
        "lang": "en",
        "replies": [
          {
            "message_id": "3352725e-f424-4e3b-a627-b6db831bdbaa",
            "text": "Math is confusing. Like those weird Irrational (..)",
            "role": "prompter",
            "lang": "en",
            "replies": [
              {
                "message_id": "f46207ca-3149-46e9-a466-9163d4ce499c",
                "text": "Irrational numbers are simply numbers (..)",
                "role": "assistant",
                "lang": "en",
                "replies": []
              },
              // ...
            ]
          }
        ]
      }
    ]
  }
}
```

Please refer to [oasst-data](https://github.com/LAION-AI/Open-Assistant/tree/main/oasst-data) for
details about the data structure and Python code to read and write jsonl files containing oasst data objects.

If you would like to explore the dataset yourself you can find a 
[`getting-started`](https://github.com/LAION-AI/Open-Assistant/blob/main/notebooks/openassistant-oasst1/getting-started.ipynb) 
notebook in the `notebooks/openassistant-oasst1` folder of the [LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant)
github repository.


## Main Dataset Files

Conversation data is provided either as nested messages in trees (extension `.trees.jsonl.gz`) 
or as a flat list (table) of messages (extension `.messages.jsonl.gz`).

### Ready For Export Trees

```
2023-04-12_oasst_ready.trees.jsonl.gz       10,364 trees with 88,838 total messages
2023-04-12_oasst_ready.messages.jsonl.gz    88,838 messages
```
Trees in `ready_for_export` state without spam and deleted messages including message labels.
The oasst_ready-trees file usually is sufficient for supervised fine-tuning (SFT) & reward model (RM) training.


### All Trees

```
2023-04-12_oasst_all.trees.jsonl.gz         66,497 trees with 161,443 total messages
2023-04-12_oasst_all.messages.jsonl.gz     161,443 messages
```
All trees, including those in states `prompt_lottery_waiting` (trees that consist of only one message, namely the initial prompt),
`aborted_low_grade` (trees that stopped growing because the messages had low quality), and `halted_by_moderator`.


### Supplemental Exports: Spam & Prompts

```
2023-04-12_oasst_spam.messages.jsonl.gz
```
These are messages which were deleted or have a negative review result (`"review_result": false`).
Besides low quality, a frequent reason for message deletion is a wrong language tag.

```
2023-04-12_oasst_prompts.messages.jsonl.gz
```
These are all the kept initial prompt messages with positive review result (no spam) of trees in `ready_for_export` or `prompt_lottery_waiting` state.

### Using the Huggingface Datasets

While HF datasets is ideal for tabular datasets, it is not a natural fit for nested data structures like the OpenAssistant conversation trees.
Nevertheless, we make all messages which can also be found in the file `2023-04-12_oasst_ready.trees.jsonl.gz` available in parquet as train/validation splits. 
These are directly loadable by [Huggingface Datasets](https://pypi.org/project/datasets/).

To load the oasst1 train & validation splits use:

```python
from datasets import load_dataset
ds = load_dataset("OpenAssistant/oasst1")
train = ds['train']      # len(train)=84437 (95%)
val = ds['validation']   # len(val)=4401 (5%)
```

The messages appear in depth-first order of the message trees.

Full conversation trees can be reconstructed from the flat messages table by using the `parent_id` 
and `message_id` properties to identify the parent-child relationship of messages. The `message_tree_id` 
and `tree_state` properties (only present in flat messages files) can be used to find all messages of a message tree or to select trees by their state.

### Languages

OpenAssistant Conversations incorporates 35 different languages with a distribution of messages as follows:

**Languages with over 1000 messages**
- English: 71956
- Spanish: 43061
- Russian: 9089
- German: 5279
- Chinese: 4962
- French: 4251
- Thai: 3042
- Portuguese (Brazil): 2969
- Catalan: 2260
- Korean: 1553
- Ukrainian: 1352
- Italian: 1320
- Japanese: 1018

<details>
  <summary><b>Languages with under 1000 messages</b></summary>
  <ul>
    <li>Vietnamese: 952</li>
    <li>Basque: 947</li>
    <li>Polish: 886</li>
    <li>Hungarian: 811</li>
    <li>Arabic: 666</li>
    <li>Dutch: 628</li>
    <li>Swedish: 512</li>
    <li>Turkish: 454</li>
    <li>Finnish: 386</li>
    <li>Czech: 372</li>
    <li>Danish: 358</li>
    <li>Galician: 339</li>
    <li>Hebrew: 255</li>
    <li>Romanian: 200</li>
    <li>Norwegian Bokmål: 133</li>
    <li>Indonesian: 115</li>
    <li>Bulgarian: 95</li>
    <li>Bengali: 82</li>
    <li>Persian: 72</li>
    <li>Greek: 66</li>
    <li>Esperanto: 59</li>
    <li>Slovak: 19</li>
  </ul>
</details>

## Contact

- Discord [Open Assistant Discord Server](https://ykilcher.com/open-assistant-discord)
- GitHub: [LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant)
- E-Mail: [open-assistant@laion.ai](mailto:open-assistant@laion.ai)