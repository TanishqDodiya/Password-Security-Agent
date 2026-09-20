# Common Weak Patterns

Predictable structural patterns reduce password security:

- **Repeated characters**: e.g., runs of the same character like extended repetition reduce diversity and make passwords easier to guess.
- **Sequential digits**: e.g., ascending or descending runs like 123 or 321 are common and easily guessed.
- **Sequential letters**: e.g., alphabetic runs like abc or xyz are predictable.
- **Repeated blocks**: e.g., repeated substrings like abab or abcabc indicate low entropy.
- **Low character diversity**: a long password with few unique characters is weaker than a shorter one with high diversity.
- **Single character type**: using only digits or only lowercase limits the search space.
- **Predictable structure**: letters followed only by digits (e.g., word plus numbers) is a common pattern observed in leaked datasets.

Mitigation: Avoid obvious sequences, repetition, and repeated blocks; increase unique character count and mix types.

Source: knowledge/common_patterns.md
