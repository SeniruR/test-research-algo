# Sound2Hap (unmodified upstream copy)

These files are byte-identical copies from
[Iris1215/Sound2Hap](https://github.com/Iris1215/Sound2Hap) at commit
`0af5934173ebd9cb35eed2d63fbdb79ac75a75d8`, distributed under its MIT `LICENSE`.

| Local file | Upstream path | Used by |
|---|---|---|
| `Percept.py` | `Signal_Processing_Algorithms/Percept.py` | generator `a_perception_mapping` |
| `FreqShift.py` | `Signal_Processing_Algorithms/FreqShift.py` | generator `b_frequency_shifting` |
| `Pitch_WebTool.py` | `Signal_Processing_Algorithms/Pitch_WebTool.py` | generator `c_pitch_matching` |
| `HapticGen.py` | `Signal_Processing_Algorithms/HapticGen.py` | generator `d_haptic_gen` |
| `utils/normalization.py` | `Signal_Processing_Algorithms/utils/normalization.py` | imported by the scripts above |

Do not edit them. Anything this project needs on top (output folders, quiet
output, a common call signature) lives in the generator components under
`haptic_gt/components/generators/`.

## Verifying

```bash
cd third_party/sound2hap && sha256sum -c SHA256SUMS      # local copies unchanged
python scripts/verify_sound2hap.py --upstream            # same bytes as GitHub
```

Or replace the `.py` files with ones you download yourself from the commit above
and re-run the pipeline; the outputs must not change.
