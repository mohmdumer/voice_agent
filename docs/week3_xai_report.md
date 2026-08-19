# Week 3 XAI Interpretation Report

## Summary

We explained the Week 2 winning model (BLIP fine-tuned at lr=5e-5, vision encoder frozen, beam-search
decoding — `config C`) on 14 curated Flickr8k examples (7 with high ROUGE-L against ground truth, 7
with low), using three complementary XAI methods: **Grad-CAM** (gradient-based, vision encoder),
**LIME** (perturbation-based, image superpixels), and **SHAP** (perturbation-based, text prefix).

**Headline finding:** most caption errors are not vision failures. In 5 of 7 "poor" examples, the
model's attention was correctly localized on the right subject — it just picked the wrong word for
the specific action or category. Only one failure mode (decoding degeneration / repetition loops)
showed the model actually disengaging from the image. These need different fixes, and neither is
"train more" in the naive sense.

## Methodology

- **Grad-CAM** (`notebooks/week3_day2_gradcam.ipynb`): BLIP's vision encoder is a ViT, not a CNN, so
  standard Grad-CAM doesn't directly apply. We implemented the standard ViT adaptation by hand: hook
  the last transformer block's patch-token activations `(1, 577, 768)`, backprop from a target
  scalar (whole-caption log-likelihood, or a single word's log-probability), average gradients over
  patches to get per-channel weights, then combine channels to get one importance score per patch —
  the same math as CNN Grad-CAM with "patches" standing in for "pixels" and "hidden dim" for
  "channels". Applied to all 14 examples (whole-caption) plus per-word heatmaps for 2 case studies.
- **LIME** (`notebooks/week3_day3_shap_lime.ipynb`): segments the image into superpixels, perturbs
  which ones are visible, and fits a local linear model to see which regions drive a target word's
  probability. Model-agnostic — a genuine cross-check against Grad-CAM's gradient-based view, not
  just a repeat of it. Applied to 2 examples.
- **SHAP** (same notebook): for an autoregressive decoder, the "input" to a given word isn't only the
  image — it's also every word generated before it. We masked subsets of the preceding words (image
  held fixed) and used SHAP's PartitionExplainer to attribute the target word's probability to each
  preceding word. This isolates *language-model priors* from *visual grounding*. Applied to 3 cases.

## What the model looks at (the "good" cases)

For the 7 high-scoring examples, Grad-CAM concentrates tightly on the actual subject — a jumping
dog, a ball in a dog's mouth, two dogs in snow. One of these (a puppy playing with a plastic bag) was
**miscaptioned as "a small pig" by the zero-shot model in Week 2**; after fine-tuning it's correctly
called "dog", and per-word Grad-CAM shows sharp, word-appropriate attention on the dog's body for
"dog" and the dog+bag region for "white"/"and" — evidence that fine-tuning sharpened genuine
vision-language grounding, not just a memorized word.

## Failure mode taxonomy (the "poor" cases)

Of the 7 low-scoring examples:

**1. Correct localization, wrong description (5 of 7).** Grad-CAM and LIME both center on the right
people/objects, but the model names the wrong specific action or category:
- Cheerleaders mid-toss → "two basketball players are playing basketball." LIME's superpixels for
  "basketball" land on the standing figure — and there's an actual basketball hoop/backboard visible
  in the background. SHAP shows "playing" strongly primes "basketball" as a continuation (SHAP value
  +0.71) independent of the literal repeated word (-0.28 for the earlier "basketball" itself). **Both
  vision (the hoop) and language (the collocation) plausibly point to "basketball"** — this is a
  reasonable misreading of an ambiguous scene, not a broken model.
- A boy carried on someone's back → "a man and a woman are posing for the camera." Attention is on
  the two people; the model gets who's in the shot right, gets the specific pose/action wrong.
- Cheerleaders standing at an intersection → "a group of bicyclists riding down the street", martial
  artists → "doing martial moves" (close, just less specific than ground truth), etc. — same pattern.

**2. Decoding degeneration (1 of 7, but the most severe).** A man in a helmet → "a bearded bearded
bearded..." repeated ~27 times. This is qualitatively different from the cases above:
- Grad-CAM on every repeated "bearded" token shows **nearly identical heatmaps stuck on background
  rock texture**, not the man's face where a beard would be.
- SHAP shows the immediately preceding "bearded" token dominates the next prediction (SHAP value
  0.50, dwarfing every other word in a 10-word prefix).
- Two independent methods, one vision-side and one text-side, agree: **once this loop starts, the
  model stops using the image at all** and just predicts its own last token again. This is a known
  failure mode of near-greedy autoregressive decoding without a repetition constraint — not a
  vision-encoder or fine-tuning problem.

**3. Ambiguous scene (1 of 7).** The remaining case (a group of uniformed men) gets a vague-but-not-
wrong caption ("a group of other men..."), with diffuse attention across the whole scene — consistent
with a genuinely busy image without one clear focal subject.

## Actionable recommendations

1. **Add a repetition constraint at generation time** (`repetition_penalty` or `no_repeat_ngram_size`
   in `model.generate()`). This directly targets failure mode 2, which produced the single worst
   caption in the curated set, and requires no retraining.
2. **Don't over-index on BLEU/ROUGE alone when debugging.** A low score doesn't distinguish "correctly
   looked at the right thing, used the wrong word" (failure mode 1 — often just needs richer/more
   varied training captions) from "stopped looking at the image" (failure mode 2 — a decoding fix).
   The XAI methods here are what separates them.
3. **Failure mode 1 is a data/training problem, not urgent to "fix" architecturally** — the model's
   visual grounding is already working; sharper captions likely come from more fine-tuning
   data/epochs, not a different approach.

## Limitations

- Small curated set (14 images) — enough to find and characterize failure patterns, not to quantify
  their prevalence across the full validation set.
- Our ViT Grad-CAM adaptation is a standard but non-canonical extension of the original CNN method;
  we cross-validated with LIME (different method family) specifically to guard against this being a
  Grad-CAM-specific artifact — the agreement between the two increases confidence.
- SHAP's PartitionExplainer approximates Shapley values (exact computation is intractable for
  realistic prefix lengths); values are directionally reliable, not exact game-theoretic attributions.

## Next steps (Week 4 preview)

Package the pipeline into modular `src/` scripts, formalize the MLflow Model Registry entry for
`config C`, add the repetition-penalty fix identified above, and build the FastAPI service that
returns a caption + Grad-CAM overlay per request.
