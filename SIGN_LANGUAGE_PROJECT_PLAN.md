# Sign Language Project Plan - A Beginner's Guide

Based on your **SLPlan.pdf**.

This plan has **five phases**. Each phase finishes one complete part of the project. The smaller numbered tasks explain what to do, why it matters, and what you should have when you finish.

Read the introduction first. Then work through one phase at a time. You do not need to understand all the advanced details before starting. Technical settings are collected in the **Coding notes** near the end.

This is a plan for future implementation. The models and experiments described here have not been built or tested as part of writing this file.

## First, what are you trying to build?

Imagine watching a video of someone signing several signs in a row. You want a computer to watch the video and write down the names of the signs in the correct order.

```text
Input: A video containing three signs.

Correct labels:  TOMORROW WEATHER GOOD
Computer output: TOMORROW WEATHER GOOD
```

These labels are called **glosses**. A gloss is a name used to identify a sign in the dataset. The example uses English words to explain the idea. Your PDF uses a German Sign Language dataset, with labels such as `MORGEN`, `WETTER`, and `GUT`.

Recognizing these labels is the first task. Turning them into a natural sentence, such as "The weather will be good tomorrow," would be a separate translation task.

### What makes your project special?

Normally, a model learns a sign by seeing training examples of it.

Your project asks a harder question:

> Can the model recognize a sign that was left out of its sign-training videos if we give it information from the sign's text label?

For example, imagine training with some weather signs while excluding the sign labeled `SNOW`. Later, you add `SNOW` to the possible answers using a numerical representation of its text.

The experiment checks whether the model can then recognize that sign. **This is an idea to test, not a guaranteed ability.** Knowing that rain and snow are related words does not tell a computer exactly how their signs look.

The PDF starts with **PHOENIX-2014-T**, a German Sign Language dataset. A dataset is a collection of examples and correct answers. If you later want Indian Sign Language, you will need suitable ISL data and a separate evaluation.

## The five phases at a glance

| Phase | What you will do | What will be finished |
|---|---|---|
| **1. Get ready** | Set up tools, organize videos, and decide fair testing rules. | A prepared project with usable data. |
| **2. Build the first working version** | Run an existing model and build a comparison model. | A working system that predicts sign labels. |
| **3. Add your main idea** | Use text information when choosing sign labels. | Text-based versions that can be compared fairly. |
| **4. Test signs left out of training** | Train without selected signs, then add their text labels. | Evidence showing whether the main idea works. |
| **5. Finish and explain the project** | Study mistakes, organize results, and prepare the final package. | A report and a project someone else can run. |

The alternatives below are choices, not a list of extra things you must all build. Start with the recommended simple route.

## A few useful words

| Word | Easy meaning |
|---|---|
| Model | A computer system that learns patterns from examples. |
| Training | Letting the model learn from examples and their correct answers. |
| Prediction | The model's answer. |
| Gloss | A dataset label naming a sign. |
| Vocabulary | The list of labels the model can choose from. |
| Checkpoint | A saved copy of a model's learned state. |
| Baseline | A comparison model used to judge whether a change helps. |
| Frozen | Kept unchanged while other parts learn. |
| Unseen sign | A sign excluded from the relevant sign-training data in an experiment. |

Other terms are explained when you need them.

---

## Phase 1 - Get the tools, data, and testing rules ready

**Goal:** Prepare the project before spending time training models.

**Finished result:** You can read the videos and labels correctly, and your experiment rules are written down.

### 1.1 Write the project goal in your own words

**Do this:** Write three short answers: What goes in? What comes out? What are we testing?

Your answers should be: a signing video goes in, ordered sign labels come out, and we test whether text information helps recognize signs excluded from training.

**Why:** This keeps new features connected to the actual research question.

**Alternative:** A live camera application can come later. Saved videos are easier to repeat and debug because the input stays the same.

**Done when:** You have a short description of the project and its expected output.

### 1.2 Set up a separate software environment

**Do this:** Create a project folder and a separate Python environment. An environment keeps the software packages for this project together. Obtain the CorrNet code and follow its setup instructions.

**CorrNet** is the existing sign-recognition model used as the starting point in your PDF.

**Why:** Separate packages make installation problems easier to fix without affecting other work.

**Alternative:** Some reference tools need Linux. A Linux machine or suitable WSL2 setup on Windows can help. Native Windows with a simpler decoder can be useful for early checks. See the [CorrNet setup instructions](https://github.com/hulianyuyy/CorrNet).

**Done when:** The code loads and a small model check runs successfully.

### 1.3 Check what your computer can handle

**Do this:** Record storage, RAM, and GPU memory. A GPU is a processor often used to speed up model training. Try a small batch and measure memory and time. A **batch** is a group of examples processed together.

**Why:** Long videos can need much more memory than short ones. Verify your actual machine; the PDF assumes an 8 GB GPU.

**Alternative:** Use smaller batches or saved video features. Consider a larger GPU only after measuring the need.

**Done when:** You know a workload that runs and roughly how long it takes.

### 1.4 Organize videos and their correct labels

**Do this:** Obtain PHOENIX-2014-T through its official distribution. Preserve its provided training, development, and test groups. Check several videos against their gloss sequences. Use gloss annotations, rather than the German sentence translations, for this project.

Think of the groups like schoolwork:

| Group | Purpose |
|---|---|
| Training | Lessons the model learns from. |
| Development, or dev | Practice checks used to choose settings. |
| Test | A final exam after choices are fixed. |

**Alternative:** A tiny subset is useful for debugging, but cannot replace the full experiment.

**Done when:** A table connects each video to its location, group, length, and correct labels.

### 1.5 Give labels consistent numbers

**Do this:** Create or reuse the label dictionary expected by your model. It connects labels to numbers.

```text
Illustration only:
1 = MORGEN
2 = WETTER
3 = GUT
```

**Why:** Number 2 must mean the same label in the data, model, and evaluation. The real IDs must match the checkpoint. Reserve the model's special blank ID, explained in Phase 2.

**Alternative:** A later experiment can use a smaller dictionary. Save the conversion between old and new IDs if you do this.

**Done when:** All project components use the correct label mapping.

### 1.6 Choose which signs to exclude from training

**Do this:** Select a group of labels for the unseen-sign experiment. The PDF suggests starting with about 10% of the vocabulary. Remove every training video containing any selected label from that experiment's training list.

```text
Held-out label: SNOW
Video labels: TOMORROW SNOW
Action: Exclude the whole video from this training list.
```

**Why:** Removing only the word would still expose the model to the sign and give the video an incorrect answer.

**Alternative:** Use a smaller held-out group if too little data remains. Decide from data counts before looking at recognition results. Check whether other labels also lose all their examples; those labels are no longer seen either.

**Done when:** The excluded labels, remaining videos, remaining seen vocabulary, and selection method are saved.

### 1.7 Keep the final evaluation separate

**Do this:** Write down which examples may influence training and which may help choose settings. For a simple strict approach, choose settings using dev videos containing only seen labels. Keep final unseen-sign performance for evaluation after those choices are fixed.

**Why:** Repeatedly redesigning the model after seeing final-test mistakes is like practicing with the exam answers.

**Alternative:** Create a separate group of development-only unseen labels. That helps tune for unfamiliar signs, but needs careful separation and leaves less training data.

The model may receive candidate label names in this project. That is different from training on final evaluation videos.

**Done when:** Your notes clearly say which data may influence each decision.

### 1.8 Start an experiment notebook

**Do this:** Record each run's model version, data list, settings, saved model, and results. Also record the **seed**, a number that controls random starting choices.

**Why:** After many runs, it is easy to forget which change produced which score. Different seeds also help check whether a result is consistent.

**Recommended start:** A simple table is enough:

| Run | Version | What changed? | Result | Saved model |
|---|---|---|---|---|
| Example only | A | First trial | Not measured | Not created |

**Alternative:** Add automatic logs and saved configuration files as the project grows.

**Done when:** Every future experiment has a clear place to record its information.

### Phase 1 is complete when

- [ ] A small model check runs.
- [ ] Videos and correct labels are organized.
- [ ] Label IDs are consistent.
- [ ] Held-out labels and reduced training lists are saved.
- [ ] Development and final-test rules are written down.
- [ ] An experiment notebook is ready.

**You now have:** A prepared project and a fair experiment design.

---

## Phase 2 - Build the first working recognition system

**Goal:** Establish a trustworthy starting point before adding your idea.

**Finished result:** A working comparison model and saved video information that later versions can reuse.

### 2.1 Run the existing CorrNet model

**Do this:** Load the PHOENIX-2014-T checkpoint with its matching data preparation and label dictionary. Start by checking development predictions.

**Why:** If the original model fails, the problem may be setup rather than your new method.

The PDF gives reference error rates of about 18.9% on dev and 20.5% on test. These are reference numbers, not your measured results. They also appear in the [CorrNet results table](https://github.com/hulianyuyy/CorrNet#inference).

**Alternative:** Train the reference yourself if necessary, but record that it is a new run and allow more time.

**Done when:** Predictions are reasonable and large differences from the reference have been investigated.

### 2.2 Understand the model's three main parts

**Do this:** Identify these components:

```text
Video
  -> Video reader: turns images into useful numbers
  -> Sequence reader: studies changes over time
  -> Answer chooser: gives possible labels scores
  -> Predicted sign-label sequence
```

Their technical names are **video encoder**, **temporal model**, and **output head**.

**Why:** Your main idea changes the answer chooser. Keeping the other parts comparable helps show whether that change helped.

**Alternative:** Redesigning everything at once would make the cause of improvements or failures harder to identify.

**Done when:** You can explain what each component receives and produces.

### 2.3 Save the useful numbers from each video

**Do this:** Run the video-reading part and save its output. These numbers are called **features**. Their saved collection is a **feature cache**.

The PDF proposes saving CorrNet's `TemporalConv` output before its `BiLSTM`. You will need those names when finding the extraction point in code.

**Why:** Later models can reuse features instead of processing every video again.

**Recommended start:** Save each video's ID, labels, valid feature length, and source checkpoint. Keep extraction settings fixed.

**Alternative:** Process videos during every training round. This allows more varied preprocessing but costs more computation.

**Done when:** A small set of correctly labeled feature files exists.

### 2.4 Check that saving features preserves predictions

**Do this:** Compare these routes for several videos:

```text
Route 1: Video -> original complete model -> prediction
Route 2: Saved features -> original remaining parts -> prediction
```

They should agree under the same settings, allowing for explicitly checked numerical differences.

**Why:** This catches mixed video IDs, wrong extraction points, and incorrect lengths before they affect all experiments.

**Alternative:** Smaller-number formats can reduce storage. Use them only after checking that predictions remain sufficiently close.

**Done when:** Both routes agree and you can confidently extract the remaining features.

### 2.5 Build the shared sequence reader

**Do this:** Build a component that reads saved features in time order. Use the same design for all answer choosers.

**Why:** Signs involve movement; one image may not provide enough information.

The PDF combines layers that study nearby time steps with a **BiLSTM**, which uses earlier and later sequence context. Exact sizes are in the coding notes.

**Recommended start:** Preserve time order and true lengths. If padding makes examples equal length for batching, ensure the model handles it correctly.

**Alternative:** Use a smaller version for debugging, keeping it identical across comparisons.

**Done when:** The sequence reader produces a feature at each valid output time step.

### 2.6 Add the first answer chooser: version A

**Do this:** Add a standard learned classifier, called **head A**. Train it using **CTC**, a method that learns label order without requiring exact start/end times for every sign.

CTC uses a blank symbol. When decoding, merge neighboring repeated labels first, then remove blanks:

```text
A A blank B B -> A B
A blank A    -> A A
```

Blank means "do not emit a new label here." It does not necessarily mean the signer paused.

**Alternative:** Exact sign timestamps could provide more detailed supervision, but they are not assumed here.

**Done when:** A calculates training loss and produces label sequences, including repeated signs correctly.

### 2.7 Make the model learn a few examples first

**Do this:** Train on a handful of videos and see whether it can learn their correct sequences. Inspect predictions as well as the training loss. A **loss** is the signal used to guide learning.

**Why:** A small learning check finds basic problems before a costly full run.

Start with **greedy decoding**: choose the highest score at each time step and apply the CTC rules.

**Alternative:** **Beam search** considers multiple possible sequences. It can help later, but adds complexity during initial debugging.

**Done when:** The small set is predicted closely and outputs are not just blanks.

### 2.8 Train A properly and measure its mistakes

**Do this:** Train your custom sequence reader and A on the larger training feature collection. Choose the saved model using development results.

Measure **Word Error Rate (WER)**. Here it counts sign-label mistakes: wrong labels, missing labels, and extra labels. For example, 20 errors over 100 reference labels gives 20% WER. Lower is better; many extra labels can push it above 100%.

**Why:** A becomes the baseline for B and C.

**Alternative:** Keep the original CorrNet result too, but in a separate row because your custom sequence reader may differ.

**Done when:** A's model, predictions, settings, and WER are saved.

### Phase 2 is complete when

- [ ] The original reference has been checked.
- [ ] Saved features preserve its predictions.
- [ ] The shared sequence reader works.
- [ ] Blank and repeated labels decode correctly.
- [ ] A learns the small set and trains on the larger set.
- [ ] Baseline results are saved.

**You now have:** A complete first recognition system.

---

## Phase 3 - Add text information and compare it fairly

**Goal:** Implement the new idea, first checking signs present in training.

**Finished result:** Text-based models and evidence about whether the text's meaning helps.

### 3.1 Prepare a text version of each label

**Do this:** Keep original labels and IDs unchanged. Make a separate text version for the word-processing tool.

```text
Original label: WETTER
Text version:   wetter
Original ID:    unchanged
```

**Why:** Text tools may work better with ordinary lowercase words, while the dataset still needs its original labels.

**Recommended start:** Lowercase ordinary words and inspect unusual labels. Do not blindly remove prefixes that may change meaning.

**Alternative:** A short, reviewed description can replace an unusual text input in a separate experiment.

**Done when:** Every label has documented text, with accidental duplicates flagged.

### 3.2 Turn text into lists of numbers

**Do this:** Use German **fastText**, as proposed in the PDF, to generate a **text embedding** for each label. An embedding is a list of numbers representing a word. Words used in related ways may have similar representations.

**Why:** The model needs numerical information to compare video and text.

**Recommended start:** Save embeddings in label-ID order. Use the binary model when subword generation for unfamiliar strings is needed. See the [fastText format explanation](https://fasttext.cc/docs/en/crawl-vectors.html).

**Alternative:** Try a phrase encoder later if descriptions are useful.

**Done when:** Embeddings match their labels, and zero or duplicate representations have been checked.

### 3.3 Handle blank and special markers separately

**Do this:** Give blank its own learned representation. Check whether unusual annotation markers must remain as target labels. If they do, give them separate representations too.

**Why:** Blank is a CTC processing symbol, not an ordinary word with a fastText meaning.

**Alternative:** Remove a marker only if the verified dataset protocol permits it, and do so consistently in targets and scoring. Do not replace retained target labels with CTC blank inside the correct sequence.

**Done when:** Ordinary labels, retained markers, and blank each have clear rules and the correct IDs.

### 3.4 Build B: compare video information with text information

**Do this:** Train a small transformation that makes video features comparable with text embeddings.

```text
Video features -> video embedding
Label text     -> text embedding
Compare them   -> score for that label
```

The PDF uses **cosine similarity**, which compares the direction of the numerical representations instead of simply favoring larger numbers.

**Why:** B uses text-based reference representations to choose answers.

**Recommended start:** Keep ordinary text embeddings fixed and let the video transformation learn.

**Alternative:** Compare fixed and learned score sharpness, called temperature, if training is unstable. Starting values are in the coding notes.

**Done when:** B trains and decodes through the same basic pipeline as A.

### 3.5 Build C: adjust all text representations with one shared rule

**Do this:** Add one small learned transformation applied to all text embeddings. Compare video information with these adjusted text representations.

**Why:** Similar word meanings do not always correspond to similar-looking signs. The adjustment gives the model some freedom to improve the comparison.

**Recommended start:** Use a simple transformation shared by all labels. New labels must be able to use the same rule later.

**Alternative:** A larger transformation is more flexible but may memorize training labels and generalize poorly. Test it separately.

**Done when:** C trains and can process a new text embedding using its existing shared transformation.

### 3.6 Compare A, B, and C under similar conditions

**Do this:** Start with A versus B using one seed. Once both work, add C. Use matching video features, sequence-reader design, data, training opportunity, and evaluation.

| Version | How it chooses answers |
|---|---|
| A | Learns classifiers for known labels. |
| B | Compares with fixed text embeddings. |
| C | Compares with text embeddings adjusted by a shared rule. |

**Why:** A carefully tuned B versus an undertrained A is not a fair comparison.

**Alternative:** If versions need different settings, give them comparable tuning opportunities and record the differences.

**Done when:** You have a fair initial comparison table.

### 3.7 Check whether text meaning actually helps

**Do this:** Run **controls**: experiments that deliberately change one component.

| Control | What changes? | What it checks |
|---|---|---|
| A-norm | Normalizes A's learned class weights. | Whether normalization explains some differences. |
| B-rand | Uses fixed random number lists. | Whether arbitrary label codes work just as well. |
| B-shuf | Assigns text embeddings to the wrong labels using a saved shuffle. | Whether the correct label-to-meaning pairing matters. |

**Why:** Strong known-sign performance could come from memorizing arbitrary codes, rather than using meaning.

**Alternative:** Delay these while debugging, but include them before claiming that text meaning explains an improvement.

**Done when:** B has been compared with random and incorrectly assigned text information.

### 3.8 Fix basic problems before adding complexity

**Do this:** If B or C is weak, check:

1. Label IDs match embedding rows.
2. Video lengths and padding are correct.
3. Text cleaning did not make different labels identical.
4. Outputs are not blank almost everywhere.
5. Learning speed and score sharpness are reasonable.

**Why:** A bigger model does not repair incorrect labels. A shared transformation cannot separate two exactly identical text inputs either.

**Alternative:** Initialize the sequence reader from trained A as a separately labeled experiment. For Phase 4, that A must also have used only allowed training videos.

**Done when:** Problems are fixed or documented with evidence.

### 3.9 Decide what the known-sign results show

**Do this:** Repeat important comparisons with three seeds when feasible. Report the average and variation.

The PDF suggests B or C being within about 1-2 absolute WER percentage points of A is a useful reason to continue. For example, 20% and 22% differ by 2 points. If both remain more than about 3 points worse, investigate before scaling up.

**Why:** This checks whether the method is usable before the harder unseen-sign experiment.

**Alternative:** A clear negative result is valid research evidence. Record why the method failed rather than calling it successful.

**Done when:** You have a report and a reasoned next-step decision.

### Phase 3 is complete when

- [ ] Text mappings and embeddings are checked.
- [ ] B and C train and decode correctly.
- [ ] A, B, and C have been compared fairly.
- [ ] Random and shuffled controls have been investigated.
- [ ] Important comparisons include repeated runs where feasible.
- [ ] Findings and remaining problems are documented.

**You now have:** The text-based idea implemented and checked on known signs.

---

## Phase 4 - Test signs left out of training

**Goal:** Answer the main research question honestly.

**Finished result:** Evidence about recognizing held-out signs and the effect on familiar-sign recognition.

### 4.1 Check what the starting model has already learned

**Do this:** Check the training history of every model component and feature cache used in this experiment.

**Why:** The released CorrNet checkpoint used earlier may already have learned the selected held-out signs. Freezing it does not make it forget them.

**Recommended start:** Follow the PDF's clean approach: begin with generic image pretraining, then train the sign model only on the reduced list from Phase 1.

**Alternative:** Generic video pretraining can be tested separately. Document its data source and disclose uncertain overlap.

**Done when:** A written record shows what training data each relevant component used.

### 4.2 Train a fresh video reader on allowed videos

**Do this:** Train the new CorrNet model using only the reduced training list. Choose checkpoints according to Phase 1's rules.

**Why:** This avoids training examples of the selected held-out signs in the project dataset.

**Recommended start:** Check a small training run, saving, and reloading before starting a long job.

**Alternatives:** A different video model, such as S3D, can be another experiment. Frozen generic features are a cheaper comparison but may represent sign details less effectively.

Do not use the original full-data sign checkpoint as a shortcut here.

**Done when:** The clean model, allowed data list, and development results are saved.

### 4.3 Save a new collection of clean features

**Do this:** Repeat Phase 2's feature extraction with the clean model. Keep the files separate from the earlier collection and record the source checkpoint.

**Why:** Accidentally using the old files would bring information from the old model into the clean experiment.

**Recommended start:** Use an obvious folder name, such as `clean_reduced_split_features`. Only allowed videos belong in its training list.

**Alternative:** Process videos during training instead of caching. This costs more computation and needs equally controlled preprocessing across comparisons.

**Done when:** The clean cache passes the original consistency checks and its training list contains no excluded videos.

### 4.4 Train new A, B, and C models on clean data

**Do this:** Train fresh sequence readers and answer choosers using clean features. Include random/shuffled comparisons where feasible.

**Recommended start:** During training, allow only seen labels, required special markers, and blank as answers. Save held-out labels for the vocabulary-expansion step.

**Why:** This makes the experiment easy to interpret: train first, then add new possible answers without more learning.

**Alternative:** Including held-out text vectors as wrong-answer candidates during training changes the protocol. If tested, report it separately and document what information the loss saw.

**Done when:** Clean models and exact training candidate lists are saved.

### 4.5 Add held-out labels without more training

**Do this:** Stop model updates. Generate text embeddings for held-out labels and add them to B/C's possible answers. For C, apply the shared transformation it already learned.

For illustration, an answer list could expand from `RAIN, WIND, SUN` to `RAIN, WIND, SUN, SNOW`.

**Why:** This is the central vocabulary-expansion step.

**Alternative:** Do not select a tiny answer list for each video from its correct labels. That reveals information about the answer. Use a predefined shared candidate list.

**Done when:** The decoder can emit the new IDs and records confirm no model weights were updated.

### 4.6 Measure correct recognition and wrong guesses

**Do this:** Evaluate with seen and held-out labels available together.

| Measure | What it tells you |
|---|---|
| Overall WER | Sequence mistakes across the evaluation videos. |
| WER on videos containing held-out signs | Performance on sentences with unfamiliar signs. |
| WER on videos containing only seen signs | Whether extra answer choices hurt familiar videos. |
| Unseen recall | How many true held-out sign occurrences were recognized. |
| Unseen precision | How many predicted held-out labels were correct. |

For example, finding 6 of 10 true occurrences gives 60% recall. If the model predicted that label 12 times and only 6 were correct, precision is 50%.

Use consistent alignment of full correct and predicted sequences to count matches. Include video and label counts, not just percentages.

**Alternative:** A seen-only classifier has zero unseen recall because it cannot emit those IDs. Beating that zero alone is weak evidence; compare with random/shuffled controls and check false guesses too.

**Done when:** All these results and the evaluation rules are recorded.

### 4.7 Try a different text representation if useful

**Do this:** After fastText is evaluated, choose one additional option from the PDF if it answers a clear question.

| Option | Reason to try it | What to check |
|---|---|---|
| A phrase encoder, such as LaBSE | Represents short descriptions. | Word meaning may still miss sign appearance. |
| mBART-derived representations | Provides another multilingual text representation. | Specify how its numbers are extracted. |
| SignCLIP text representations | May benefit from sign/text training. | Audit possible earlier exposure to relevant signs and report it separately if needed. |
| Reviewed sign descriptions | Can describe handshape, location, and motion. | Use reliable independent descriptions, not guesses or descriptions derived from final test videos. |

**Recommended start:** Keep everything else fixed when changing text representation.

**Alternative:** Complete fastText alone if resources are limited. Every proposed model need not be implemented.

**Done when:** Added comparisons have a purpose, documented inputs, and an exposure record.

### 4.8 Answer the research question in ordinary language

**Do this:** Write the conclusion supported by the measurements. Possible conclusions include:

- New labels could be added, but were rarely recognized correctly.
- Text helped on some tested groups but not others.
- Recall increased, but too many false guesses appeared.
- Improvement held across repeated runs with acceptable seen-sign performance.

These are example conclusions, not results already obtained.

**Why:** Accepting a new label and recognizing it reliably are different achievements.

**Alternative:** If results are poor, finish a failure analysis. Do not replace the held-out set just to obtain a better score.

**Done when:** Your conclusion includes supporting evidence and limitations.

### Phase 4 is complete when

- [ ] Model and feature training histories are checked.
- [ ] Held-out signs are excluded from relevant sign-training videos.
- [ ] Clean models and clean features are used.
- [ ] New labels are added without model updates.
- [ ] Correct recognition and false guesses are measured.
- [ ] The conclusion matches the evidence.

**You now have:** The main research result, whether positive, mixed, or negative.

---

## Phase 5 - Finish the results, report, and project package

**Goal:** Make the work understandable and repeatable.

**Finished result:** A documented project, final evidence, and clearly identified optional work.

### 5.1 Fix your choices before the final test

**Do this:** Choose models, preprocessing, settings, answer lists, and decoding using allowed development information. Then run the final test evaluation.

**Why:** The final test should assess the selected method rather than guide repeated redesign.

**Recommended start:** Repeat important runs with three seeds if possible. State which parts were retrained.

**Alternative:** With one trained video reader, you can repeat head training. Explain that this measures variation in the heads, not variation across complete model training runs.

**Done when:** Each final result is linked to fixed, documented choices.

### 5.2 Explain mistakes with examples

**Do this:** Inspect correct predictions, missing labels, extra labels, confused signs, and repeated-sign errors.

```text
Correct:   TOMORROW WEATHER GOOD
Predicted: TOMORROW GOOD
Problem:   WEATHER was missed.
```

**Why:** One total score cannot explain what the model finds difficult.

**Alternative:** A table of frequently confused label pairs helps when there are many examples. Time-based score plots can help too, but a prediction peak is not proof of an exact sign boundary.

**Done when:** The main error patterns are explained with real examples from your experiment.

### 5.3 Consider the spiking-model extension later

**Do this:** The PDF proposes replacing the video reader with `SpikingVideoNet3D` after the main study. A spiking network uses discrete activity events called spikes.

First inspect its implementation, input requirements, and output features. Keep the data split and evaluation rules comparable. Measure recognition and speed, and report spike activity or energy estimates with their assumptions.

**Why:** This explores a separate question about computation and the video reader.

**Alternative:** Defer it if the main study is unstable or the implementation is unavailable. Fewer spikes alone do not prove lower measured energy use.

**Done when:** If attempted, it has its own fair comparison report; otherwise, its deferred status is explicit.

### 5.4 Consider the optional whole-video summary experiment

**Do this:** The PDF's "superposition" analysis asks whether one combined numerical summary can retain information about several signs.

For example, it might retain that `TOMORROW`, `WEATHER`, and `GOOD` appeared, while losing their order or repeated occurrences. Test recovery at different sentence lengths and summary sizes.

**Why:** This studies stored information but cannot replace ordered sequence recognition.

**Alternative:** Other extra training tasks, such as matching segments to text, can be tested after basic training is stable. Add one at a time so its effect is clear.

**Done when:** The optional result is separately reported or explicitly deferred.

### 5.5 Make final result tables

**Do this:** Separate known-sign comparisons from the clean unseen-sign experiment.

| Version | Dev WER | Test WER | Runs | Notes |
|---|---|---|---|---|
| A | To measure | To measure | To record | Standard chooser |
| B | To measure | To measure | To record | Fixed text information |
| C | To measure | To measure | To record | Shared text adjustment |
| Controls | To measure | To measure | To record | Random/shuffled information |

For unseen signs, add recall, precision, seen-only sentence WER, data counts, and training history. Record model size and decoder settings too.

**Why:** Exposed and clean models must not appear as equivalent unseen-sign experiments.

**Alternative:** Start in a spreadsheet; generate tables automatically from saved results later.

**Done when:** Every measured value can be traced to a model, settings, data, and predictions.

### 5.6 Organize the project so it can run again

**Do this:** Keep code, settings, data lists, label dictionaries, model locations, and results organized. A suggested future structure is:

```text
Sign_Language/
  SIGN_LANGUAGE_PROJECT_PLAN.md
  README.md          Setup and running instructions
  configs/           Model and training settings
  data_lists/        Videos used in each experiment
  dictionaries/      Label IDs and text mappings
  src/               Program code
  results/           Scores and predictions
  reports/           Explanations and figures
```

**Why:** Someone should be able to repeat a result, not just read that it worked once.

**Alternative:** Notebooks are useful for exploration. Reusable scripts are easier for repeated full runs. Document large-file storage locations separately.

**Done when:** A fresh run can load a saved model and reproduce a documented example.

### 5.7 Write the report and prepare a small demonstration

**Do this:** Explain the problem, differences between A/B/C, data rules, results, mistakes, and next questions. Use the PDF's references as a reading starting point and check current work before claiming novelty.

**Recommended demonstration:** Load a saved video and show its predicted gloss sequence. Explain the tested limits of adding new labels.

**Why:** A clear report shows what you actually achieved and how to assess it.

**Alternative:** A command-line example is sufficient. Add an interface for presentation only if useful.

**Done when:** The report and demo explain measured behavior without claiming untested translation or universal sign recognition.

### Phase 5 is complete when

- [ ] Final results follow the declared rules.
- [ ] Mistakes and limitations are explained.
- [ ] Optional work is reported or clearly deferred.
- [ ] Code, settings, data lists, and results are organized.
- [ ] A model can be reloaded and demonstrated.
- [ ] The report explains what was actually achieved.

**You now have:** A completed core research project. Optional extensions remain unfinished unless they were actually implemented and evaluated.

---

## What should you learn along the way?

You do not need to master all of machine learning first. Learn each topic when it becomes useful.

| Topic | Why you need it | When |
|---|---|---|
| Basic Python: variables, lists, loops, functions, files | Read data and write small programs. | Before coding Phase 1. |
| Arrays and tensors | Understand numerical data and its shape. | Phases 1-2. |
| Basic PyTorch: models, loss, optimizer, saving/loading | Train and reuse models. | Phase 2. |
| Training/dev/test separation | Make results trustworthy. | Phase 1 onward. |
| CTC and decoding | Turn video outputs into label sequences. | Phase 2. |
| Text embeddings and similarity | Understand B and C. | Phase 3. |
| Fair comparisons and repeated runs | Check whether changes really help. | Phases 3-5. |

A **tensor** is an array of numbers, possibly with several dimensions. An **optimizer** updates the model's learned numbers using the training loss.

## Your first session: do only these small tasks

1. Write the one-paragraph goal from task 1.1.
2. Check Python and create the project environment.
3. Locate the dataset instructions and check access.
4. Open one video and find its correct sign-label sequence.
5. Record what you understood and what failed.

Your first useful result is **one video connected to its correct labels**. Then build toward one prediction, a small training run, and finally complete comparisons.

## A flexible schedule

These are rough allowances after you have the needed basic skills. Learning time, dataset access, installation problems, and hardware limits can add substantial time. Update estimates after measuring actual runs.

| Work | Initial allowance |
|---|---|
| Phase 1: setup and data rules | 2-4 working days, plus access/download time |
| Phase 2: reference and A | 2-5 working days |
| Phase 3: B, C, and controls | 3-7 working days |
| Phase 4: clean unseen-sign study | 2-4 weeks or more |
| Phase 5: core analysis and report | 3-7 working days |

Estimate spiking and other optional extensions separately after inspecting their implementations.

With one GPU, follow the phases in order initially. During long training jobs, work on documentation, text processing, or evaluation code. With separate compute, the clean video-reader training can start earlier once the split and training code are verified. Keep its models and features separate from the ordinary reference experiments.

## Coding notes - return here when you start implementation

This section preserves the PDF's useful technical details. You can skip it during your first reading. These are starting choices, not guaranteed best settings.

### Video feature extraction

The intended extraction point is CorrNet's `TemporalConv` output before its BiLSTM. The per-video shape is approximately `T' x 1024`: `T'` time positions, each containing 1024 numbers.

Use actual output lengths from the model; do not assume they are exactly one quarter of the input. Save valid positions only. Inspect the checked-out code before applying the PDF's suggested feature-saving patch.

Use evaluation mode, disable gradient tracking, and keep preprocessing fixed. Record the source checkpoint for every cache. If the checkpoint or preprocessing changes, regenerate the affected features.

### Shared sequence reader

The PDF proposes:

```text
Saved features: 1024 numbers per time position
  -> Linear layer: 1024 to 512
  -> Layer normalization, ReLU, dropout 0.1
  -> Two residual temporal convolution blocks
  -> Two-layer BiLSTM: 256 hidden units per direction
  -> 512 numbers per time position
  -> A, B, or C
```

These layers transform the numerical sequence. Preserve the time-step count in this custom reader. Check padding behavior, especially for the bidirectional recurrent model.

### Output heads and score sharpness

| Head | Starting design |
|---|---|
| A | Learned linear output from 512 features to vocabulary-plus-blank scores. |
| B | Video projection from 512 to 300; compare with fixed 300-dimensional fastText embeddings. |
| C | Video projection from 512 to 512; shared text projection from 300 to 512. |

For B/C, normalize both sides and use cosine similarity divided by temperature. Start temperature at `0.07`, following the PDF. Keep it positive and monitor it. Fixed values such as `0.03`, `0.05`, and `0.1` are debugging alternatives selected using dev data.

Blank and retained special markers use learned representations. Ordinary fastText embeddings stay frozen. In C, every new label uses the already learned shared text projection. The PDF places C's blank/special representations directly in its 512-dimensional comparison space.

### CTC implementation checks

The planned PyTorch setup uses blank ID `0` and log probabilities shaped `(time, batch, classes)`. Correct target sequences must not contain blank. See the [PyTorch CTC documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html).

Check the length requirement:

```text
Available output time steps
    >= number of target labels
       + number of adjacent repeated target labels
```

For example, target `A A` needs an intervening blank in a valid CTC path. Too few time steps makes the alignment impossible. If using `zero_infinity=True`, also log invalid examples so zeroed losses do not hide a problem.

### Starting training settings

| Setting | Cached-feature head training | Clean video-reader training |
|---|---|---|
| Optimizer | Adam | Follow the verified CorrNet recipe. |
| Learning rate | Start at `1e-3`; try `3e-4` with a short warm-up if unstable. | The PDF suggests around `1e-4`. |
| Weight decay | `1e-3` | Use the verified reference configuration. |
| Learning-rate schedule | Cosine decay | Follow the reference recipe. |
| Batch size | Try 8-16 feature sequences if memory allows. | Measure memory first; full videos need more. |
| Epoch budget | 40 as a starting budget | 40 as a starting budget, subject to verification. |
| Selection | Development WER | Phase 1's clean selection rule. |
| Seeds | One for debugging; three for important reported results. | State how many full backbones were trained. |
| Decoding | Greedy first; consistent beam settings later. | Keep evaluation consistent. |

An **epoch** is one pass through the training set. **Learning rate** controls the size of learning updates. **Warm-up** begins with smaller updates and increases them gradually. **Weight decay** discourages some weights from becoming unnecessarily large.

The PDF proposes beam width 10 for final beam comparisons. Verify the decoder's label mapping and blank behavior before relying on its scores. A setting is a starting point, not a promise of accuracy or runtime.

### Extra checks that protect the research result

- Save the exact training and evaluation candidate lists.
- Identify extra labels that lost all training examples when videos were excluded.
- Keep full-data and reduced-data models and features separate.
- Do not repeatedly tune on final unseen-sign results.
- Use one fixed sequence-alignment rule for counting correct unseen occurrences.
- Report recall across occurrences and average recall across held-out label types where possible.
- Break down related-group and unrelated-group results. Related words are not automatically visually related signs.
- Check unseen precision and random/shuffled controls, not just whether the model beats zero recall.
- Record possible prior sign-data exposure of additional text models.

These checks help another person judge whether the method really transfers to held-out signs.

## How this relates to your PDF

| PDF topic | Where it appears here |
|---|---|
| Goal, background, and CTC | Introduction and Phase 2 |
| Setup, data, CorrNet, and features | Phases 1-2 |
| A, B, C, and controls | Phase 3 |
| Held-out glosses and clean training | Planning in Phase 1; experiments in Phase 4 |
| Spiking encoder and superposition | Optional tasks in Phase 5 |
| Risks, reporting, and references | Relevant phase tasks, coding notes, and final report |

The PDF supplies the research proposal. The simpler explanations, completion checks, and additional evaluation guidance make it easier to carry out. The five phases reorganize the PDF's material; their numbering differs from its original phases.
