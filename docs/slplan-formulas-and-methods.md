# SLPlan.pdf formulas and methods

This reference follows the 17-page **SLPlan.pdf**, *Open-Vocabulary Continuous Sign Language Recognition via Text-Embedding CTC Heads*. Page numbers refer to the original PDF. The PDF calls the frozen-CorrNet comparison **Phase 1** and the unseen-sign experiment **Phase 2**. The workspace's five-phase beginner plan uses different numbering.

The equations use consistent symbols. **Expanded notation** means a formula is written out here to explain a method the PDF describes in words or code.

| Symbol | Meaning |
|---|---|
| $T_{\mathrm{raw}}$ | Number of original picture frames in a clip |
| $T'$ | Number of valid feature steps after the video encoder |
| $T$ | Number of steps supplied to CTC in the general explanation |
| $t$ | One time step |
| $V$ | Number of gloss labels, excluding blank |
| $g$ | Gloss ID. ID 0 is CTC blank |
| $\mathcal S$ | Set of special annotation tokens, such as the ON and OFF markers |
| $h_t$ | Shared temporal model output at step $t$ |
| $z_t$ | Video vector after a head's video projection |
| $e_g$ | Text or class vector for gloss $g$ |
| $s_t(g)$ | Raw score, or *logit*, for gloss $g$ at step $t$ |
| $\tau$ | Temperature, a positive scale for cosine scores |
| $B$ in a shape | Number of clips in a batch |

## 1. Task, WER, and CTC (PDF §1–2, pages 1–2)

The input is a signed sentence, such as a 25 fps German Sign Language clip from PHOENIX-2014-T. The correct answer is an ordered sequence of glosses, for example **MORGEN WETTER GUT**. The model learns from that order without needing annotated start and end times for individual signs.

### Word error rate

$$
\mathrm{WER}=\frac{N_{\mathrm{substitutions}}+N_{\mathrm{deletions}}+N_{\mathrm{insertions}}}{N_{\mathrm{reference\ glosses}}}.
$$

A substitution is a wrong label, a deletion is a missing label, and an insertion is an extra label. Align the predicted and reference sequences to count the errors. Multiply the ratio by 100 to report WER as a percentage.

### CTC path, probability, and loss

At each time step, the model gives probabilities over $V$ glosses plus blank. A **path** $\pi=(\pi_1,\ldots,\pi_T)$ chooses one output at every step. Collapse a path by merging adjacent repeats first and then deleting blanks.

$$
\operatorname{collapse}(\text{M M - W W})=\text{M W},\qquad
\operatorname{collapse}(\text{GUT - GUT})=\text{GUT GUT}.
$$

The following equations use $x$ for the input clip and $y$ for its correct gloss sequence.

$$
P(\pi\mid x)=\prod_{t=1}^{T}p_t(\pi_t),\qquad
P(y\mid x)=\sum_{\pi:\operatorname{collapse}(\pi)=y}P(\pi\mid x),\qquad
\mathcal L_{\mathrm{CTC}}=-\log P(y\mid x).
$$

Multiply the probabilities along each path. Add the probabilities of all paths that produce the target, then take the negative log. The **CTC forward–backward algorithm** uses dynamic programming to calculate this sum efficiently. PyTorch's **CTCLoss** takes log probabilities shaped $(T,B,V+1)$ and the true sequence lengths.

### Worked example from the PDF

The target is **M W**. The dash is blank.

| Frame | M | W | Blank |
|---|---:|---:|---:|
| 1 | 0.6 | 0.1 | 0.3 |
| 2 | 0.3 | 0.3 | 0.4 |
| 3 | 0.1 | 0.7 | 0.2 |

| Valid path | Product | Probability |
|---|---|---:|
| M M W | $0.6\times0.3\times0.7$ | 0.126 |
| M W W | $0.6\times0.3\times0.7$ | 0.126 |
| – M W | $0.3\times0.3\times0.7$ | 0.063 |
| M – W | $0.6\times0.4\times0.7$ | 0.168 |
| M W – | $0.6\times0.3\times0.2$ | 0.036 |

$$
P(\text{M W}\mid x)=0.519,\qquad
\mathcal L_{\mathrm{CTC}}=-\ln(0.519)\approx0.66.
$$

### Soft alignment and gradient

The **posterior** $\gamma_t(g)$ is the probability that step $t$ is assigned gloss $g$ when we consider only paths that match the correct sequence. The PDF gives approximately 0.88 for M at frame 1 and 0.93 for W at frame 3. These probabilities act as soft targets.

The next equation expands the definition of the posterior.

$$
\gamma_t(g)=
\frac{\sum_{\pi:\operatorname{collapse}(\pi)=y,\;\pi_t=g}P(\pi\mid x)}
     {\sum_{\pi:\operatorname{collapse}(\pi)=y}P(\pi\mid x)}.
$$

The PDF gives the following gradient for a single clip.

$$
\frac{\partial\mathcal L_{\mathrm{CTC}}}{\partial s_t(g)}
=p_t(g)-\gamma_t(g).
$$

Training can raise the score if the model probability is below the posterior target and lower it if the probability is above that target.

### Minimum length and decoding

$$
T\ge\operatorname{length}(y)+\#\{\text{adjacent equal pairs in }y\}.
$$

**GUT GUT** needs at least three steps because a blank must separate the two copies. Apply the length rule to the actual steps going into CTC.

- **Greedy decoding** picks the highest score at each valid step and collapses the resulting path. In symbols, $\hat y=\operatorname{collapse}((\arg\max_g s_t(g))_{t=1}^{T})$.
- **Beam search** keeps several candidate sequences while decoding. The PDF proposes a beam width of 10 for final evaluation.

## 2. Video and text pipeline (PDF §3, pages 2–4)

| Branch | Method |
|---|---|
| Video | Encode the clip at roughly one feature step per four frames. A temporal model then produces a video embedding $z_t$ per step. |
| Text | Normalize the gloss strings and look them up in German fastText **cc.de.300**. Assemble the gloss vectors and a learned blank vector in a table indexed by gloss ID. |
| Join | Compare each video vector with every class vector. The resulting score table goes to CTC during training and to the decoder during inference. |

### Similarity matrix

The following score appears in the PDF's text-based pipeline diagram.

$$
s_t(g)=\frac{\cos(z_t,e_g)}{\tau}.
$$

The table has one row per feature step and one column per gloss or blank. Its shape is $T\times(V+1)$ in the overview, or $T'\times(V+1)$ using the detailed feature notation. Head C uses $P(e_g)$ in place of $e_g$ for ordinary glosses.

The equations below expand cosine, L2 normalization, softmax, and log softmax.

$$
\hat a=\frac{a}{\lVert a\rVert_2},\qquad
\cos(a,b)=\frac{a\cdot b}{\lVert a\rVert_2\lVert b\rVert_2}
=\hat a\cdot\hat b,\qquad
p_t(g)=\frac{\exp s_t(g)}{\sum_{j=0}^{V}\exp s_t(j)}.
$$

$$
\log p_t(g)=s_t(g)-\log\!\left(\sum_{j=0}^{V}\exp s_t(j)\right).
$$

Cosine measures agreement in direction and ranges from $-1$ to $1$ before dividing by temperature. The PDF gives about $0.05$–$0.1$ as a typical temperature range. The model learns $\log\tau$. A smaller $\tau$ makes score differences have a larger effect on the softmax probabilities.

### How the plan uses contrastive learning

Softmax compares each time step against the entire gloss vocabulary. CTC infers the soft assignment $\gamma_t(g)$, and $p_t(g)-\gamma_t(g)$ supplies the training signal. The PDF describes this as InfoNCE-style classification with competing labels from the vocabulary and comparisons going from video steps to glosses.

### Three optional auxiliary losses

| Method | How the PDF proposes to use it |
|---|---|
| Gloss-level InfoNCE | Once CTC alignment is stable, pool the frames assigned to each gloss. Compare the pooled video and text vectors in both directions across the batch. The PDF cites Papastratis et al. for the risk of adding alignment loss too early. |
| Bag of glosses, or superposition | Average the clip's $z_t$ vectors. Use one sigmoid per gloss to predict which labels occur anywhere in the sentence. The target is a set of glosses and contains no ordering. |
| Sentence-level CLIP | Compare a whole-video vector with a whole-sentence gloss-text vector. Use the other pairs in the batch as negatives. This auxiliary task trains video–text retrieval. |

These equations expand the pooling and sigmoid operations. Here $a_g$ is a score from the pooled vector.

$$
\bar z=\frac{1}{T'}\sum_{t=1}^{T'}z_t,\qquad
\sigma(a_g)=\frac{1}{1+\exp(-a_g)},\qquad
y_g^{\mathrm{bag}}=\mathbf 1[g\text{ occurs in the reference}].
$$

The optional losses are described in words. The PDF leaves their exact loss weights and variants open.

### Gloss text preprocessing

1. Lowercase gloss text for embedding, for example **WETTER** becomes **wetter**.
2. Handle **loc-**, **cl-**, and **neg-** prefixes. The general method offers stripping or encoding them separately. The detailed Head B recipe strips them and logs the changed strings.
3. Keep compounds and numbers in the detailed recipe. Use fastText's subword information to build their vectors.
4. The general method lists mapping special markers to blank or giving them separate learned vectors. The detailed B/C heads give special markers their own learned vectors and reserve ID 0 for CTC blank. A retained target label keeps a nonblank ID.

## 3. Related methods (PDF §4 and §8, pages 5–6 and 17)

| Work | Method described or named in the PDF |
|---|---|
| CorrNet, Hu et al. | ResNet18 with correlation and identification modules, 1D temporal convolution, BiLSTM, and a weight-normalized linear CTC classifier. The PDF describes VAC-style dual CTC and gives reference WER of 18.9% dev and 20.5% test for PHOENIX-2014T. |
| VAC, Min et al. | Visual Alignment Constraint, cited as CorrNet's base code. |
| TwoStream-SLR, Chen et al. | RGB and keypoint-heatmap S3D streams, lateral connections, a sign pyramid, self-distillation, and linear CTC heads. The reference WER listed is 17.7% dev and 19.3% test. |
| Papastratis et al. | Train a gloss language model and map visual and text vectors to a joint space. Weight Euclidean alignment by CTC forward–backward posteriors and use a shared linear classifier. Text supports training. |
| NLA-SLR, Zuo et al. | Use fastText gloss similarity for label smoothing and vision–gloss mixup in isolated sign recognition. |
| CVT-SLR and SignVTCL | Add contrastive visual–textual alignment to continuous sign recognition. The CVT-SLR reference also names variational alignment. |
| SignCLIP | CLIP-style pose and text pretraining on roughly 500,000 dictionary clips covering 44 sign languages. The PDF proposes its text encoder as a comparison. |
| Bilge et al. | Zero-shot recognition of isolated signs using text or attribute descriptions. |
| Reverse Sign Language Dictionary | Open-vocabulary sign retrieval from continuous signing via video captioning and description retrieval. |
| MIL-NCE, Miech et al. | Train S3D and a word-embedding text encoder contrastively on video with loosely aligned narration. The PDF names its S3D features as a generic alternative. |
| Graves et al. and Bojanowski et al. | The CTC training method and fastText subword vectors, respectively. |

## 4. Frozen CorrNet comparison (PDF §5, Phase 1, pages 6–14)

### Reference setup and feature caching (pages 7–8)

1. Use the official PHOENIX-2014-T train, development, and test splits, with gloss annotations as targets. The PDF's CorrNet preprocessing resizes $210\times260$ frames to $256\times256$ and builds the matching gloss dictionary. It names **dataset_preprocess-T.py --process-image --multiprocessing** and points to CorrNet issue #10 for a preprocessing index error.
2. Load the released PHOENIX-2014T checkpoint with its matching gloss IDs and blank at 0. The PDF sets **dataset: phoenix2014-T** in **configs/baseline.yaml** and runs CorrNet's **main.py** in test mode with **--load-weights**. Compare measured dev/test WER using CorrNet's scorer, normally **sclite**, with **python** as another option.
3. Cache the TemporalConv output before the original BiLSTM. The path is frames → ResNet18 and correlation modules → 512 numbers per frame → TemporalConv with 1D convolutions and pooling → **visual_feat**, shape $T'\times1024$ → original BiLSTM → NormLinear. The PDF gives $T'\approx T_{\mathrm{raw}}/4$. CorrNet has already trained the extraction point with its **ConvCTC** loss.
4. The proposed upstream patch returns **visual_features** and **feat_len** from **SLRModel.forward**. From the batched shape $(T',B,1024)$ it saves **visual_features[:feat_len[i], i]** for clip $i$, giving $(T'_i,1024)$. Keep the clip ID, labels, split, and length. The PDF proposes one NumPy feature file per clip and allows float16.
5. Extract in **model.eval()** mode with evaluation preprocessing and the non-augmented **train_eval** loader for train clips. Compare predictions from a few original dev videos with predictions from their saved features passed through CorrNet's original BiLSTM and classifier. An optional augmentation experiment caches 2–4 alternate views per training clip and samples one per epoch.

### Shared temporal model (page 8)

| Order | Layer described in the PDF |
|---|---|
| 1 | Linear $1024\rightarrow512$, LayerNorm, ReLU, and dropout 0.1. |
| 2 | Two residual Conv1d blocks, each with 512 input and output channels, kernel size 3, padding 1, BatchNorm, and ReLU. |
| 3 | Two-layer BiLSTM with 256 hidden units per direction. It gives $h_t\in\mathbb R^{512}$ at each of the same $T'$ steps. |
| 4 | The chosen output head. |

Use the same architecture and starting weights for head comparisons. The PDF also proposes optionally initializing from CorrNet's BiLSTM weights. Report parameter counts. The temporal model preserves sequence length.

### A — linear classifier (pages 3 and 9–10)

$$
s_t(g)=W_g\cdot h_t+b_g,\qquad
W\in\mathbb R^{(V+1)\times512},\qquad b\in\mathbb R^{V+1}.
$$

Learn a row and bias for each gloss ID, including blank. The PDF specifies Kaiming-uniform weight initialization and zero bias. This is the baseline for the other heads.

$$
N_{\mathrm A}=513(V+1)\approx0.56\ \text{million}\quad\text{for }V\approx1{,}100.
$$

All heads give batched logits shaped $(B,T',V+1)$. Apply **log_softmax** across the last dimension, then swap batch and time axes to make $(T',B,V+1)$ for CTC.

### A-norm — normalized linear classifier (pages 9–10)

$$
s_t(g)=h_t\cdot\frac{W_g}{\lVert W_g\rVert_2}.
$$

Learn each class vector and normalize it before the dot product. The CorrNet-style **NormLinear** uses no bias. The PDF says its parameter count scales with $V$. Counting the rows gives $512(V+1)$ parameters.

### B — fixed fastText vectors (pages 9–11)

$$
z_t=Uh_t+c,\qquad U\in\mathbb R^{300\times512},\qquad
s_t(g)=\frac{\cos(z_t,e_g)}{\tau}=\frac{\hat z_t\cdot\hat e_g}{\tau}.
$$

1. Build the text table once. Normalize each ordinary gloss and obtain its 300-dimensional vector using **ft.get_word_vector(s)**. Save the ID, original text, normalized text, and vector. Report changed labels and pairs with cosine similarity greater than $0.95$.
2. Keep the ordinary fastText rows fixed. Learn $e_0=e_{\mathrm{blank}}\in\mathbb R^{300}$ and one $e_s\in\mathbb R^{300}$ for each $s\in\mathcal S$. Initialize these learned vectors from $\mathcal N(0,1)$ and L2-normalize. The video projection uses the default linear initialization in the PDF.
3. Learn $U,c$, blank and special vectors, and log temperature along with the temporal model. The forward pass uses the temperature below.

$$
\log\tau_{\mathrm{init}}=\log(0.07),\qquad
\tau_{\mathrm{forward}}=\operatorname{clip}\big(\exp(\mathrm{log\_tau}),\,0.01,\,1.0\big).
$$

4. Assemble $E\in\mathbb R^{(V+1)\times300}$ in gloss-ID order. L2-normalize its rows and the projected video rows. For one clip, compute the score matrix by multiplying the normalized video matrix by the transpose of the normalized class table.

$$
\mathrm{logits}=\frac{\hat Z\hat E^\mathsf T}{\tau_{\mathrm{forward}}}.
$$

5. To add a candidate at inference, append its fastText vector as a row of $E$ and use the same CTC decoder. The PDF notes that related words such as **REGEN** and **SCHNEE** may produce nearby class vectors even if the signs are visually distinct.

$$
N_{\mathrm B}=513\cdot300+300(1+|\mathcal S|)+1\approx0.16\ \text{million}.
$$

This is the trainable output-layer count. The frozen vector table grows with the vocabulary.

### C — fastText with a shared text projection (pages 9 and 11–12)

$$
z_t=U'h_t+c',\qquad U'\in\mathbb R^{512\times512},\qquad
p_g=P(e_g),\qquad P:\mathbb R^{300}\rightarrow\mathbb R^{512},\qquad
s_t(g)=\frac{\cos(z_t,p_g)}{\tau}.
$$

1. Keep the source fastText vectors fixed and train the shared map $P$ for all ordinary glosses. Start with a linear map $300\rightarrow512$. The two-layer alternative is $300\rightarrow512\rightarrow512$ with GELU.
2. Learn blank and special vectors directly in the 512-dimensional space. These vectors bypass $P$. Initialize them from $\mathcal N(0,1)$ and normalize. Use default linear initialization for the projections, with the same temperature initialization and clamp as B.
3. Learn $U',c',P$, blank and special vectors, and log temperature along with the temporal model. Build the class table from the projected ordinary vectors and directly learned vectors. L2-normalize the rows on both sides and compute $\hat Z\hat E^\mathsf T/\tau_{\mathrm{forward}}$.
4. To add a candidate at inference, pass its fastText vector through the trained $P$ and append the resulting class row. The PDF recommends keeping $P$ small and logging its weight norm because it can overfit to training glosses.

$$
N_{\mathrm C}=513\cdot512+301\cdot512+512(1+|\mathcal S|)+1
\approx0.42\ \text{million}\quad\text{with linear }P.
$$

### B-rand and B-shuf — controls (pages 9 and 13)

| Control | Method | What it tests |
|---|---|---|
| B-rand | Use Head B with fixed random 300-dimensional unit vectors and a recorded seed per gloss. Expanded notation is $e_g=r_g/\lVert r_g\rVert_2$ for a sampled vector $r_g$. | How much a fixed class vector helps on its own. |
| B-shuf | Use Head B with the fastText rows randomly permuted across ordinary gloss IDs. Expanded notation is $e_g=e^{\mathrm{fastText}}_{\rho(g)}$, where $\rho$ is a permutation. | Whether matching the vector to the correct word helps. |

Start with A and B. Add C, A-norm, B-rand, and B-shuf once B works. If B, B-rand, and B-shuf have similar WER, the seen-gloss runs give weak evidence that word meaning helped.

### Training and model selection (pages 13–14)

| Item | PDF method |
|---|---|
| Loss | **torch.nn.CTCLoss(blank=0, zero_infinity=True)** on the log softmax of the logits, in shape $(T',B,V+1)$. Pass true lengths after padding. |
| Optimizer | Adam with learning rate $10^{-3}$, weight decay $10^{-3}$, and cosine annealing. If B/C output almost entirely blanks early, try $3\times10^{-4}$ and a 1–2 epoch linear warm-up. |
| Batch and epochs | 8–16 clips per batch, 40 epochs. |
| Model selection | Evaluate dev WER every epoch with greedy decoding and choose the checkpoint with the lowest dev WER. |
| Final decoding | CorrNet's **ctcdecode** beam search with width 10 and its WER scorer. |
| Seeds | One for the first A/B check and three per variant for reporting. Give mean $\pm$ standard deviation across seeds. |
| Reporting | Save dev/test WER for greedy and beam, substitutions/deletions/insertions, output-layer parameter counts, temperature, blank rate, and per-gloss dev recall grouped by training frequency. Plot a time-step by gloss posterior heatmap on 2–3 dev clips per variant. |

The following equations expand the PDF's blank-rate and per-gloss recall measures.

$$
\mathrm{blank\ rate}=\frac{\#\{t:\arg\max_g s_t(g)=0\}}{\#\{\text{valid steps}\}},
\qquad
\mathrm{recall}(g)=
\frac{\#\{\text{correctly recovered occurrences of }g\}}{\#\{\text{reference occurrences of }g\}}.
$$

Proceed if B or C is within about 1–2 absolute dev WER points of A. If both are more than about 3 points worse after debugging, revisit the comparison before the held-out study. The numbered checks follow the PDF's order.

1. Check normalization for zero vectors, shared vectors, and near-duplicates.
2. Try fixed $\tau\in\{0.03,0.05,0.1\}$.
3. Start from the trained A temporal model and switch its output layer.
4. Try a larger Head C space with $P$ projecting to 1,024 dimensions and video vectors of the same size.
5. Inspect the blank-rate curve for collapse to mostly blank outputs.

## 5. Held-out glosses (PDF Phase 2, pages 14–15)

1. Hold out about 10% of gloss types, with both shared sign families and unrelated glosses. Save the held-out list and remove each training clip containing any held-out gloss. The expanded formula below uses $\mathcal H$ for the held-out set.

   $$
   \mathcal D_{\mathrm{reduced}}=
   \{(x,y)\in\mathcal D_{\mathrm{train}}:\operatorname{set}(y)\cap\mathcal H=\varnothing\}.
   $$

2. Start CorrNet from ImageNet ResNet18 and retrain it with the reduced split. The PDF proposes Adam, learning rate $10^{-4}$, and 40 epochs. It notes the original batch-2, two-GPU recipe and suggests testing batch size 1 on an 8 GB GPU or using cloud compute. Cache the new features, then train B/C. Optionally train the backbone and B/C together end to end.
3. As an alternative, fine-tune Kinetics-400 S3D with CTC on the reduced split. For a generic baseline, freeze ImageNet ResNet18 or Kinetics/MIL-NCE S3D. A clean study uses a backbone without sign-training exposure to the held-out labels. The released sign-trained CorrNet and TwoStream models serve as seen-sign references.
4. Compare German fastText, multilingual LaBSE, averaged mBART gloss sub-token embeddings, the SignCLIP text encoder, and short sign descriptions of handshape, location, and movement. Report potential German Sign Language exposure in SignCLIP's pretraining data.
5. Add the held-out candidates to B/C at inference as class vectors. Use A for seen glosses. A with the seen-only vocabulary has zero recall for held-out labels by construction.
6. Measure WER on test clips containing held-out glosses, recall on those glosses, and seen-gloss WER. Break down results by shared sign family and unrelated glosses.

Expanded notation for mBART averaging. For sub-token embeddings $u_1,\ldots,u_K$, use $e_g=\frac{1}{K}\sum_{j=1}^{K}u_j$.

## 6. Spiking and superposition experiments (PDF Phases 3–4, pages 14–15)

- **Spiking encoder.** Replace the video encoder with SpikingVideoNet3D and keep the embedding head the same. Compare seen and unseen WER, spike counts, and energy estimates with the conventional backbone. The PDF leaves the spike and energy accounting equations open.
- **Superposition analysis.** Use the bag-of-glosses objective on pooled video embeddings. Measure how many glosses can be recovered as sentence length and embedding dimension change.
- **Later backbone comparison.** The PDF lists TwoStream-SLR with RGB plus keypoints as another backbone to compare with A/B/C if the required assets become available.

## 7. Research workflow and risks (PDF §5.0 and §6–8, pages 6 and 16–17)

- Start reduced-split CorrNet training in parallel with the frozen-reference check and the first A/B runs. Save the held-out gloss list and split early.
- Test related and unrelated signs, and compare descriptions and other text encoders, because word meaning alone need not predict sign form.
- Monitor temperature and blank rate, try a CTC-trained warm start, and introduce auxiliary contrastive losses after CTC alignment stabilizes.
- Compare larger Head C embeddings if the 300-dimensional text space appears limiting.
- Cache frozen features to reduce repeated work. For backbone training the PDF considers batch size 1, mixed precision, a lighter ResNet18-based model, and cloud compute.
- The PDF records a checkpoint availability problem for TwoStream at the time of writing and proposes asking the authors for updated links. It also proposes reviewing the literature on open-vocabulary CSLR and zero-shot CSLR before making novelty claims.

