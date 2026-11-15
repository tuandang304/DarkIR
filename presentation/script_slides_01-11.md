# NightClear: Presentation Script (Slides 1–11)

Target time for slides 1–11: about **10 minutes** (roughly 50–60 s per slide).
Lines in *[brackets]* are stage cues, not things to say.

---

## Slide 1: Title (~30 s)

Good morning, Mr. Tai, and everyone.
We are Mai Xuan Duc, Phan Minh Nhat Truong and Dang Thanh Tuan, and today we present our project, **NightClear: Joint Low-Light Image Enhancement and Deblurring**.

In one sentence: NightClear takes a dark, noisy and blurry night photo and restores it in a **single deep-learning model**, so the image becomes brighter, cleaner and sharper at the same time.

---

## Slide 2: Table of Contents (~30 s)

Our talk has seven parts.
- We start with the **problem** and our **objectives**, then look at **related work**.
- Next comes the **data**: the datasets, our data pipeline and what the exploratory analysis told us.
- Then we explain the **method**, show the **results**, give a short **demo** of our web app, and finish with **limitations and future work**.

---

## Slide 3: Problem Statement (~60 s)

Everyone has taken a bad photo at night. The image usually suffers from three problems **at once**.
- **Low illumination**: there is not enough light, so the image is dark and details disappear.
- **Sensor noise**: to compensate, the camera raises its gain, and the sensor noise becomes very visible.
- **Motion blur**: the exposure is longer, so any movement of the hand or the object smears the image.

*[Point to the formula.]*
We can describe this with a simple image formation model: **y = γ(x ∗ k) + n**.
- *x* is the clean, sharp scene we want.
- It is convolved with a blur kernel *k*, which is the motion blur.
- γ is the camera response that squeezes the dynamic range and makes everything dark.
- Finally, sensor noise *n* is added, and *y* is what we actually capture.

Our task is the inverse problem: given only *y*, recover *x*. The three degradations are mixed together, which is what makes this problem hard.

---

## Slide 4: Project Objectives (~60 s)

Our **research motivation** is that brightening, denoising and deblurring are **coupled tasks**. If we solve them one after another, the errors from one step pass into the next, and this gets worse with low-quality cameras.

We set three objectives.
1. **Low-light enhancement**: improve brightness, visibility and natural colour.
2. **Quality restoration**: reduce noise and motion blur while keeping structure and fine detail.
3. **Robustness and efficiency**: check quality, speed and robustness across different camera inputs.

Our scope has three steps.
- **Reproduce**: implement the core method and set up baselines.
- **Adapt**: test it on low-resolution, noisy images from cheap cameras.
- **Extend**: look for evidence-based improvements in robustness and efficiency.

---

## Slide 5: Why Data Science? Objectives & KPIs (~60 s)

Why do we need a data-driven model at all, instead of a hand-made filter? There are three reasons.
- **No single curve fits every scene**: the right amount of brightening is different for every photo, as our EDA will show.
- **Errors pass down a chain**: if separate brightening and deblurring models run one after another, each passes its mistakes to the next. One model trained jointly avoids this.
- **Large paired datasets exist**: a network can learn the inverse of the night-time degradation directly from examples.

*[Point to the KPI table.]*
To judge success we use four KPIs.
- **Fidelity**: the output should be close to the ground truth, measured with PSNR and SSIM, and better than the previous joint method.
- **Perceptual quality**: it should look natural to a person, with no colour shift or artefacts, measured with LPIPS.
- **Efficiency**: the model should be small enough to run on a laptop CPU or a single consumer GPU.
- **Usability**: we want a working application that restores photos and video end to end, not only numbers in a table.

---

## Slide 6: Existing Approaches (~50 s)

Existing work falls into three groups.
- **Low-light enhancement** methods such as RetinexNet and Zero-DCE focus on illumination and contrast. They brighten the image but do not remove blur.
- **Image deblurring** methods such as DeblurGAN and NAFNet remove blur and restore detail, but they are usually trained on well-lit images.
- **Joint restoration** methods such as **LEDNet** and **DarkIR** handle low light and blur together in one network.

Our project belongs to this third group. NightClear is inspired by DarkIR and LEDNet, and its encoder and decoder modules are adapted from these two works.

---

## Slide 7: Limitation of Sequential Restoration (~50 s)

*[Follow the diagram from left to right.]*
The simple solution would be a cascade: first an enhancement model, then a deblurring model.

This has two problems.
- **Noise amplification**: when we brighten a dark image, we also brighten its noise, so the deblurring model receives a noisy input it was never trained for.
- **Error propagation**: any mistake from the first model cannot be fixed by the second one, which only adds its own errors on top.

Visibility and sharpness are damaged together, so they should be restored together. This is the motivation for a **joint restoration** model and for studying robust night-time restoration under difficult conditions.

---

## Slide 8: Dataset Overview (~60 s)

Every training sample is a **pair**: a low-light image and its normal-light ground truth.
- **LOL-Blur** is the main dataset: about 10,200 training pairs with low light, blur and noise together, at 1120 by 640.
- **LOLv2** has a real part with 689 training pairs and a synthetic part with 900. Each part has 100 test pairs.
- **LSRW** adds about 5,600 real pairs from Nikon and Huawei cameras.
- We also use **real night photos without ground truth**, such as DICM, LIME, MEF, NPE and VV, only for visual testing.

*[Point to the left column.]*
We found three issues in the raw data.
- **Files are clean**: there are no missing or corrupt files, and every input has its ground truth.
- **Imbalance**: LOL-Blur alone is 57% of all paired training data.
- **Tiny test sets**: LSRW, for example, has only a few dozen test pairs, so scores on it are noisy.

*[Point to the right column.]*
For preprocessing:
- we use random crops, flips and rotations during training;
- pixels are scaled to [0, 1], and every image is padded to a multiple of 8, because the network downsamples three times;
- for deployment, large images can optionally be resized.

---

## Slide 9: Data Pipeline (~60 s)

This slide shows how data flows from the internet to the GPU.

*[Top row, left to right.]*
The first three steps run **once, offline**.
1. **Paired datasets**: we start from the public datasets.
2. **Download and unpack**: the archives are fetched automatically and the folder layout is checked.
3. **Pair and split**: each low-light image is matched with its normal-light image by ID, unmatched files are dropped, and we keep the official train/test split.

The next three steps run **on the fly, for every batch**, so each epoch sees slightly different data.
4. **Pre-process**: convert to RGB, scale to [0, 1], and take a random crop at the **same position** in the input and the target.
5. **Augment**: apply the same random flip and 90-degree rotation to both images of a pair.
6. **Mini-batches**: shuffled batches are sent to the GPU to train NightClear.

*[Bottom row.]*
From the pair-and-split step, two more branches start.
- **Test branch (orange)**: held-out test pairs are kept at full resolution with no augmentation. They pass through NightClear and are scored with PSNR, SSIM and LPIPS.
- **Analysis branch (grey)**: all pairs go to EDA and quality checks. The findings guide our model design, as the next two slides show.

---

## Slide 10: Key EDA Insights (1/2): How Dark Is the Data? (~60 s)

We analysed all 1,789 LOLv2 pairs using luminance, or brightness, from 0 to 1.

*[Left chart.]*
- **Real images are very dark**: real low-light images have an average brightness of only **0.06**, against **0.44** for the ground truth.
- **Synthetic images are much brighter**, at **0.24**.

So there is a clear **domain gap** between real and synthetic data. A model trained mostly on synthetic images may not work well on real night photos.

*[Right chart.]*
- **Most pixels are almost black**: **81%** of the pixels in real low-light images have a brightness below 0.1.
- **Noise dominates there**: almost all the useful information is packed into a few dark grey levels, where noise is strongest. This is why denoising has to be part of the model and not an afterthought.

---

## Slide 11: Key EDA Insights (2/2): Brightening Is Not One Curve (~60 s)

Here we measure the **gain**: how many times brighter the ground truth is than the low-light input, for each pair.
- **Real pairs**: the median gain is about **7.5 times**, and 90% of pairs fall between **2.8 and 30 times**, more than a tenfold range.
- **Synthetic pairs**: the gain is much smaller, with a median of about **2.5 times**.

*[Point to the scatter plot.]*
The points do not lie on one line. Images that look equally dark can need very different amounts of brightening.

**Conclusion**: a fixed global curve, such as a gamma correction or histogram equalisation, cannot fit all of these pairs. The model has to look at the **content** of each image and decide locally how much to brighten. This is exactly what a learned model can do.

*[Transition.]*
This leads us to our method. *[Hand over to the next speaker.]*
