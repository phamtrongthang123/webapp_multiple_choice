# SAM 3D: 3Dfy Anything in Images - Multiple Choice Quiz (Part 2)

## 50 Questions with Trade-Off Explanations

Each answer includes an explanation using the format:
**Input:** [what you're working with], **Goal:** [what you want to achieve], **Approach:** [your solution idea], **Because:** [reasoning and trade-offs]

---

### **Section 1: Core Concepts and Motivation (Questions 1-10)**

**1. Why do previous image-to-3D models struggle with natural scene images?**
- A) They require too much GPU memory
- B) They are trained on isolated objects and struggle with occlusion and clutter
- C) They only work with synthetic images
- D) They cannot process high-resolution images

**Answer: B**

**Input:** Previous models trained on isolated synthetic objects, **Goal:** Reconstruct 3D from real-world cluttered scenes, **Approach:** These models fail because they lack training on occluded/cluttered scenes, **Because:** Training on clean, centered objects doesn't transfer to images where objects are distant, partially visible, or surrounded by visual noise—the distribution shift is too large.

---

**2. What makes the "familiar object" cue important for single-image 3D reconstruction?**
- A) It allows faster processing
- B) Recognition of known object types enables inference of their 3D structure
- C) It reduces memory requirements
- D) It improves color accuracy

**Answer: B**

**Input:** A single 2D image with ambiguous depth, **Goal:** Recover 3D shape without multiple views, **Approach:** Leverage recognition to infer structure, **Because:** If we recognize a chair, we can infer it has legs, a seat, and a back—even parts we can't see. Recognition fills in geometric priors that pure geometry cannot provide from one view.

---

**3. Why can't generalist human annotators directly create 3D shape ground truth?**
- A) They don't have access to computers
- B) Creating 3D meshes requires specialized skills that take hours even for trained artists
- C) 3D software is too expensive
- D) There are copyright restrictions

**Answer: B**

**Input:** Need for large-scale 3D annotations, **Goal:** Collect millions of 3D training samples, **Approach:** Can't use direct mesh creation by annotators, **Because:** The trade-off is annotation quality vs. scalability—skilled 3D artists take hours per mesh, making it impossible to scale to millions of samples with direct creation.

---

**4. What is the "virtuous cycle" in SAM 3D's data engine?**
- A) A type of neural network architecture
- B) Model improvements lead to better annotations, which further improve the model
- C) A data augmentation technique
- D) A loss function optimization method

**Answer: B**

**Input:** Initial weak model and limited training data, **Goal:** Continuously improve both model and data quality, **Approach:** Use improved model to generate better candidates for annotation, **Because:** This creates a positive feedback loop—better models propose better 3D candidates, annotators select higher quality samples, and training on these improves the model further.

---

**5. Why does SAM 3D use both cropped object and full image as inputs?**
- A) To increase batch size
- B) Cropped view provides detail while full image provides context and recognition cues
- C) To reduce memory usage
- D) To speed up inference

**Answer: B**

**Input:** An object in a cluttered scene, **Goal:** Maximize information for accurate reconstruction, **Approach:** Encode both focused crop and full context, **Because:** The trade-off is detail vs. context—crops give high-resolution object features but lose scene cues; full images provide recognition context but at lower object resolution. Using both captures complementary information.

---

**6. What problem does the Mixture-of-Transformers (MoT) architecture solve?**
- A) Reducing training time
- B) Enabling independent training of different modalities while maintaining cross-modal interaction
- C) Increasing model size
- D) Improving image quality

**Answer: B**

**Input:** Multiple output modalities (shape, rotation, translation, scale), **Goal:** Train flexibly on datasets with partial labels while maintaining consistency, **Approach:** Use MoT with structured attention masks, **Because:** The trade-off is modularity vs. joint reasoning—separate transformers allow training on shape-only data, but shared attention ensures rotation predictions are consistent with the predicted shape.

---

**7. Why does SAM 3D model reconstruction as a conditional distribution rather than a deterministic function?**
- A) To use less memory
- B) Because the 3D-to-2D mapping is lossy and multiple valid 3D shapes can explain one image
- C) To speed up training
- D) To simplify the architecture

**Answer: B**

**Input:** A 2D image where depth information is lost, **Goal:** Handle inherent ambiguity in single-view reconstruction, **Approach:** Model p(S,T,R,t,s|I,M) as a distribution, **Because:** The trade-off is certainty vs. realism—a deterministic model would give one answer but might be wrong; a generative model can sample multiple plausible reconstructions, better reflecting true uncertainty.

---

**8. What is the purpose of using point maps as optional conditioning?**
- A) To replace the image encoder
- B) To provide additional depth cues that help with layout estimation
- C) To reduce model parameters
- D) To improve texture quality

**Answer: B**

**Input:** Ambiguous layout from RGB alone, **Goal:** Improve translation and scale accuracy, **Approach:** Optionally condition on point maps from LiDAR or monocular depth, **Because:** The trade-off is accessibility vs. accuracy—RGB-only works everywhere but has layout ambiguity; point maps add geometric constraints but require additional sensors or estimation.

---

**9. Why did the authors choose 6D rotation representation instead of Euler angles or quaternions?**
- A) It uses less memory
- B) 6D representation is continuous and avoids discontinuities in neural network learning
- C) It's faster to compute
- D) It's more intuitive for humans

**Answer: B**

**Input:** Need to predict object orientation, **Goal:** Learn rotation prediction effectively with neural networks, **Approach:** Use 6D continuous rotation representation, **Because:** The trade-off is compactness vs. learnability—Euler angles (3D) and quaternions (4D) have discontinuities or constraints that make gradient-based learning harder; 6D is redundant but continuous and easier to learn.

---

**10. What distinguishes SAM 3D from methods like NeRF or 3D Gaussian Splatting?**
- A) SAM 3D uses more GPUs
- B) SAM 3D reconstructs from a single image at test time, while others typically need multiple views
- C) SAM 3D only works with synthetic images
- D) SAM 3D produces lower quality results

**Answer: B**

**Input:** Various 3D reconstruction methods, **Goal:** Enable practical real-world 3D perception, **Approach:** Single-image reconstruction vs. multi-view, **Because:** The trade-off is convenience vs. geometric precision—multi-view methods can be more geometrically accurate but require capturing multiple images; single-view is practical for any existing photo but must rely on learned priors.

---

### **Section 2: Training Strategy Deep Dive (Questions 11-20)**

**11. Why does SAM 3D use synthetic pretraining before real-world data?**
- A) Synthetic data is more colorful
- B) Synthetic data provides unlimited 3D ground truth to learn shape/texture vocabulary
- C) Real data is not available
- D) Synthetic data trains faster

**Answer: B**

**Input:** Need to learn 3D shape generation, **Goal:** Build foundational 3D generation capabilities, **Approach:** Pretrain on rendered synthetic objects with perfect ground truth, **Because:** The trade-off is domain gap vs. data quality—synthetic data has perfect 3D labels at unlimited scale but differs from real images; this gap is later closed in post-training.

---

**12. What is the purpose of "Flying Occlusions" in mid-training?**
- A) To make training faster
- B) To teach the model occlusion robustness by compositing occluder-occludee pairs
- C) To reduce model size
- D) To improve color prediction

**Answer: B**

**Input:** Model trained on isolated objects, **Goal:** Handle partially visible objects in real scenes, **Approach:** Composite synthetic objects as occluders over target objects, **Because:** The trade-off is realism vs. control—real occlusions are diverse but lack ground truth; synthetic occlusions give exact visibility masks for supervised learning of shape completion.

---

**13. Why is the quality threshold α in post-training increased over time?**
- A) To reduce training time
- B) To progressively raise the bar as the model improves, similar to curriculum learning
- C) To save storage space
- D) To decrease model size

**Answer: B**

**Input:** Data collected across multiple iterations, **Goal:** Maximize training signal quality, **Approach:** Increase acceptance threshold α over time, **Because:** The trade-off is data quantity vs. quality—early on, accepting lower quality samples provides learning signal; as the model improves, stricter thresholds ensure only high-quality examples refine the model further.

---

**14. What problem does Object Swap - Random (OS-R) solve that Flying Occlusions doesn't?**
- A) Color accuracy
- B) Learning translation and scale estimation with depth-aware visual cues
- C) Faster processing
- D) Better textures

**Answer: B**

**Input:** Need to learn object layout in scenes, **Goal:** Predict translation and scale from visual cues, **Approach:** Replace objects with depth-aware rendering that preserves occlusion relationships, **Because:** The trade-off is pose information vs. semantic relevance—Flying Occlusions teaches occlusion handling but lacks realistic layout; OS-R provides T-junction and depth ordering cues needed for layout estimation.

---

**15. Why does SAM 3D use DPO (Direct Preference Optimization) after SFT?**
- A) To reduce model size
- B) To align outputs with human aesthetic preferences like symmetry and closure
- C) To speed up inference
- D) To use less training data

**Answer: B**

**Input:** Model after supervised fine-tuning, **Goal:** Eliminate subtle undesirable outputs (floaters, asymmetry), **Approach:** Use human preference pairs in DPO, **Because:** The trade-off is explicit supervision vs. implicit preferences—SFT teaches correct shapes but can't capture subjective quality aspects; DPO learns from comparative judgments what humans find aesthetically pleasing.

---

**16. What is the purpose of the Art-3DO dataset created by professional 3D artists?**
- A) To replace all other training data
- B) To provide high-quality supervision for the hardest cases where models fail
- C) To reduce training time
- D) To improve inference speed

**Answer: B**

**Input:** Hard examples where no model produces acceptable meshes, **Goal:** Break the chicken-and-egg problem for tail distribution, **Approach:** Use skilled 3D artists to create ground truth for failure cases, **Because:** The trade-off is cost vs. coverage—artists are expensive (hours per mesh) but essential for seeding model capability in regions where self-improvement can't bootstrap.

---

**17. Why does mid-training use 2.7 trillion tokens, matching pretraining?**
- A) It's a random choice
- B) Sufficient exposure is needed to learn new skills while retaining pretraining knowledge
- C) To use all available GPUs
- D) To fill storage capacity

**Answer: B**

**Input:** Pretrained model learning new capabilities, **Goal:** Add mask-following and occlusion handling without forgetting shape generation, **Approach:** Extensive mid-training matching pretraining scale, **Because:** The trade-off is capability acquisition vs. catastrophic forgetting—too little mid-training doesn't teach new skills; this scale ensures robust skill integration while maintaining foundation capabilities.

---

**18. What advantage does flow matching have over diffusion for SAM 3D?**
- A) Uses less memory
- B) Provides straight probability paths that are easier to learn and faster to sample
- C) Produces more colorful outputs
- D) Requires fewer parameters

**Answer: B**

**Input:** Need for generative 3D modeling, **Goal:** Efficient training and inference for 3D generation, **Approach:** Use rectified conditional flow matching, **Because:** The trade-off is sample quality vs. efficiency—diffusion models work well but require many steps; rectified flow provides straighter paths enabling fewer sampling steps with comparable quality.

---

**19. Why does SAM 3D distill the model to reduce NFE from 25 to 4?**
- A) To improve quality
- B) To enable sub-second inference for real-time applications like robotics
- C) To increase accuracy
- D) To use more parameters

**Answer: B**

**Input:** Trained flow matching model requiring 25 steps, **Goal:** Enable real-time 3D perception, **Approach:** Distill using shortcut models to reduce to 4 steps, **Because:** The trade-off is quality vs. speed—25 steps give best quality but too slow for robotics; distillation sacrifices some quality for ~6x speedup, enabling practical deployment.

---

**20. What is the significance of using both MITL-3DO and Art-3DO in SFT sequencing?**
- A) Random ordering
- B) Train on noisier crowd-sourced data first, then fine-tune on smaller high-quality artist data
- C) Use them simultaneously
- D) Only use Art-3DO

**Answer: B**

**Input:** Two datasets of different quality and size, **Goal:** Maximize final model quality, **Approach:** SFT on MITL-3DO first, then Art-3DO, **Because:** The trade-off is scale vs. polish—MITL-3DO (millions of samples) provides broad coverage but has annotation noise; following with smaller, pristine Art-3DO aligns outputs with artist aesthetic preferences.

---

### **Section 3: Data Engine Mechanics (Questions 21-30)**

**21. Why do annotators make pairwise comparisons rather than rating individual meshes?**
- A) It's faster
- B) Relative judgments are more reliable and consistent than absolute quality scores
- C) It uses less screen space
- D) It requires less training

**Answer: B**

**Input:** Multiple 3D mesh candidates to evaluate, **Goal:** Collect reliable quality judgments, **Approach:** Pairwise comparison tournament, **Because:** The trade-off is information richness vs. reliability—absolute ratings vary by annotator calibration; relative comparisons ("A is better than B") are more consistent and avoid calibration issues.

---

**22. What is the purpose of randomizing candidate presentation order in Stage 2?**
- A) To confuse annotators
- B) To prevent order-based biases from affecting selection
- C) To speed up annotation
- D) To reduce storage

**Answer: B**

**Input:** N candidates shown sequentially to annotators, **Goal:** Unbiased quality assessment, **Approach:** Randomize presentation order, **Because:** The trade-off is experimental rigor vs. simplicity—fixed ordering could create primacy/recency biases; randomization ensures selections reflect true quality rather than position effects.

---

**23. Why does the data engine use an ensemble of different 3D generation methods?**
- A) To use more GPUs
- B) To maximize the chance of having at least one good candidate for any given input
- C) To increase training time
- D) To reduce model quality

**Answer: B**

**Input:** Diverse objects with varying difficulty, **Goal:** Ensure successful annotation for most inputs, **Approach:** Combine retrieval, text-to-3D, and image-to-3D methods, **Because:** The trade-off is complexity vs. coverage—single methods fail on certain inputs; ensembling different approaches (retrieval for semantic matches, generation for novel shapes) provides complementary strengths.

---

**24. What is the "cold start problem" in SAM 3D's data engine?**
- A) GPU temperature issues
- B) Initial model produces few high-quality candidates because no real-world 3D data exists yet
- C) Slow network connections
- D) Memory limitations

**Answer: B**

**Input:** First iteration with untrained model, **Goal:** Begin collecting quality training data, **Approach:** Leverage existing methods as ensemble to bootstrap, **Because:** The trade-off is bootstrapping vs. self-improvement—you can't improve a model without data, but you can't get good data without a good model; external methods break this cycle initially.

---

**25. Why does Stage 3 use point clouds rather than RGB images for pose annotation?**
- A) Point clouds are more colorful
- B) Point clouds provide explicit 3D structure that enables consistent placement
- C) They load faster
- D) They use less memory

**Answer: B**

**Input:** 3D mesh needing pose alignment, **Goal:** Accurate and consistent R, t, s annotations, **Approach:** Align mesh to 2.5D point cloud from depth estimation, **Because:** The trade-off is annotation difficulty vs. accuracy—placing 3D objects in 2D images is ambiguous; point clouds make depth explicit, enabling annotators to anchor meshes to physical scene structure.

---

**26. What happens to mesh candidates that don't meet the quality threshold?**
- A) They are deleted
- B) They become negative examples for preference alignment (DPO)
- C) They are used for pretraining
- D) They are sent to artists

**Answer: B**

**Input:** Rejected mesh candidates from annotation, **Goal:** Extract value from failed annotations, **Approach:** Use as negative examples in preference pairs, **Because:** The trade-off is data utilization vs. quality—rejected meshes aren't good enough for SFT but still provide signal; pairing them with accepted meshes teaches the model what NOT to produce.

---

**27. Why does the reward model pipeline generate 50 seeds for hard examples?**
- A) It's a random number
- B) Higher N increases probability of finding at least one acceptable output for tail distribution
- C) To fill GPU memory
- D) To slow down training

**Answer: B**

**Input:** Examples where standard sampling fails, **Goal:** Recover training data from difficult inputs, **Approach:** Scale up best-of-N search from 8 to 50, **Because:** The trade-off is compute vs. coverage—50 samples is expensive but dramatically increases success probability for hard cases (food category improved 9x from 4% to 36%).

---

**28. Why can't annotators directly edit meshes in Stage 2?**
- A) The software doesn't support it
- B) Editing requires specialized skills; selection/verification scales better with generalist annotators
- C) It would be too fast
- D) Copyright restrictions

**Answer: B**

**Input:** Need for large-scale 3D annotation, **Goal:** Scale annotation to millions of samples, **Approach:** Restrict annotators to selection only, **Because:** The trade-off is flexibility vs. scalability—mesh editing requires training and produces variable quality; binary selection is a simpler task that generalists can perform reliably and quickly.

---

**29. What determines whether an example is routed to 3D artists in Stage 2.5?**
- A) Random selection
- B) When no model in the ensemble produces a mesh meeting the quality threshold
- C) Based on image size
- D) Based on file format

**Answer: B**

**Input:** Examples where all model candidates are rejected, **Goal:** Provide data for model blind spots, **Approach:** Route genuine failure cases to artists, **Because:** The trade-off is artist cost vs. model improvement—artists are expensive so we only use them where essential; failure filtering ensures each artist-created mesh addresses a real model limitation.

---

**30. How does the data engine handle the object category distribution?**
- A) Random sampling
- B) Curriculum-inspired sampling progressing from simple to complex, with adaptive rebalancing
- C) Alphabetical ordering
- D) Size-based ordering

**Answer: B**

**Input:** Diverse object categories with varying difficulty, **Goal:** Balanced coverage across all categories, **Approach:** Start with simple shapes, progressively add complex/deformable objects, monitor and rebalance, **Because:** The trade-off is learning efficiency vs. coverage—starting with hard examples wastes annotation effort; curriculum lets the model build capability before tackling challenging categories.

---

### **Section 4: Evaluation and Comparisons (Questions 31-40)**

**31. Why does SA-3DAO use professional 3D artists to create ground truth?**
- A) It's cheaper
- B) Artists represent expert human upper bound for visually grounded 3D reconstruction
- C) They work faster
- D) They have more GPUs

**Answer: B**

**Input:** Need for benchmark ground truth, **Goal:** Establish gold standard for evaluation, **Approach:** Use professional 3D artists, **Because:** The trade-off is cost vs. quality—automated methods would introduce their own biases; human artists provide the best possible reference for what a perfect reconstruction should look like.

---

**32. Why does SAM 3D significantly outperform baselines on SA-3DAO but show smaller gaps on ISO3D?**
- A) Random variation
- B) SA-3DAO tests real-world conditions (occlusion, clutter) where SAM 3D's training excels
- C) ISO3D is harder
- D) Different image formats

**Answer: B**

**Input:** Two different evaluation benchmarks, **Goal:** Understand where SAM 3D's advantages lie, **Approach:** Compare performance across datasets, **Because:** The trade-off revealed is specialization vs. generalization—baselines trained on isolated objects do okay on similar ISO3D images; SAM 3D's real-world training pays off on challenging SA-3DAO conditions.

---

**33. What does the 5:1 win rate in human preference tests indicate?**
- A) SAM 3D is 5 times faster
- B) Humans prefer SAM 3D outputs 5 times more often than alternatives
- C) SAM 3D uses 5 times more parameters
- D) SAM 3D costs 5 times more

**Answer: B**

**Input:** Pairwise human evaluations, **Goal:** Measure perceptual quality beyond metrics, **Approach:** Head-to-head preference comparison, **Because:** The trade-off is objectivity vs. relevance—automated metrics are consistent but may not capture what matters to users; human preference directly measures practical quality.

---

**34. Why does SAM 3D evaluate texture separately using its own geometry?**
- A) To hide poor geometry
- B) To isolate texture quality assessment—better geometry from SAM 3D actually helps competing texture methods
- C) It's faster
- D) To reduce memory

**Answer: B**

**Input:** Need to fairly evaluate texture generation, **Goal:** Measure texture quality independent of geometry, **Approach:** Use SAM 3D geometry for all texture methods, **Because:** The trade-off is fairness vs. end-to-end evaluation—poor geometry makes texture look bad regardless of texture quality; providing good geometry isolates what we're measuring.

---

**35. What does the ADD-S metric measure that IoU doesn't capture?**
- A) Color accuracy
- B) Pose accuracy through surface distance normalized by object diameter
- C) Training time
- D) Memory usage

**Answer: B**

**Input:** Need to evaluate 3D layout prediction, **Goal:** Measure pose accuracy appropriately, **Approach:** Use ADD-S (Average Distance of Distinguishable points - Symmetric), **Because:** The trade-off is what aspect to measure—IoU captures overlap but doesn't penalize rotation errors well; ADD-S directly measures how close predicted surfaces are to ground truth.

---

**36. Why do pipeline approaches (shape model + pose estimator) underperform joint SAM 3D?**
- A) They use more memory
- B) Errors compound across stages and shape/pose are estimated independently without mutual consistency
- C) They're newer
- D) They use different GPUs

**Answer: B**

**Input:** Two approaches: pipeline vs. joint generation, **Goal:** Best overall layout prediction, **Approach:** Compare separate shape+pose pipeline vs. joint prediction, **Because:** The trade-off is modularity vs. consistency—pipelines allow mixing different methods but lose the benefit of joint reasoning; SAM 3D's joint prediction ensures shape and pose are mutually consistent.

---

**37. What does the ICP-Rotation metric measure?**
- A) Image quality
- B) Rotation error in degrees after optimal alignment
- C) Color precision
- D) Training speed

**Answer: B**

**Input:** Predicted and ground truth meshes, **Goal:** Measure rotation accuracy specifically, **Approach:** ICP (Iterative Closest Point) alignment then measure residual rotation, **Because:** The trade-off is metric interpretability vs. completeness—raw rotation error is intuitive (degrees off) but requires alignment; ICP provides that alignment baseline.

---

**38. Why does SAM 3D use Elo ratings to track historical model improvements?**
- A) It's a gaming reference
- B) Elo provides a continuous scale where differences map to win probabilities
- C) It's faster to compute
- D) It uses less storage

**Answer: B**

**Input:** Need to track model improvement over time, **Goal:** Quantify progress meaningfully, **Approach:** Use Elo rating system, **Because:** The trade-off is interpretability vs. simplicity—raw win rates are hard to compare across different matchups; Elo provides a unified scale where 400 points = 10:1 odds, enabling fair comparison.

---

**39. What does the near-linear Elo scaling in Figure 10a demonstrate?**
- A) Random fluctuation
- B) Consistent improvement from cumulative training stages and data engine iterations
- C) Model degradation
- D) Hardware improvements

**Answer: B**

**Input:** Historical checkpoints over development, **Goal:** Validate the training approach, **Approach:** Track Elo over time, **Because:** The trade-off validated is development complexity vs. payoff—multi-stage training with data engine iteration is complex but the linear Elo improvement confirms each component contributes measurable gains.

---

**40. Why does Table 4 show diminishing returns for later training stages?**
- A) Bugs in the code
- B) Each stage addresses remaining errors, which become progressively harder to fix
- C) Less data
- D) Smaller models

**Answer: B**

**Input:** Cascaded training stages with metrics, **Goal:** Understand contribution of each stage, **Approach:** Ablation showing incremental improvements, **Because:** The trade-off is effort vs. marginal gain—early stages fix common errors for large improvements; later stages (Art-3DO, DPO) address subtle issues with smaller but still valuable gains.

---

### **Section 5: Technical Details and Applications (Questions 41-50)**

**41. Why does SAM 3D use VAE latent space rather than direct mesh prediction?**
- A) It's simpler
- B) Latent space enables efficient generative modeling and consistent structured representations
- C) It uses less memory
- D) It's faster at inference

**Answer: B**

**Input:** Need to generate complex 3D outputs, **Goal:** Efficient and high-quality 3D generation, **Approach:** Generate in VAE latent space then decode, **Because:** The trade-off is representation complexity vs. generation quality—direct mesh prediction is hard due to variable topology; VAE latents provide fixed-size, structured representations that flow models can generate effectively.

---

**42. What is the purpose of the sparse latent flow transformer in the Texture & Refinement model?**
- A) To increase model size
- B) To efficiently process only active voxels rather than the full volume
- C) To reduce quality
- D) To slow down training

**Answer: B**

**Input:** Coarse 64³ shape from Geometry model, **Goal:** Add geometric detail and texture efficiently, **Approach:** Sparse transformer operating on active voxels only, **Because:** The trade-off is computation vs. resolution—dense processing of high-res volumes is expensive; sparse attention only computes on occupied regions, enabling detail refinement at practical cost.

---

**43. Why does SAM 3D support both mesh and 3D Gaussian splat outputs?**
- A) To confuse users
- B) Different applications need different representations—meshes for editing, splats for novel view synthesis
- C) It was accidental
- D) To use more storage

**Answer: B**

**Input:** Single learned latent representation, **Goal:** Support diverse downstream applications, **Approach:** Two decoders sharing the same encoder/latent space, **Because:** The trade-off is universality vs. specialization—meshes are standard for graphics/robotics, splats excel at view synthesis; shared latents enable both without retraining.

---

**44. What enables SAM 3D to handle objects it has never seen before?**
- A) Memorization of all objects
- B) Objects are composed of parts and shapes learned during training that generalize compositionally
- C) Internet lookup
- D) User specification

**Answer: B**

**Input:** Novel objects not in training set, **Goal:** Generalize to unseen object categories, **Approach:** Learn compositional shape vocabulary, **Because:** The trade-off is specificity vs. generalization—memorizing exact objects wouldn't generalize; learning parts and shapes that compose allows reconstructing novel objects from familiar components.

---

**45. Why is mask-following important for SAM 3D's practical use?**
- A) It's not important
- B) Users can specify exactly which object to reconstruct in multi-object scenes
- C) It speeds up training
- D) It reduces memory

**Answer: B**

**Input:** Image with multiple objects, **Goal:** Reconstruct user-specified object, **Approach:** Condition on mask input, **Because:** The trade-off is automation vs. control—fully automatic methods might reconstruct the wrong object; mask conditioning gives users precise control over which object to 3Dfy.

---

**46. What role does CFG (Classifier-Free Guidance) play in SAM 3D?**
- A) Classification
- B) Strengthening conditioning signal for higher quality outputs aligned with input
- C) File formatting
- D) Memory management

**Answer: B**

**Input:** Conditional generation at inference, **Goal:** Outputs that closely match input image/mask, **Approach:** Apply CFG to balance unconditional and conditional predictions, **Because:** The trade-off is diversity vs. fidelity—pure conditional sampling might be noisy; CFG with weight 2.0 pushes outputs to more strongly match the conditioning.

---

**47. Why does the shortcut distillation initialize step-size embedder weights to zero?**
- A) Random choice
- B) Ensures the model initially behaves identically to the pre-distillation model
- C) To save memory
- D) To increase speed

**Answer: B**

**Input:** Trained model being distilled, **Goal:** Add shortcut capability without degrading existing quality, **Approach:** Zero-initialize new step-size parameters, **Because:** The trade-off is stability vs. flexibility—random initialization would disrupt learned behavior; zero init means d=0 recovers original model exactly, and shortcut capability is learned incrementally.

---

**48. What makes SAM 3D suitable for robotics applications?**
- A) It only works with robots
- B) Single-image input, joint shape+pose prediction, and sub-second inference after distillation
- C) It's very large
- D) It requires special hardware

**Answer: B**

**Input:** Robot needing 3D scene understanding, **Goal:** Enable real-time 3D perception from camera, **Approach:** Fast single-image 3D reconstruction with layout, **Because:** The trade-off is information vs. practicality—multi-view methods need robot movement; SAM 3D gives instant 3D understanding from any single camera frame at interactive speeds.

---

**49. Why does SAM 3D filter Iso-3DO pretraining data for geometry quality?**
- A) To reduce dataset size
- B) Poor quality meshes (degenerate shapes, outliers) can harm model learning even at scale
- C) To speed up training
- D) To change colors

**Answer: B**

**Input:** Large-scale 3D assets of variable quality, **Goal:** Effective pretraining, **Approach:** Rule-based filtering for geometry quality, **Because:** The trade-off is quantity vs. quality—more data usually helps but degenerate meshes (point-like structures, flat sheets) teach wrong priors; filtering removes harmful examples while keeping useful diversity.

---

**50. What is the significance of SAM 3D achieving ADD-S @ 0.1 of 77% compared to 2% for baselines on SA-3DAO?**
- A) It's a minor improvement
- B) It demonstrates SAM 3D's breakthrough capability for joint shape and layout prediction in real-world conditions
- C) It indicates worse performance
- D) It measures color accuracy

**Answer: B**

**Input:** Layout prediction on challenging real-world benchmark, **Goal:** Evaluate practical 3D scene understanding, **Approach:** ADD-S @ 0.1 threshold metric, **Because:** The trade-off this reveals is joint vs. pipeline approaches—2% for baselines means they essentially fail on real-world layout; 77% for SAM 3D demonstrates a qualitative leap in capability for visually grounded 3D reconstruction.

---

## Answer Key Summary

| Q# | Ans | Q# | Ans | Q# | Ans | Q# | Ans | Q# | Ans |
|----|-----|----|----|----|----|----|----|----|----|
| 1  | B   | 11 | B  | 21 | B  | 31 | B  | 41 | B  |
| 2  | B   | 12 | B  | 22 | B  | 32 | B  | 42 | B  |
| 3  | B   | 13 | B  | 23 | B  | 33 | B  | 43 | B  |
| 4  | B   | 14 | B  | 24 | B  | 34 | B  | 44 | B  |
| 5  | B   | 15 | B  | 25 | B  | 35 | B  | 45 | B  |
| 6  | B   | 16 | B  | 26 | B  | 36 | B  | 46 | B  |
| 7  | B   | 17 | B  | 27 | B  | 37 | B  | 47 | B  |
| 8  | B   | 18 | B  | 28 | B  | 38 | B  | 48 | B  |
| 9  | B   | 19 | B  | 29 | B  | 39 | B  | 49 | B  |
| 10 | B   | 20 | B  | 30 | B  | 40 | B  | 50 | B  |

---

## Study Tips

Each explanation follows the trade-off template to help you understand not just WHAT the answer is, but WHY it matters:

- **Input**: What situation or constraint are we starting with?
- **Goal**: What are we trying to achieve?
- **Approach**: What solution was chosen?
- **Because**: What trade-off or reasoning led to this choice?

Understanding trade-offs is key to deeply grasping research decisions!
