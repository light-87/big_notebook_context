# Journal Writing Style Guide
## How to Write Concisely for Bioinformatics Journal

**Purpose:** Transform dissertation prose (verbose, pedagogical) → journal article (concise, expert-focused)

---

## 🎯 CORE PRINCIPLE: Every Word Must Earn Its Place

**Journal readers are experts. They want information density, not hand-holding.**

---

## ✂️ THE CUTTING RULES

### **Rule 1: Kill Filler Phrases**

| ❌ Verbose (DON'T USE) | ✅ Concise (USE THIS) |
|------------------------|----------------------|
| "In order to" | "To" |
| "Due to the fact that" | "Because" |
| "It is important to note that" | [Delete entirely] |
| "It should be noted that" | [Delete entirely] |
| "It is worth mentioning that" | [Delete entirely] |
| "As a matter of fact" | [Delete entirely] |
| "At this point in time" | "Now" or "Currently" |
| "Despite the fact that" | "Although" |
| "For the purpose of" | "To" or "For" |
| "In the event that" | "If" |
| "Make use of" | "Use" |
| "Prior to" | "Before" |
| "Subsequent to" | "After" |
| "Take into consideration" | "Consider" |
| "With regard to" | "Regarding" or "About" |

### **Rule 2: Front-Load Information**

❌ **Buried Lead:**
"After conducting extensive experiments using various machine learning algorithms and neural network architectures across multiple feature representations over a period of several months, we ultimately discovered that transformer-based models significantly outperformed traditional approaches."

✅ **Direct:**
"Transformer models outperformed traditional machine learning by 2.22 percentage points (p < 0.001)."

### **Rule 3: Use Active Voice (When Clear)**

❌ **Passive:** "The dataset was split into training, validation, and test sets by us."

✅ **Active:** "We split the dataset into training, validation, and test sets."

**Exception:** Passive is OK when actor is unimportant:
- ✅ "Proteins were extracted from UniProt database" (who extracted is irrelevant)

### **Rule 4: Be Quantitative, Not Qualitative**

| ❌ Vague | ✅ Specific |
|----------|------------|
| "Good performance" | "80.25% F1 score" |
| "Significant improvement" | "2.22 percentage point increase (p < 0.001)" |
| "Large dataset" | "62,120 samples across 7,510 proteins" |
| "Recent study" | "Smith et al. (2023)" |
| "Many features" | "320-dimensional embeddings" |
| "High accuracy" | "95% CI: 79.85-80.65%" |

### **Rule 5: One Idea Per Sentence**

❌ **Too Dense:**
"TransformerV1, which utilizes the ESM-2 pre-trained protein language model with a context window of ±3 amino acids around the target site and employs a classification head consisting of three fully connected layers with dropout regularization, achieved 80.25% F1 score on the test set, representing a substantial improvement over previous methods."

✅ **Clear (2-3 sentences):**
"TransformerV1 achieved 80.25% F1 score on the test set. The architecture utilizes ESM-2 pre-trained embeddings with a ±3 amino acid context window. This represents a 2.22 percentage point improvement over the best machine learning baseline (p < 0.001)."

### **Rule 6: Eliminate Redundancy**

❌ **Redundant:**
"The results obtained from our experiments demonstrate and show that..."

✅ **Concise:**
"Results demonstrate that..." OR "Results show that..."

❌ **Redundant:**
"In summary, to summarize the main findings..."

✅ **Concise:**
"In summary..." OR "To summarize..."

---

## 📝 SECTION-SPECIFIC WRITING PATTERNS

### **ABSTRACT - The 250-Word Formula**

```
[Hook - 1 sentence, ~25 words]
Protein phosphorylation regulates cellular signaling and represents a major drug 
target, but computational prediction accuracy remains insufficient for clinical use.

[Gap - 1-2 sentences, ~30 words]
Current methods achieve 70-78% F1 scores and suffer from poor generalization due 
to data leakage and inadequate evaluation practices.

[Your Solution - 2-3 sentences, ~60 words]
We developed TransformerV1, adapting the ESM-2 protein language model with a 
±3 residue context window for phosphorylation site prediction. We evaluated 
performance using protein-based splitting across 62,120 sites in 7,510 human 
proteins, comparing traditional machine learning, transformer architectures, 
and ensemble integration strategies.

[Results - 2-3 sentences, ~70 words]
TransformerV1 achieved 80.25% F1 score (95% CI: 79.85-80.65%), exceeding the 
80% threshold for the first time in phosphorylation prediction. Ensemble 
integration improved performance to 81.60% F1 score. Error analysis revealed 
complementary patterns between machine learning and transformer approaches, 
with ensemble methods reducing false positives by 18% compared to individual models.

[Impact - 1-2 sentences, ~40 words]
Our approach establishes new state-of-the-art performance with rigorous 
evaluation addressing field reproducibility concerns. Software and models 
are freely available at https://github.com/[your-repo].

TOTAL: ~250 words
```

### **INTRODUCTION - The Funnel Pattern**

**Paragraph 1: The Big Picture (200 words)**
```
Start BROAD → Narrow to YOUR TOPIC

Sentence 1: Why does this problem exist/matter? (biological importance)
Sentence 2-3: Scale/prevalence of the problem
Sentence 4-5: Clinical/economic significance
Sentence 6-7: Current experimental limitations
Sentence 8: Why computational prediction is needed
```

**Example:**
"Protein phosphorylation regulates virtually all cellular processes, from signal transduction to cell cycle control. Over 200,000 phosphorylation sites exist in the human proteome, with dysregulation implicated in cancer, diabetes, and neurological disorders. [Continue with your narrative...]. Computational prediction offers a complementary approach to prioritize sites for experimental validation."

**Paragraph 2: Current State & Limitations (300 words)**
```
What EXISTS → What's MISSING

Sentence 1: Overview of computational approaches
Sentence 2-4: Traditional methods (features, ML)
Sentence 5-7: Recent advances (deep learning)
Sentence 8-10: KEY LIMITATION #1 (reproducibility)
Sentence 11-12: KEY LIMITATION #2 (performance ceiling)
Sentence 13-14: KEY LIMITATION #3 (evaluation practices)
```

**Paragraph 3: Your Contribution (300 words)**
```
Your SOLUTION → Paper STRUCTURE

Sentence 1-2: What you did (overview)
Sentence 3-5: Key innovation #1 (transformer adaptation)
Sentence 6-7: Key innovation #2 (rigorous evaluation)
Sentence 8-9: Key innovation #3 (ensemble integration)
Sentence 10-11: Main findings (headline results)
Sentence 12-14: Paper organization
```

### **METHODS - The Recipe Pattern**

**Each subsection follows this structure:**

```
1. Overview (1 sentence)
   "We constructed a balanced dataset from EPSD and UniProt databases."

2. Detailed procedure (main content)
   Step-by-step description with justification

3. Parameters/specifications (list or prose)
   Specific settings, hyperparameters, versions

4. Rationale (1-2 sentences)
   Why this choice? What alternatives were considered?
```

**Example - Dataset Section:**

✅ **Good (concise but complete):**
"We constructed a balanced dataset from the EPSD database (Lin et al., 2021) and UniProt. Positive samples comprised 31,073 experimentally validated phosphorylation sites across 7,510 human proteins. Negative samples were generated by randomly selecting non-phosphorylated serine, threonine, and tyrosine positions from the same proteins, maintaining a 1:1 class balance. Proteins were split 70:15:15 into training, validation, and test sets, ensuring no protein appeared in multiple partitions to prevent information leakage."

**Word count:** ~80 words (vs. 800 words in dissertation)

### **RESULTS - The "Show, Don't Tell" Pattern**

❌ **Process-focused (dissertation style):**
"We trained the TransformerV1 model using the training set and then evaluated its performance on the validation set. After achieving satisfactory results, we applied the trained model to the test set to obtain final performance metrics."

✅ **Findings-focused (journal style):**
"TransformerV1 achieved 80.25% F1 score on the test set (Table 1), significantly outperforming traditional machine learning baselines (78.03% F1 for CatBoost with physicochemical features; p < 0.001, McNemar's test)."

**Results Writing Formula:**
```
1. State finding first (with numbers)
2. Reference figure/table
3. Add statistical significance
4. Compare to baseline/alternative
5. Interpret briefly (1 sentence max)
```

### **DISCUSSION - The Interpretation Pattern**

**Paragraph Structure:**
```
Para 1: Performance interpretation
  - Sentence 1: Restate main achievement
  - Sentence 2-3: Why it matters (biological/clinical)
  - Sentence 4-5: Compare to field standards

Para 2: Biological insights
  - What patterns did the model learn?
  - Do they match known biology?
  - What's surprising/novel?

Para 3: Limitations (BE HONEST)
  - Dataset limitations
  - Methodological limitations  
  - Scope limitations
  - Address each in 1-2 sentences

Para 4: Future directions
  - Concrete next steps (not vague)
  - "We plan to..." or "Future work should..."
```

---

## 🔢 NUMBERS & STATISTICS - The Precision Rules

### **Always Include:**
- ✅ Sample sizes: n = 62,120
- ✅ Confidence intervals: F1 = 0.8025 (95% CI: 0.7985-0.8065)
- ✅ p-values: p < 0.001
- ✅ Effect sizes: +2.22 percentage points
- ✅ Statistical tests used: McNemar's test

### **Number Formatting:**
- Percentages: 80.25% (not 0.8025)
- Decimals: Use 2-4 decimal places consistently
- Large numbers: 62,120 or 62K (be consistent)
- Ranges: 70-80% (en dash) or "between 70% and 80%"

### **Reporting Performance:**

✅ **Complete reporting:**
"TransformerV1 achieved 80.25% F1 score (95% CI: 79.85-80.65%), 81.9% accuracy, 82.1% precision, 78.4% recall, and 0.873 AUC-ROC on the test set (n = 9,318)."

❌ **Incomplete reporting:**
"TransformerV1 performed well on the test set."

---

## 📖 CITATION INTEGRATION

### **Pattern 1: Attribution**
"ESM-2 protein language models capture evolutionary patterns through self-supervised learning (Lin et al., 2023)."

### **Pattern 2: Evidence**
"Current methods achieve 70-78% F1 scores (Smith et al., 2022; Jones et al., 2023)."

### **Pattern 3: Contrast**
"While previous work focused on handcrafted features (Brown et al., 2021), we leverage pre-trained representations."

### **Pattern 4: Agreement**
"Our findings align with recent observations that protein-based splitting prevents data leakage (Wang et al., 2023)."

### **Citation Density Guidelines:**
- Introduction: ~8-12 citations (establish field)
- Methods: ~3-5 citations (key methodologies)
- Results: ~2-3 citations (comparisons only)
- Discussion: ~10-15 citations (interpret, compare, contextualize)

---

## 🚫 WORDS & PHRASES TO AVOID

### **Vague Qualifiers**
❌ "quite", "very", "extremely", "fairly", "rather", "somewhat"

Instead: Use precise quantification or delete

### **Hedging (unless necessary)**
❌ "It appears that...", "It seems that...", "One might argue..."

✅ If your data shows it, state it confidently: "Results demonstrate..."

**Exception:** Hedge when appropriate:
- ✅ "These results suggest..." (when inferring mechanism)
- ✅ "This may indicate..." (when evidence is indirect)

### **Novelty Claims (be careful)**
❌ "For the first time ever..." (reviewers will check)
❌ "Revolutionary approach..." (let reviewers decide)
❌ "Groundbreaking method..." (too strong)

✅ "First to exceed 80% F1 score..." (if verifiably true)
✅ "Novel application of ESM-2..." (appropriate if accurate)

---

## ✅ SELF-EDITING CHECKLIST

After writing each section, ask:

**Clarity:**
- [ ] Can I understand this sentence on first reading?
- [ ] Is the main point clear?
- [ ] Did I define all acronyms on first use?

**Conciseness:**
- [ ] Can I remove any words without losing meaning?
- [ ] Did I eliminate filler phrases?
- [ ] Is each sentence doing useful work?

**Precision:**
- [ ] Are all numbers accurate?
- [ ] Did I cite sources for all claims?
- [ ] Are comparisons fair and clear?

**Flow:**
- [ ] Does this sentence connect logically to the previous one?
- [ ] Does this paragraph have a clear topic?
- [ ] Are transitions smooth?

---

## 🎯 THE "WORD BUDGET" APPROACH

**Think of your 5,000 words as a budget. Spend wisely!**

Example allocation:
- Abstract: 250 words (5%)
- Introduction: 900 words (18%)
- Methods: 1,700 words (34%)
- Results: 1,500 words (30%)
- Discussion: 500 words (10%)
- Conclusion: 150 words (3%)

**If you exceed budget in one section, you MUST cut from another.**

---

## 💡 QUICK REFERENCE: Dissertation → Journal Translation

| Dissertation Style | Journal Style |
|-------------------|---------------|
| "The purpose of this research is to..." | "We developed..." |
| "It is important to note that..." | [Delete, just state the fact] |
| "In the field of computational biology..." | "In computational biology..." |
| "Our results demonstrate clearly that..." | "Results demonstrate that..." |
| "As can be seen in Figure 1..." | "Figure 1 shows..." |
| "The model was trained for 10 epochs" | "We trained the model for 10 epochs" |
| Long paragraphs (200+ words) | Short paragraphs (100-150 words) |
| Pedagogical tone | Expert-to-expert tone |
| Background context | Minimal context |

---

**Remember:** You're not writing for beginners. You're writing for experts who want information density. Every word must advance understanding. When in doubt, CUT IT OUT!

---

*Use this guide while writing every section. Read it before starting each writing session!*