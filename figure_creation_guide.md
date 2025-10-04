# Figure Creation Guide for Journal Paper
## Using Your Existing 23 Figures Strategically

**Your Situation:** You have 23 high-quality figures from dissertation → Need to select 3-5 for main paper

**Strategy:** Choose figures that tell your core story, move rest to supplementary materials

---

## 📋 FIGURE SELECTION STRATEGY

### **Main Paper (3-5 figures MAX)**
Focus on figures that demonstrate:
1. ✅ Dataset quality and scale
2. ✅ Your novel architecture (TransformerV1)
3. ✅ Performance breakthrough vs baselines
4. ✅ Why ensemble works (complementarity)
5. ✅ (Optional) Overall achievement

### **Supplementary Materials (remaining ~18 figures)**
All other valuable figures:
- Detailed feature extraction methods
- All baseline comparisons
- Extended validation
- Additional analyses

---

## 🎯 RECOMMENDED MAIN PAPER FIGURES

Based on your 23 existing figures, here's the optimal selection:

### **MAIN FIGURE 1: Dataset Overview (NEW - Combine Figs 2-4)**

**Status:** CREATE NEW by combining existing figures  
**Your existing figures to merge:**
- Fig 2: `sequence_length_distribution.png`
- Fig 3: `amino_acid_distribution.png`  
- Fig 4: `class_balance_verification.png`

**Recommended Layout: 3-Panel**
```
┌────────────────────────────────────────────┐
│  A) Sequence Lengths  │  B) Class Balance  │
│     (Fig 2)           │     (Fig 4)        │
├────────────────────────────────────────────┤
│  C) Phosphorylation Site Distribution      │
│              (Fig 3)                       │
└────────────────────────────────────────────┘
```

**How to Create:**
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig = plt.figure(figsize=(12, 8))

# Panel A: Sequence lengths
ax1 = plt.subplot(2, 2, 1)
img1 = mpimg.imread('sequence_length_distribution.png')
ax1.imshow(img1)
ax1.axis('off')
ax1.set_title('A', fontweight='bold', fontsize=16, loc='left')

# Panel B: Class balance  
ax2 = plt.subplot(2, 2, 2)
img2 = mpimg.imread('class_balance_verification.png')
ax2.imshow(img2)
ax2.axis('off')
ax2.set_title('B', fontweight='bold', fontsize=16, loc='left')

# Panel C: Amino acid distribution (spans bottom)
ax3 = plt.subplot(2, 1, 2)
img3 = mpimg.imread('amino_acid_distribution.png')
ax3.imshow(img3)
ax3.axis('off')
ax3.set_title('C', fontweight='bold', fontsize=16, loc='left')

plt.tight_layout()
plt.savefig('Figure1_Dataset_Overview.pdf', dpi=300, bbox_inches='tight')
plt.savefig('Figure1_Dataset_Overview.png', dpi=300, bbox_inches='tight')
```

**NEW Caption (Condensed from originals):**

"**Figure 1. Dataset characteristics and quality.** (A) Protein sequence length distribution showing mean of 798 amino acids and median of 619 amino acids, with typical right-skewed eukaryotic pattern. (B) Class balance verification demonstrating 31,073 positive phosphorylation sites and 31,047 negative samples (balance ratio = 0.999). (C) Phosphorylation site distribution by amino acid type: serine (~25,000 sites, 80.6%), threonine (~5,500 sites, 17.7%), and tyrosine (~573 sites, 1.8%), consistent with known kinase specificity patterns. Total dataset: 62,120 samples across 7,510 human proteins."

**Word savings:** 3 separate captions (~150 words) → 1 combined caption (~85 words)

---

### **MAIN FIGURE 2: TransformerV1 Architecture**

**Status:** USE EXISTING (partial from Fig 11)  
**Your figure:** Fig 11 (left panel only - `images/trans.png`)

**What to do:**
1. **Extract ONLY the left panel** (TransformerV1 architecture)
2. **Remove TransformerV2** (right panel) - it goes to supplementary since it underperformed
3. **Enhance if needed:** Add dimension labels more clearly

**Why:** TransformerV1 is your main contribution (80.25% F1). TransformerV2 is interesting but secondary.

**Modified Caption:**

"**Figure 2. TransformerV1 architecture for phosphorylation site prediction.** The BasePhosphoTransformer architecture (8.4M parameters) processes protein sequences through the pre-trained ESM-2 encoder (facebook/esm2_t6_8M_UR50D), extracting contextual embeddings from a ±3 amino acid context window around each candidate site. Feature concatenation produces 2,240-dimensional representations (7 positions × 320 dimensions) that feed through a progressive three-layer classification head (2,240 → 256 → 64 → 1) with layer normalization, ReLU activation, and dropout regularization (p=0.3), culminating in sigmoid-activated phosphorylation probability prediction."

**Alternative:** If you want to keep both architectures for comparison, keep Fig 11 as-is but emphasize V1's superiority in caption.

---

### **MAIN FIGURE 3: Performance Comparison (NEW - Combine Figs 12-13 + add baselines)**

**Status:** CREATE NEW combining multiple figures  
**Your existing figures:**
- Fig 12: `f1_score_comparison.png` (feature types)
- Fig 13: `auc_score_comparison.png` (feature types)
- Fig 20: `confusion_matrix_v1_updated.png` (TransformerV1 results)

**Recommended Layout: Multi-panel showing progression**
```
┌────────────────────────────────────────────┐
│  A) F1 Score Comparison                    │
│  [Bar chart: ML baselines → Transformer    │
│   → Ensemble, with error bars]            │
├────────────────────────────────────────────┤
│  B) Confusion Matrix  │  C) ROC Curves     │
│     (TransformerV1)   │  (Key models)      │
└────────────────────────────────────────────┘
```

**Data to include in Panel A:**
```
Method                          F1 Score    95% CI
──────────────────────────────────────────────────
Best ML (CatBoost + PhysChem)   0.7803     [0.776-0.785]
TransformerV1                   0.8025     [0.798-0.807]
TransformerV2                   0.7898     [0.785-0.794]
Ensemble (Optimal Voting)       0.8160     [0.812-0.820]
```

**Python template:**
```python
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(14, 6))

# Panel A: F1 Score comparison
ax1 = plt.subplot(1, 3, (1, 2))  # Takes 2/3 width

methods = ['CatBoost\n+PhysChem', 'TransformerV1', 'TransformerV2', 
           'Ensemble']
f1_scores = [0.7803, 0.8025, 0.7898, 0.8160]
ci_lower = [0.776, 0.798, 0.785, 0.812]
ci_upper = [0.785, 0.807, 0.794, 0.820]
errors = [[f1 - ci_lower[i], ci_upper[i] - f1] 
          for i, f1 in enumerate(f1_scores)]

colors = ['steelblue', 'orange', 'lightcoral', 'green']
bars = ax1.bar(methods, f1_scores, color=colors, alpha=0.8, 
               edgecolor='black', linewidth=1.5)
ax1.errorbar(methods, f1_scores, 
             yerr=np.array(errors).T, fmt='none', 
             color='black', capsize=5, capthick=2)

# Add significance markers
ax1.plot([0, 1], [0.81, 0.81], 'k-', linewidth=1)
ax1.text(0.5, 0.815, '***', ha='center', fontsize=14)
ax1.plot([1, 3], [0.835, 0.835], 'k-', linewidth=1)
ax1.text(2, 0.840, '***', ha='center', fontsize=14)

ax1.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
ax1.set_ylim(0.75, 0.85)
ax1.set_title('A) Performance Comparison', fontsize=14, fontweight='bold')
ax1.axhline(y=0.80, color='red', linestyle='--', alpha=0.5, 
            label='80% Threshold')
ax1.legend()
ax1.grid(alpha=0.3, axis='y')

# Add value labels
for bar, score in zip(bars, f1_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.003,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel B: Confusion matrix
ax2 = plt.subplot(1, 3, 3)
img = mpimg.imread('images/confusion_matrix_v1_updated.png')
ax2.imshow(img)
ax2.axis('off')
ax2.set_title('B) TransformerV1\nTest Set Results', 
              fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('Figure3_Performance_Comparison.pdf', dpi=300, bbox_inches='tight')
```

**Caption:**

"**Figure 3. Performance comparison demonstrating state-of-the-art achievement.** (A) F1 score comparison across computational paradigms showing progressive improvement from traditional machine learning baseline (CatBoost with physicochemical features, 78.03%) through TransformerV1 breakthrough (80.25%, first to exceed 80% threshold, p < 0.001) to ensemble optimization (81.60%, p < 0.001 vs. TransformerV1). Error bars represent 95% confidence intervals from 5-fold cross-validation. Statistical significance indicated by *** (p < 0.001, McNemar's test). TransformerV2's lower performance (78.98%) demonstrates that architectural complexity without proper regularization degrades performance. (B) TransformerV1 confusion matrix on test set (n=10,122) showing balanced performance: 4,091 true positives, 4,017 true negatives, with nearly equal error distribution (1,044 false positives vs. 970 false negatives), indicating unbiased prediction suitable for practical applications."

---

### **MAIN FIGURE 4: Model Complementarity & Ensemble Justification**

**Status:** USE EXISTING with modifications  
**Your best options:**
- **Option A:** Fig 22 (`error_correlation_matrix.png`) - shows why ensemble works
- **Option B:** Fig 21 (`consensus_analysis.png`) - shows model agreement patterns
- **Recommendation:** Use Fig 22 (more quantitative, better justification)

**What to do:**
1. Use Fig 22 as main panel
2. **Optional:** Add small inset showing ensemble weight distribution (from Fig 15)

**Caption (if using Fig 22):**

"**Figure 4. Model error complementarity justifying ensemble integration.** Error correlation matrix across nine models reveals distinct architectural clustering patterns and diversity essential for ensemble effectiveness. Transformer models (V1, V2, V3, V4) exhibit high inter-correlation (r=0.62-0.69), sharing similar error patterns despite architectural variations. Traditional machine learning models show varied correlations (r=0.31-0.71), indicating diverse decision boundaries. Notably, the TPC-based model demonstrates consistently low correlations with transformers (r=0.30-0.38), capturing unique patterns missed by embedding-based approaches. This complementarity validates the ensemble strategy, as low error correlation between base models enables ensemble methods to correct individual model failures, resulting in 81.60% F1 score performance improvement."

---

### **MAIN FIGURE 5 (OPTIONAL): Achievement Summary**

**Status:** CREATE NEW or use Fig 23  
**Your figure:** Fig 23 (currently shows "INSERT" placeholder)

**Purpose:** Visual summary of your research impact - optional "hero shot"

**Recommended design if creating:**
```
┌────────────────────────────────────────────┐
│         PERFORMANCE PROGRESSION            │
│                                            │
│  Traditional ML    →    Transformer    →   │
│    (78.03% F1)          (80.25% F1)        │
│                                            │
│              ↘         ↙                   │
│                                            │
│            Ensemble                        │
│           (81.60% F1)                      │
│         STATE-OF-THE-ART                   │
└────────────────────────────────────────────┘
```

**Alternative:** Simple bar chart showing progression with milestones highlighted

**Caption:**

"**Figure 5. Research achievement summary.** Performance progression from traditional machine learning baseline (CatBoost with physicochemical features, 78.03% F1) through transformer-based breakthrough (TransformerV1, 80.25% F1, first method to exceed 80% threshold) to ensemble integration optimization (81.60% F1), establishing new state-of-the-art performance for phosphorylation site prediction. Each advancement demonstrates systematic improvement through: (1) optimal feature engineering and algorithm selection in ML phase, (2) adaptation of protein language models with appropriate architectural constraints in transformer phase, and (3) strategic model combination exploiting complementary error patterns in ensemble phase."

---

## 📎 SUPPLEMENTARY MATERIALS ORGANIZATION

### **All remaining figures go here (18 figures):**

#### **Supplementary Figure S1: Complete Methodology Pipeline**
- **Use:** Your Fig 1 (`images/pipeline_2.png`)
- **Why supplementary:** Too detailed for main paper, but valuable for reproducibility

#### **Supplementary Figures S2-S5: Data Splitting Details**
- **Use:** Your Fig 5 (`split_distribution.png`)
- **Why supplementary:** Important for reproducibility, but main paper just needs to state the strategy

#### **Supplementary Figures S6-S10: Feature Extraction Methods**
- **Use:** Your Figs 6-10 (AAC, DPC, TPC, Binary, PhysChem diagrams)
- **Why supplementary:** Educational/methodological detail, main paper just needs to define features briefly

#### **Supplementary Figure S11: TransformerV2 Architecture**
- **Use:** Your Fig 11 (right panel only - `images/trans.png`)
- **Why supplementary:** Didn't outperform V1, so relegated to "alternative approaches"

#### **Supplementary Figures S12-S14: Feature Performance Details**
- **Use:** Your Figs 14-17 (performance matrix, ensemble weights, feature importance)
- **Why supplementary:** Detailed breakdowns interesting but not essential for main story

#### **Supplementary Figures S15-S16: Training Dynamics**
- **Use:** Your Figs 18-19 (training curves for V1 and V2)
- **Why supplementary:** Implementation details, main paper just reports final performance

#### **Supplementary Figure S17: Model Consensus Analysis**
- **Use:** Your Fig 21 (`consensus_analysis.png`)
- **Why supplementary:** If you use Fig 22 in main paper, this provides additional perspective

**Organization template:**
```
supplementary_materials/
├── supplementary_figures.pdf
│   ├── Figure S1: Complete methodology pipeline
│   ├── Figure S2: Data splitting strategy details
│   ├── Figures S3-S7: Feature extraction methods
│   ├── Figure S8: TransformerV2 architecture
│   ├── Figures S9-S11: Feature performance analysis
│   ├── Figures S12-S13: Training dynamics
│   └── Figure S14: Consensus analysis
└── figure_captions_supplementary.docx
```

---

## ✂️ FIGURE MODIFICATION GUIDE

### **How to Extract/Combine Figures in Python**

#### **Option 1: Combine multiple PNGs into multi-panel figure**
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

# Create figure with custom grid
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Load and place images
ax1 = fig.add_subplot(gs[0, 0])
img1 = mpimg.imread('figure1.png')
ax1.imshow(img1)
ax1.axis('off')
ax1.set_title('A', fontweight='bold', fontsize=16, loc='left', pad=10)

ax2 = fig.add_subplot(gs[0, 1])
img2 = mpimg.imread('figure2.png')
ax2.imshow(img2)
ax2.axis('off')
ax2.set_title('B', fontweight='bold', fontsize=16, loc='left', pad=10)

ax3 = fig.add_subplot(gs[1, :])  # Spans both columns
img3 = mpimg.imread('figure3.png')
ax3.imshow(img3)
ax3.axis('off')
ax3.set_title('C', fontweight='bold', fontsize=16, loc='left', pad=10)

plt.savefig('combined_figure.pdf', dpi=300, bbox_inches='tight')
plt.savefig('combined_figure.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### **Option 2: Extract single panel from multi-panel figure**
```python
from PIL import Image

# Load image
img = Image.open('images/trans.png')
width, height = img.size

# Extract left half (TransformerV1 only)
left_panel = img.crop((0, 0, width//2, height))
left_panel.save('TransformerV1_architecture_only.png', dpi=(300, 300))
```

#### **Option 3: Add panel labels to existing figure**
```python
from PIL import Image, ImageDraw, ImageFont

img = Image.open('your_figure.png')
draw = ImageDraw.Draw(img)

# Add "A" label in top-left corner
font = ImageFont.truetype("arial.ttf", 60)  # Adjust size
draw.text((30, 30), "A", fill='black', font=font)

img.save('figure_with_label.png', dpi=(300, 300))
```

---

## 📊 FIGURE QUALITY CHECKLIST

### **Before Submission (For Each Main Figure):**

#### **Technical Quality:**
- [ ] Resolution: 300+ DPI minimum (check with `file properties` or `identify -verbose`)
- [ ] Format: PDF preferred, PNG acceptable, JPG avoid if possible
- [ ] Size: Reasonable file size (<10 MB per figure)
- [ ] Color: Colorblind-friendly palette (use tools like Color Oracle to test)
- [ ] Fonts: Readable at print size (minimum 8pt, prefer 10-12pt)
- [ ] Line weights: Visible (minimum 0.5pt)

#### **Content Quality:**
- [ ] Panel labels: Clear (A, B, C) in consistent location
- [ ] Axis labels: Present and readable
- [ ] Legend: Included if needed, not redundant with caption
- [ ] Error bars: Included for all performance metrics
- [ ] Statistical significance: Marked clearly (*, **, ***)
- [ ] Scale bars: Appropriate and labeled
- [ ] No unnecessary elements: Remove grid if not needed

#### **Caption Quality:**
- [ ] Self-contained: Can understand figure without reading text
- [ ] Structured: (A) describes panel A, (B) describes panel B, etc.
- [ ] Quantitative: Includes key numbers from figure
- [ ] Concise: 3-6 sentences typical
- [ ] Definitions: All abbreviations defined
- [ ] Statistics: Mentions significance, sample sizes, error bars

---

## 🎯 FINAL FIGURE SELECTION RECOMMENDATION

**For your Bioinformatics paper, I recommend these 4 main figures:**

### **Main Paper (exactly 4 figures):**

1. ✅ **Figure 1:** Dataset Overview (combine Figs 2-4)
   - Shows scale and quality
   - ~600 words saved by combining

2. ✅ **Figure 2:** TransformerV1 Architecture (Fig 11 left panel)
   - Your novel contribution
   - Core methodology visualization

3. ✅ **Figure 3:** Performance Comparison (new, combining Figs 12-13 + 20)
   - Shows breakthrough results
   - Demonstrates state-of-the-art achievement

4. ✅ **Figure 4:** Error Complementarity (Fig 22)
   - Justifies ensemble approach
   - Shows model diversity

**Optional 5th figure if space allows:**
- **Figure 5:** Achievement Summary (Fig 23, if you create it)

### **Supplementary Materials (~18 figures):**
- **All remaining figures** from your list (Figs 1, 5-10, 14-19, 21, 23)
- Organized by section (Methods, Results, Analysis)
- Each properly captioned and numbered (S1, S2, etc.)

---

## 💡 TIME-SAVING TIPS

### **Don't recreate from scratch:**
1. ✅ Your existing figures are publication-quality
2. ✅ Just combine/extract as needed
3. ✅ Focus on: selecting, combining, captioning
4. ❌ Don't: remake figures unless necessary

### **Priority order:**
1. **Week 1:** Select which 4-5 figures go in main paper
2. **Week 2:** Create combined figures (Figs 1 and 3)
3. **Week 2:** Write captions for main figures
4. **Week 3:** Organize supplementary figures
5. **Week 3:** Write supplementary captions

### **Quick caption writing:**
- **Start with existing captions** (you already have them!)
- **Condense** by removing methodological details
- **Focus on results and interpretation**
- **Add panel descriptions** (A), (B), (C)

---

## 📞 QUESTIONS TO ASK YOURSELF

Before finalizing figure selection:

1. **Can a reader understand my core contribution from these 4 figures alone?**
   - Yes: Good selection
   - No: Rethink which figures tell story best

2. **Do the figures build logically on each other?**
   - Fig 1: Here's quality data
   - Fig 2: Here's our method
   - Fig 3: Here are results
   - Fig 4: Here's why ensemble works

3. **Have I minimized redundancy?**
   - Don't show same information multiple ways in main paper
   - Move alternative visualizations to supplementary

4. **Are captions self-contained?**
   - Test: Give figure+caption to someone unfamiliar
   - Can they understand without reading paper?

---

## ✅ FIGURE PREPARATION ACTION CHECKLIST

### **Week 1 Tasks:**
- [ ] Review all 23 figures you have
- [ ] Decide: 4 or 5 figures in main paper?
- [ ] List files needed for each main figure
- [ ] Identify which need combining/extracting

### **Week 2 Tasks:**
- [ ] Create Figure 1 (combined dataset overview)
- [ ] Extract Figure 2 (TransformerV1 from Fig 11)
- [ ] Create Figure 3 (performance comparison)
- [ ] Select Figure 4 (complementarity - Fig 22)
- [ ] Write all main figure captions

### **Week 3 Tasks:**
- [ ] Organize remaining 18 figures for supplementary
- [ ] Renumber as S1, S2, S3, etc.
- [ ] Write/adapt supplementary captions
- [ ] Create supplementary figures PDF

### **Week 4 Tasks:**
- [ ] Final quality check all figures
- [ ] Verify all figures referenced in text
- [ ] Confirm figure numbering sequential
- [ ] Export at proper resolution for submission

---

**Remember:** Your figures already exist and are high quality. Your job is SELECTION and ORGANIZATION, not recreation. Focus your time on writing and polishing the text - that's where the real work is!

**You've got excellent figures. Now just choose wisely and combine strategically! 🎨**

### **Panel A: Sequence Length Distribution**
**Type:** Histogram with KDE overlay

**Python code template:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel A: Sequence lengths
ax = axes[0]
sns.histplot(data=protein_lengths, bins=50, kde=True, ax=ax, color='steelblue')
ax.set_xlabel('Protein Sequence Length (amino acids)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('A) Protein Length Distribution', fontsize=14, fontweight='bold')
ax.axvline(protein_lengths.mean(), color='red', linestyle='--', 
           label=f'Mean: {protein_lengths.mean():.0f}')
ax.axvline(protein_lengths.median(), color='orange', linestyle='--', 
           label=f'Median: {protein_lengths.median():.0f}')
ax.legend()
ax.grid(alpha=0.3)
```

**Key statistics to show:**
- Mean length (798 aa)
- Median length (619 aa)
- Range

### **Panel B: Class Balance**
**Type:** Bar chart with exact counts

**Python code template:**
```python
# Panel B: Class balance
ax = axes[1]
classes = ['Positive\nSites', 'Negative\nSites']
counts = [31073, 31047]
colors = ['#2ecc71', '#e74c3c']

bars = ax.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('B) Class Balance', fontsize=14, fontweight='bold')
ax.set_ylim(0, 35000)

# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add balance ratio
ax.text(0.5, 0.95, f'Balance Ratio: 0.999', 
        transform=ax.transAxes, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=11)
ax.grid(alpha=0.3, axis='y')
```

### **Panel C: Amino Acid Distribution**
**Type:** Stacked bar chart

**Python code template:**
```python
# Panel C: Amino acid distribution
ax = axes[2]
aa_types = ['Serine', 'Threonine', 'Tyrosine']
counts = [25000, 5500, 573]  # Approximate from your data
colors = ['#3498db', '#2ecc71', '#e74c3c']

bars = ax.bar(aa_types, counts, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('Number of Sites', fontsize=12)
ax.set_xlabel('Amino Acid Type', fontsize=12)
ax.set_title('C) Phosphorylation Site Distribution', fontsize=14, fontweight='bold')

# Add percentages
total = sum(counts)
for bar, count in zip(bars, counts):
    height = bar.get_height()
    percentage = (count/total) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=10)

ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Figure1_Dataset_Overview.pdf', dpi=300, bbox_inches='tight')
plt.savefig('Figure1_Dataset_Overview.png', dpi=300, bbox_inches='tight')
```

### **Caption Template**

"**Figure 1. Dataset characteristics and composition.** (A) Distribution of protein sequence lengths showing mean length of 798 amino acids (red dashed line) and median of 619 amino acids (orange dashed line). (B) Class balance verification demonstrating nearly perfect 1:1 ratio between positive phosphorylation sites (31,073 samples) and negative samples (31,047 samples), achieving balance ratio of 0.999. (C) Amino acid type distribution at phosphorylation sites showing predominance of serine (~25,000 sites, 80.6%), followed by threonine (~5,500 sites, 17.7%) and tyrosine (~573 sites, 1.8%), consistent with known kinase specificity patterns in human proteome."

---

## 🏗️ FIGURE 2: TRANSFORMERV1 ARCHITECTURE

### **Purpose**
Show your novel contribution clearly - how you adapted ESM-2 for phosphorylation prediction

### **Recommended Layout: Flow Diagram**

```
Input Sequence
      ↓
[ESM-2 Tokenizer]
      ↓
[ESM-2 Encoder: 8M parameters]
      ↓
Context Window Extraction (±3 positions)
      ↓
Feature Concatenation (7 × 320 = 2,240 dim)
      ↓
[FC Layer 1: 2240 → 256]
      ↓
[LayerNorm + ReLU + Dropout]
      ↓
[FC Layer 2: 256 → 64]
      ↓
[FC Layer 3: 64 → 1]
      ↓
[Sigmoid]
      ↓
Phosphorylation Probability
```

### **Design Tips**
- Use **boxes with rounded corners** for operations
- Use **different colors** for different types:
  - Blue: Pre-trained components (ESM-2)
  - Green: Your novel components
  - Orange: Output
- **Show dimensions** at each step
- **Include example**: Show actual sequence going through pipeline

### **Tools**
- **PowerPoint/Keynote:** Quick and flexible
- **Draw.io (diagrams.net):** Free, professional
- **Inkscape:** Free, vector graphics
- **BioRender:** For biological elements
- **Python (matplotlib):** Programmatic

### **Caption Template**

"**Figure 2. TransformerV1 architecture for phosphorylation site prediction.** The model processes input protein sequences through the ESM-2 tokenizer, generating token IDs compatible with the pre-trained ESM-2 encoder (facebook/esm2_t6_8M_UR50D, 8M parameters). The encoder produces 320-dimensional contextual embeddings for each amino acid position. For each candidate phosphorylation site, embeddings from a ±3 amino acid context window are extracted and concatenated (7 positions × 320 dimensions = 2,240 features). A three-layer classification head with progressive dimensionality reduction (2,240 → 256 → 64 → 1) processes these features through layer normalization, ReLU activation, and dropout (p=0.3) regularization. The final sigmoid activation produces phosphorylation probability. Total parameters: 8.4M."

---

## 📊 FIGURE 3: PERFORMANCE COMPARISON

### **Purpose**
Show that your method achieves state-of-the-art performance

### **Recommended Layout: 3-Panel Comparison**

```
┌──────────────────────────────────────────────────┐
│  A) F1 Score Comparison (Bar chart)              │
│     [ML baseline | TransformerV1 | Ensemble]    │
├──────────────────────────────────────────────────┤
│  B) ROC Curves         C) Precision