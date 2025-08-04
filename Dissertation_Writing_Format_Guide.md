# Dissertation Writing Format and Structure Guide
## University of Surrey MSc Requirements

Based on your University of Surrey dissertation template and academic requirements, this guide provides the complete formatting and structural specifications for your phosphorylation site prediction dissertation.

---

## 📋 **UNIVERSITY OF SURREY TECHNICAL SPECIFICATIONS**

### **Document Class and Layout**
```latex
\documentclass[11pt,a4paper,oneside]{report}
```

**Key Requirements:**
- **Font**: Times Roman (11pt body text)
- **Page Layout**: A4 paper, one-sided
- **Margins**: Top/Bottom: 1in, Left: 1.38in, Right: 0.79in (Microsoft Normal margins)
- **Line Spacing**: Custom spacing with forced paragraph skips
- **Header**: Author Name, MSc dissertation (ash grey color)

### **Typography Hierarchy**
```latex
% Chapter titles: 12pt bold, UPPERCASE
\titleformat{\chapter}[block]{\fontsize{12}{15}\bfseries}{\thechapter}{1em}{\MakeUppercase}

% Section titles: 12pt bold
\titleformat{\section}[block]{\fontsize{12}{15}\bfseries}{\thesection}{1em}{}

% Subsection titles: 12pt bold
\titleformat{\subsection}[block]{\fontsize{12}{15}\bfseries}{\thesubsection}{1em}{}

% Subsubsection titles: 12pt regular
\titleformat{\subsubsection}[block]{\fontsize{12}{15}}{\thesubsubsection}{1em}{}
```

### **Page Numbering System**
- **Roman numerals**: Preliminary pages (i, ii, iii...)
- **Arabic numerals**: Main content (1, 2, 3...)
- **No page numbers**: Title page
- **Header includes**: "Author Name, MSc dissertation" in ash grey

---

## 📚 **MANDATORY DOCUMENT STRUCTURE**

### **Preliminary Pages (Roman numerals)**
1. **Title Page** (no page number)
2. **Optional Dedication Page**
3. **Acknowledgments**
4. **Declaration of Originality** ✅ *Required*
5. **Word Count Page** ✅ *Required*
6. **Abstract** ✅ *Required*
7. **Table of Contents**
8. **List of Figures**
9. **List of Tables** (if applicable)

### **Main Content (Arabic numerals)**
Based on template structure:
1. **Chapter 1: Introduction** (`chapters/introduction.tex`)
2. **Chapter 2: Theory/Literature Review** (`chapters/theory.tex`)
3. **Chapter 3: Technical Implementation** (`chapters/technical.tex`)
4. **Chapter 4: Technical Results** (`chapters/technical2.tex`)
5. **Chapter 5: Conclusions** (`chapters/conclusions.tex`)

### **End Matter**
1. **Bibliography** (using natbib with square brackets, numbers)
2. **List of Publications** (if applicable)
3. **Appendices**

---

## 🎯 **CHAPTER-SPECIFIC FORMATTING REQUIREMENTS**

### **Chapter 1: Introduction**
**University Requirements:**
- Chapter title in UPPERCASE (automatic via template)
- Clear problem statement and research questions
- Contribution summary
- Dissertation structure outline

**Your Content Mapping:**
- **1.1**: Research Context and Motivation (biological significance, clinical importance)
- **1.2**: Problem Statement (computational need, dataset scale)
- **1.3**: Research Questions (4 primary questions from your work)
- **1.4**: Research Contributions (80.25% F1 breakthrough, comprehensive benchmarking)
- **1.5**: Dissertation Structure (chapter roadmap)

**Writing Style Requirements:**
- **Academic tone**: Formal but accessible
- **Present tense**: For established facts ("Protein phosphorylation represents...")
- **Past tense**: For your work ("This research achieved...")
- **Clear transitions**: Between sections and concepts

### **Chapter 2: Theory/Literature Review**
**University Requirements:**
- Comprehensive field coverage
- Critical analysis, not just summary
- Clear gap identification
- Proper citation density (3-5 per major point)

**Your Content Structure:**
- **2.1**: Biological Foundation (4-5 citations)
- **2.2**: Experimental Methods and Limitations (3-4 citations)
- **2.3**: Computational Prediction Evolution (8-10 citations)
- **2.4**: Feature Engineering in Protein Sequences (4-5 citations)
- **2.5**: Modern Deep Learning and Transformers (3-4 citations)
- **2.6**: Ensemble Methods (2-3 citations)
- **2.7**: Research Gaps and Opportunities

**Critical Analysis Approach:**
```
"While Method X achieves Y% accuracy (Author, Year), it suffers from Z limitation. 
This gap is addressed by our approach through [your innovation]."
```

### **Chapter 3: Methodology**
**University Requirements:**
- Sufficient detail for reproducibility
- Clear justification for choices
- Proper mathematical notation
- Algorithm descriptions

**Your Content Structure:**
- **3.1**: Dataset Preparation and Processing
- **3.2**: Feature Engineering Framework (5 feature types)
- **3.3**: Machine Learning Implementation (30+ combinations)
- **3.4**: Transformer Architecture Development (ESM-2 adaptation)
- **3.5**: Ensemble Method Implementation (6 approaches)
- **3.6**: Experimental Design and Evaluation

**Technical Writing Style:**
- **Precise language**: "The dataset was split using stratified sampling..."
- **Justified choices**: "CatBoost was selected due to its superior handling of categorical features..."
- **Reproducible details**: "All experiments used random seed 42 for reproducibility..."

### **Chapter 4: Results**
**University Requirements:**
- Clear presentation of findings
- Statistical significance testing
- Proper figure/table integration
- Honest reporting of limitations

**Your Content Structure:**
- **4.1**: Feature Analysis and Optimization (performance matrix, dimensionality reduction)
- **4.2**: Machine Learning Performance Analysis (30 combinations, statistical testing)
- **4.3**: Transformer Architecture Results (80.25% F1 breakthrough)
- **4.4**: Error Analysis and Model Complementarity (9-model analysis)
- **4.5**: Ensemble Method Performance (81.60% F1 achievement)
- **4.6**: Statistical Analysis and Significance Testing

**Results Presentation Style:**
- **Confidence intervals**: "F1 = 0.8025 ± 0.008 (95% CI)"
- **Statistical testing**: "Performance improvements were statistically significant (p < 0.001)"
- **Clear comparisons**: "TransformerV1 outperformed the best ML model by 2.22 percentage points"

### **Chapter 5: Discussion**
**University Requirements:**
- Interpretation of results
- Comparison with state-of-the-art
- Limitations acknowledgment
- Future work suggestions

**Your Content Structure:**
- **5.1**: Performance Interpretation and Biological Insights
- **5.2**: Methodological Contributions and Innovations
- **5.3**: Comparison with State-of-the-Art Methods
- **5.4**: Limitations and Challenges
- **5.5**: Practical Implications and Applications
- **5.6**: Future Research Directions
- **5.7**: Research Impact and Significance

---

## 📊 **FIGURES AND TABLES FORMATTING**

### **Figure Requirements**
**University Standards:**
- **High resolution**: 300+ DPI for final submission
- **Professional appearance**: Clear labels, readable fonts
- **Proper captions**: Descriptive, self-contained
- **Consistent style**: Same color scheme throughout

**Your Key Figures:**
1. **Dataset Overview**: Sample distribution and class balance
2. **Feature Performance Comparison**: Bar charts with confidence intervals
3. **ML Performance Heatmap**: 30 model-feature combinations
4. **Transformer Learning Curves**: Training dynamics visualization
5. **Error Analysis**: Confusion matrices and error patterns
6. **Ensemble Performance**: Improvement visualization

**Figure Caption Style:**
```
Figure X.Y: Comprehensive performance comparison across feature types. 
Error bars represent 95% confidence intervals from 5-fold cross-validation. 
Physicochemical features achieved the highest performance (F1=0.7803).
```

### **Table Requirements**
**University Standards:**
- **Caption above table**: Unlike figures, table captions go on top
- **Clear headers**: Bold or appropriately formatted
- **Consistent alignment**: Numbers right-aligned, text left-aligned
- **Statistical notation**: Proper ± symbols, significance markers

**Your Key Tables:**
1. **Literature Comparison**: State-of-the-art method comparison
2. **Feature Analysis Summary**: Complete performance matrix
3. **Hyperparameter Settings**: Optimal configurations
4. **Statistical Testing Results**: P-values and effect sizes
5. **Ensemble Method Comparison**: Detailed combination strategies
6. **Computational Requirements**: Resource usage analysis

**Table Format Example:**
```
Table X.Y: Machine Learning Performance Summary

Feature Type     | Model    | F1 Score      | Accuracy     | AUC
Physicochemical  | CatBoost | 0.7803±0.008 | 0.7821±0.007 | 0.8456±0.006
AAC             | CatBoost | 0.7656±0.009 | 0.7689±0.008 | 0.8342±0.007
...
```

---

## 🔍 **ACADEMIC WRITING STYLE GUIDELINES**

### **Tone and Voice**
**University Standards:**
- **Formal academic register**: Avoid colloquialisms
- **Third person**: "This research demonstrates..." not "I demonstrate..."
- **Objective language**: Present findings without emotional language
- **Precise terminology**: Use technical terms correctly and consistently

### **Citation Integration**
**University Requirements:**
- **Natbib format**: [1], [2], [3] (numerical, square brackets)
- **Strategic placement**: Support claims, don't interrupt flow
- **Critical engagement**: Analyze sources, don't just cite

**Citation Examples:**
```
Protein phosphorylation represents a critical cellular mechanism [1], with over 
200,000 human phosphosites identified through mass spectrometry studies [2,3]. 
However, experimental approaches suffer from reproducibility challenges, with 
52% of sites identified in only single studies [4].
```

### **Technical Writing Best Practices**
**Methodology Descriptions:**
- **Passive voice acceptable**: "The dataset was preprocessed using..."
- **Clear sequence**: Present methods in logical order
- **Sufficient detail**: Enable reproducibility without overwhelming

**Results Presentation:**
- **Active voice preferred**: "TransformerV1 achieved 80.25% F1 score"
- **Quantitative focus**: Always include numerical results
- **Statistical rigor**: Report confidence intervals and significance

**Discussion Analysis:**
- **Balanced assessment**: Acknowledge both strengths and limitations
- **Comparative context**: Position results within field standards
- **Forward-looking**: Connect to future research opportunities

---

## 📏 **SPECIFIC FORMATTING REQUIREMENTS**

### **Mathematical Notation**
**University Standards:**
- **Inline math**: Use $...$ for simple expressions
- **Display math**: Use \begin{equation}...\end{equation} for important formulas
- **Consistent symbols**: Define variables clearly
- **Professional appearance**: Use proper LaTeX mathematical formatting

**Examples from Your Work:**
```latex
The F1 score is calculated as:
\begin{equation}
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\end{equation}

Where precision and recall are defined for the positive class (phosphorylation sites).
```

### **Algorithm Descriptions**
**University Requirements:**
- **Pseudocode format**: Clear, structured presentation
- **Proper indentation**: Show algorithmic structure
- **Variable definitions**: Explain all symbols used

### **Code Integration**
**Best Practices:**
- **Minimal code**: Only essential algorithmic details
- **Pseudocode preferred**: Over actual implementation code
- **Supplementary material**: Full code in appendices or online repository
- **Clear comments**: Explain non-obvious steps

---

## 📖 **CHAPTER LENGTH GUIDELINES**

### **Word Count Distribution** (Total: ~15,000-20,000 words)
- **Introduction**: 2,000-3,000 words
- **Literature Review**: 4,000-5,000 words
- **Methodology**: 3,000-4,000 words
- **Results**: 3,000-4,000 words
- **Discussion**: 2,000-3,000 words
- **Conclusions**: 1,000-1,500 words

### **Content Density Guidelines**
**Introduction:**
- **Concise motivation**: 2-3 paragraphs biological importance
- **Clear problem statement**: 1-2 paragraphs computational need
- **Research questions**: 4 specific, answerable questions
- **Contribution preview**: 1 paragraph highlighting achievements

**Literature Review:**
- **Comprehensive coverage**: 22 papers strategically integrated
- **Critical analysis**: Not just summary, but evaluation
- **Gap identification**: Clear research opportunities
- **Theoretical foundation**: Support for your approach

**Methodology:**
- **Complete reproducibility**: Sufficient detail for replication
- **Justified choices**: Explain why decisions were made
- **Technical depth**: Appropriate for MSc level
- **Clear organization**: Logical flow from data to evaluation

**Results:**
- **Comprehensive presentation**: All major findings
- **Statistical rigor**: Proper significance testing
- **Visual integration**: Effective use of figures and tables
- **Honest reporting**: Include negative results

**Discussion:**
- **Insightful interpretation**: What results mean
- **Biological relevance**: Connect to domain knowledge
- **Methodological contributions**: Your innovations
- **Future directions**: Logical extensions

---

## 🎨 **VISUAL DESIGN STANDARDS**

### **Color Scheme Consistency**
**Recommended Palette:**
- **Primary**: Deep blue (#2E4057) for main elements
- **Secondary**: Orange (#FF8C00) for highlights
- **Success**: Green (#28A745) for positive results
- **Warning**: Gold (#FFC107) for attention
- **Neutral**: Grey (#6C757D) for supporting elements

### **Typography in Figures**
- **Font consistency**: Same family as document (Times/Arial)
- **Readable sizes**: Minimum 10pt in final figures
- **Clear labels**: Descriptive axis labels and titles
- **Professional appearance**: Consistent styling throughout

### **Layout Principles**
- **White space**: Don't overcrowd figures or tables
- **Alignment**: Consistent positioning and spacing
- **Hierarchy**: Clear visual organization
- **Accessibility**: Colorblind-friendly palettes

---

## ✅ **QUALITY ASSURANCE CHECKLIST**

### **Content Review**
- [ ] All research questions explicitly answered
- [ ] Statistical significance properly reported
- [ ] Limitations honestly acknowledged
- [ ] Future work logically connected
- [ ] Contributions clearly stated

### **Technical Review**
- [ ] All figures have descriptive captions
- [ ] Tables properly formatted with captions above
- [ ] Citations integrated smoothly into text
- [ ] Mathematical notation consistent
- [ ] References complete and properly formatted

### **Format Review**
- [ ] Chapter titles in UPPERCASE (automatic)
- [ ] Consistent font and spacing
- [ ] Proper page numbering (Roman/Arabic)
- [ ] All required preliminary pages included
- [ ] Professional appearance throughout

### **Language Review**
- [ ] Formal academic tone maintained
- [ ] Technical terms used correctly
- [ ] Clear and concise writing
- [ ] Proper grammar and punctuation
- [ ] Consistent terminology throughout

---

## 🚀 **WRITING WORKFLOW RECOMMENDATIONS**

### **Phase 1: Structure and Outline (Week 1)**
1. **Complete chapter outlines**: Detailed section planning
2. **Figure/table planning**: Identify all visual elements needed
3. **Citation mapping**: Distribute 22 papers across chapters
4. **Timeline establishment**: Realistic writing schedule

### **Phase 2: First Draft (Weeks 2-4)**
1. **Chapter-by-chapter writing**: Focus on content first
2. **Results integration**: Transfer from your experimental data
3. **Figure generation**: Create all visualizations
4. **Reference management**: Maintain proper citations

### **Phase 3: Revision and Polish (Weeks 5-6)**
1. **Content revision**: Strengthen arguments and clarity
2. **Format compliance**: Ensure University requirements met
3. **Figure refinement**: Professional appearance and consistency
4. **Language polish**: Academic tone and precision

### **Phase 4: Final Review (Week 7)**
1. **Complete proofreading**: Grammar, spelling, formatting
2. **Citation verification**: All references complete and accurate
3. **Quality assurance**: Final checklist completion
4. **Submission preparation**: PDF generation and final checks

---

## 📋 **SPECIFIC UNIVERSITY OF SURREY COMPLIANCE**

### **Mandatory Elements**
- ✅ **Declaration of Originality**: Required separate page
- ✅ **Word Count**: Separate page with exact count
- ✅ **Abstract**: Concise summary of work and findings
- ✅ **Bibliography**: Using natbib style with square brackets
- ✅ **Professional binding**: If physical submission required

### **Template Integration**
- **LaTeX template**: Use provided .tex structure
- **Style files**: Include all required style packages
- **Figure placement**: Use proper float environments
- **Reference style**: myabbrvnat bibliography style
- **Margin compliance**: Exact specifications followed

### **Submission Requirements**
- **Electronic format**: PDF with embedded fonts
- **Print format**: If required, professional binding
- **File naming**: Follow university conventions
- **Backup copies**: Maintain multiple secure copies
- **Timeline compliance**: Submit by university deadlines

---

**This comprehensive format guide ensures your exceptional research (80.25% F1 TransformerV1, 81.60% F1 ensemble) is presented in full compliance with University of Surrey requirements while maintaining the highest standards of academic writing and visual presentation. Your breakthrough achievements deserve a dissertation that meets the most rigorous academic standards!**