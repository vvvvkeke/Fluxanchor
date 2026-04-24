# 代谢通量预测误差指标总结

## 背景

在评估代谢通量预测模型（FluxAnchor / KinLLM / FBA）的准确性时，选择合适的误差指标至关重要。不同指标对"误差大小"的定义不同，会导致对同一份预测结果产生截然不同的视觉印象和结论。

---

## 三种指标对比

### 方案1：绝对误差 `abs(pred − true)`

**公式**

$$\text{score} = |pred - true| \quad \text{(mmol/gDW/h)}$$

**特点**

- 直接反映通量预测偏差的物理量（有单位）
- 颜色：白色 = 0 误差，深红 = 误差大
- 本项目数据中，最大绝对误差约为 2.5～3 mmol/gDW/h

**优点**

- 直观，生物学含义明确
- 大通量反应即使相对误差小，绝对误差仍会被如实反映

**缺点**

- **被大通量反应主导**：大通量反应（true = 100 mmol/gDW/h）差 3 个单位只偏了 3%，而小通量反应（true = 1 mmol/gDW/h）差 3 个单位却偏了 300%，两者在图中颜色深浅相近，极不公平
- 无法区分"高估"和"低估"（无方向性）

**示例计算**

| true | pred | abs 误差 | 实际情况 |
|------|------|---------|---------|
| 100  | 103  | **3.0** | 仅偏 3%，预测很准 |
| 1    | 4    | **3.0** | 偏了 300%，预测很差 |
| 0.001 | 0.032 | **0.031** | 颜色极浅，但实际偏了 32 倍 |

---

### 方案2：相对误差 `log2FC`（原始版本）

**公式**

$$\text{score} = \log_2\!\left(\frac{pred + \varepsilon}{true + \varepsilon}\right), \quad \varepsilon = 10^{-5}$$

**特点**

- 0 = 预测完美，+1 = 高估 2 倍，−1 = 低估 2 倍，±5 ≈ 32 倍误差
- 颜色：蓝 = 低估，白 = 准确，红 = 高估
- 再叠加列 Z-score 后变成"相对于其他样本的偏差程度"

**优点**

- 对所有反应一视同仁，无论通量大小，均以自身为基准评估
- 正负对称，天然区分高估与低估

**缺点（关键问题）**

- **对近零通量极度敏感**：当 `true ≈ 0`（但 > FLUX\_THRESHOLD = 0）时，分母极小，微小的绝对偏差会产生极大的 log2FC，热力图出现深色格子，但绝对误差几乎为零，在生物学上毫无意义
- 容易给审稿人造成"模型预测极差"的错误印象

**示例计算（为何 ±5 并不代表绝对误差大）**

| true | pred | abs 误差 | log2FC | 热力图颜色 |
|------|------|---------|--------|-----------|
| 0.001 | 0.032 | **0.031**（极小，可忽略）| **+5.0** | 深红 |
| 0.01  | 0.16  | **0.15**（很小）| **+4.0** | 较红 |
| 1.0   | 32.0  | **31.0**（较大）| **+5.0** | 深红 |
| 100   | 103   | **3.0**（更大）| **+0.04** | 几乎白色 |

> **结论**：同样是 log2FC = 5，可能来自绝对误差仅 0.031 的近零通量反应，也可能来自绝对误差高达 31 的大通量反应。热力图无法区分这两种情况。

---

### 方案3：通量加权的 log2FC（推荐）

**公式**

$$\text{score} = \log_2\!\left(\frac{pred + \varepsilon}{true + \varepsilon}\right) \times \frac{|true|}{|true| + \tau}$$

其中 $\tau$ 为数据集中所有非零真实通量绝对值的**中位数**（本项目约为数个 mmol/gDW/h）。

**加权系数的行为**

$$w = \frac{|true|}{|true| + \tau} = \begin{cases} \approx 0 & \text{当 } |true| \ll \tau \text{（近零通量，误差被压制）} \\ = 0.5 & \text{当 } |true| = \tau \\ \approx 1 & \text{当 } |true| \gg \tau \text{（大通量，保留完整 log2FC）} \end{cases}$$

**示例计算（τ = 1 mmol/gDW/h）**

| true | pred | log2FC | weight $w$ | **加权 score** | 解读 |
|------|------|--------|-----------|--------------|------|
| 0.001 | 0.032 | +5.0 | 0.001/1.001 = **0.001** | **+0.005** | 近零通量，误差被压至接近 0 |
| 0.01  | 0.16  | +4.0 | 0.01/1.01 = **0.010**  | **+0.040** | 小通量，误差大幅压制 |
| 1.0   | 32.0  | +5.0 | 1.0/2.0 = **0.500**    | **+2.500** | 中等通量，保留一半误差 |
| 10    | 320   | +5.0 | 10/11 = **0.909**      | **+4.545** | 大通量，误差基本保留 |
| 100   | 103   | +0.04 | 100/101 = **0.990**    | **+0.040** | 大通量但相对误差本来就小 |

**优点**

- 同时兼顾相对误差和绝对通量量级：只有当通量本身足够大时，相对误差才会被充分计入
- 颜色含义与原始 log2FC 完全一致（蓝/白/红），对审稿人友好
- τ 参数物理含义明确（通量中位数），可报告


---

## 三方案综合对比

| 指标 | 有无方向 | 量纲 | 小通量近零爆炸 | 大通量主导 | 审稿人友好度 |
|------|---------|------|--------------|-----------|------------|
| abs(pred − true) | 无 | mmol/gDW/h | 不爆炸，但被掩盖 | **是** | ★★★★ |
| log2FC | 有 | 无 | **严重** | 否 | ★★ |
| **Weighted log2FC** | **有** | **无** | **被 τ 压制** | **否** | **★★★★★** |

---

## 建议用法

1. **主图**：使用 **Weighted log2FC**，向审稿人展示有生物学意义的相对误差
2. **补充图**：加入 **abs(pred − true)** 图，明确说明绝对误差范围（最大 2~3 mmol/gDW/h），打消审稿人对"误差极大"的顾虑
3. **方法说明**中明确写出 τ 的取值及计算方式，并给出加权公式

---

## 附：对审稿人的解释模板

> The heatmap displays flux-weighted log2 fold change (log2FC), defined as:
>
> $$\text{score} = \log_2\!\left(\frac{v_{pred}}{v_{true}}\right) \times \frac{|v_{true}|}{|v_{true}| + \tau}$$
>
> where $\tau$ is the median absolute flux across the dataset. This weighting suppresses apparent errors in near-zero flux reactions, whose absolute deviations are biologically negligible (< 0.1 mmol/gDW/h), while preserving the full relative error signal for high-flux reactions that drive cellular phenotype. As shown in Supplementary Figure X (absolute error heatmap), the maximum absolute prediction error across all conditions is within 2–3 mmol/gDW/h.
