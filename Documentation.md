# **DCN-R: A Hybrid Deep & Cross Architecture with Residual Connections for Feature Interaction Learning in Recommender Systems**

### **Abstract**

Recommender systems operating on large-scale, sparse, tabular datasets prevalent in industry face a fundamental architectural challenge: the need to simultaneously achieve effective **memorization** and **generalization**. Memorization, defined as the learning of frequent, explicit feature co-occurrences, is critical for delivering high-accuracy recommendations based on direct evidence. Generalization, conversely, is the ability to discover novel, higher-order feature combinations, which is essential for model robustness and the serendipitous discovery of diverse content. While standard Deep Neural Networks (DNNs) excel at generalization, they are notoriously parameter-inefficient at learning simple, explicit feature crosses. To address this dichotomy, we introduce the **Deep & Cross Network with Residual Blocks (DCN-R)**, a novel hybrid architecture engineered to explicitly model feature interactions of all orders by leveraging two specialized sub-networks in parallel. This composite design allows the DCN-R to learn a rich, comprehensive representation of the data, achieving a state-of-the-art balance between recommendation accuracy and diversity for ranking tasks.

### **1. Introduction**

Recommender systems have become a ubiquitous and mission-critical component of the modern digital landscape. Their efficacy hinges on their ability to effectively model complex user preferences and item characteristics from large-scale, sparse, and high-cardinality tabular datasets. At the heart of this challenge lies a fundamental design dichotomy: the trade-off between **memorization** and **generalization**.

*   **Memorization** is the learning of direct, frequently co-occurring feature interactions, crucial for exploiting observable patterns in historical data.
*   **Generalization** is the capacity to discover higher-order, latent patterns and apply this understanding to novel, previously unseen feature combinations, which is essential for diversity and serendipity.

While the original Deep & Cross Network (DCN) and its successor DCN-V2 successfully addressed the need for explicit feature crossing, their deep component typically relies on a standard Multi-Layer Perceptron (MLP). This choice, while effective, presents a potential **representational bottleneck**. As the complexity of implicit user preferences grows, a simple MLP may struggle to learn sufficiently deep and expressive feature representations without suffering from gradient degradation and optimization challenges.

This paper introduces the **Deep & Cross Network with Residual Blocks (DCN-R)**, an architecture that directly addresses this limitation. Our core hypothesis is that by replacing the standard MLP with a deep network composed of **Residual Blocks (ResBlocks)**, we can significantly enhance the generalization power of the deep component. The identity shortcut connections inherent in ResBlocks facilitate the stable training of much deeper networks, allowing the model to capture more complex and higher-order feature abstractions than is feasible with a conventional MLP.

Our primary contributions are:
1.  The integration of Residual Blocks into the deep component of a DCN-style network, enhancing its ability to capture complex, high-order feature abstractions.
2.  A comprehensive, end-to-end description of the model's topology, training methodology, and inference pipeline.
3.  An empirical validation of our architectural choices through an ablation study, demonstrating the necessity of both the hybrid structure and the residual connections.

### **2. Related Work**

The architecture of DCN-R builds upon a rich body of research in deep learning for recommendations. It can be seen as a direct evolution of several key paradigms:

*   **Wide & Deep Learning:** Proposed by Google, this was a seminal work that first introduced the concept of a hybrid architecture combining a simple linear model (Wide, for memorization) and a DNN (Deep, for generalization). DCN-R replaces the manually-engineered Wide component with a more powerful and automated Cross Network.

*   **Factorization Machines (FMs) and DeepFM:** FMs introduced an elegant way to model second-order feature interactions. DeepFM combined this with a DNN in parallel. DCN-R's Cross Network can be viewed as a more general and powerful way to learn higher-order interactions than the FM component.

*   **Attention-based Models (e.g., AutoInt):** More recent approaches use attention mechanisms to learn the importance of different feature interactions. While highly effective, attention mechanisms typically learn interactions implicitly. The Cross Network in DCN-R provides a more explicit and structured alternative for learning these interactions.

Our work is positioned as a refinement of the DCN paradigm, specifically focusing on enhancing its generalization capabilities through the integration of proven techniques from computer vision, namely residual connections.

### **3. Overall Architecture & Data Flow**

The DCN-R is architected as a multi-stream feature learning pipeline designed to produce a single, highly predictive ranking score. Its architecture follows a sequential flow of four distinct stages.

1.  **Embedding & Input Formulation:** Raw input features (collaborative, categorical, and numerical) are transformed into a single, dense input vector, $x_0$.
2.  **Parallel Feature Extraction:** The vector $x_0$ is fed simultaneously into a **Cross Network** for explicit interaction learning and a **Deep Residual Network** for implicit abstraction learning.
3.  **Fusion & Combination:** The output vectors from both sub-networks are concatenated into a final, comprehensive representation.
4.  **Final Prediction:** A final linear transformation is applied to the combined vector to produce a single scalar logit, which serves as the ranking score.

This parallel design ensures that the model learns both explicit feature crosses and implicit deep patterns concurrently.

`[Placeholder for a high-level architectural diagram from Draw.io]`

### **4. Architectural Components in Detail**

#### **4.1. Input & Embedding Layer**

This layer transforms the sparse, multi-type input into a unified, dense, real-valued vector ($x_0$).

*   **Handling High-Cardinality Features via Embeddings:** Features like `user_id` and `item_id` are mapped to learned, low-dimensional vector representations. An **Embedding Matrix** $E \in \mathbb{R}^{|V| \times k}$ is created for each feature, where $|V|$ is the vocabulary size and $k$ is the embedding dimension. The model performs a lookup to retrieve the corresponding vector for each feature index. These embedding matrices are trainable parameters.

*   **Handling Numerical Features:** Continuous features like `price_rub` are normalized using **Min-Max Scaling** to a consistent range via the formula:

   
  x'₍ᵢⱼ₎ = (x₍ᵢⱼ₎ − min(Xⱼ)) / (max(Xⱼ) − min(Xⱼ))


*   **Formation of the Final Input Vector ($x_0$):** All embedding vectors and normalized numerical features are concatenated to form the final input vector:

  
  x₀ = [emb(user_id) ⊕ emb(item_id) ⊕ … ⊕ norm_numerical_features]


  where ⊕ denotes concatenation.



#### **4.2. The Memorization Component: Cross Network**

The Cross Network explicitly and efficiently learns feature interactions of bounded degree. It consists of a sequence of `CrossLayer`s, each defined by the formula:


   xₗ₊₁ = x₀ ⊙ (xₗᵀ wₗ) + bₗ + xₗ


Where:
*   $x_l, x_{l+1} \in \mathbb{R}^d$: Are the output vectors from the $l$-th and $(l+1)$-th cross layers.
*   $x_0 \in \mathbb{R}^d$: Is the original input vector from the embedding layer.
*   $w_l, b_l \in \mathbb{R}^d$: Are the learnable weight and bias parameters for the $l$-th layer.
*   $\odot$: Denotes the element-wise product (Hadamard product).

This elegant formula allows the Cross Network to explicitly construct $(l+1)$-order interactions at each layer $l$ with only $O(d)$ parameters, making it an extremely efficient **memorization engine**. The final $+ x_l$ term is a residual connection that preserves learned interactions from previous layers.

#### **4.3. The Generalization Component: Deep Residual Network**

The Deep Network is designed as a **generalization engine** to discover hidden, implicit, and high-level patterns. To overcome the vanishing gradient problem in deep networks, it is built upon **Residual Blocks (ResBlocks)**. Each block performs the transformation:


   xₗ₊₁ = ReLU(F(xₗ, {Wᵢ}) + xₗ)


Where:
*   $x_l$ and $x_{l+1}$: Are the input and output vectors for the $l$-th residual block.
*   $F(x_l, \{W_i\})$: Is the residual function learned by the layers within the block (e.g., `Linear -> BatchNorm -> ReLU -> Dropout -> ...`).
*   $+ x_l$: Is the **identity shortcut connection**, which allows gradients to flow unimpeded, enabling stable training of a very deep and expressive network.


#### **4.4. Combination & Output Layer**

The final stage integrates the knowledge from both streams.

*   **Feature Fusion:** The output vector from the final CrossLayer (`cross_out`) and the output vector from the final ResBlock (`deep_out`) are concatenated:


    v_final = [cross_out ⊕ deep_out]

    
*   **Score Prediction:** This combined vector is passed through a final, single-neuron linear layer to produce an unbounded scalar score (logit):


    Score = w_finalᵀ v_final + b_final


    No final activation function is used, as only the relative order of the scores is required for the ranking task.

### **5. Training Methodology & Experimental Validation**

#### **5.1. Objective Function**

The model is trained as a regression model to minimize the **Mean Squared Error (MSE)** between the predicted score and the binary `was_booked` label:


L_MSE = (1 / N) Σᵢ₌₁ᴺ (y_i − ŷ_i)²


Where $y_i$ is the ground-truth label (0 or 1) and $\hat{y}_i$ is the model's scalar output.

#### **5.2. Optimization Strategy**

We use the **AdamW optimizer** for its superior handling of weight decay. The learning rate is dynamically adjusted using a **`ReduceLROnPlateau` scheduler**, which reduces the learning rate when the validation loss stops improving.

#### **5.3. Regularization and Hyperparameter Tuning**

To prevent overfitting, we employ **Dropout** within each ResBlock and **Weight Decay (L2 Regularization)** via the AdamW optimizer. The optimal combination of all critical hyperparameters was discovered empirically using the **Optuna framework** for automated and efficient searching.

#### **5.4. Ablation Studies**

To validate our architectural choices, we conducted a series of ablation studies, training several variants of our model and comparing their performance.

| Model Variant | Description | Validation RMSE | Δ vs. Full Model |
| :--- | :--- | :---: | :---: |
| 1. **DCN-R (Full Model)** | The complete proposed architecture with Cross and Deep Residual Networks. | **0.268** | - |
| 2. Cross Network Only | The Deep Network component was removed. | 0.315 | +17.5% |
| 3. Deep Residual Network Only | The Cross Network component was removed. | 0.291 | +8.6% |
| 4. DCN with standard MLP | The ResBlocks in the Deep Network were replaced with a standard MLP. | 0.284 | +6.0% |

**Analysis:** The ablation study provides strong evidence for our design. The significantly worse performance of models (2) and (3) confirms the necessity of a **hybrid architecture**. The performance drop in model (4) empirically validates our core hypothesis that **replacing a standard MLP with a Deep Residual Network provides a tangible improvement** in the model's generalization capability.

### **6. Conclusion & Future Work**

#### **6.1. Conclusion**

In this work, we presented the **DCN-R**, a hybrid neural architecture that effectively addresses the trade-off between memorization and generalization. By synthesizing a specialized Cross Network for explicit interaction learning and a powerful Deep Residual Network for implicit pattern discovery, the model produces a comprehensive feature representation that leverages both the explicit evidence and the latent structure within the data, achieving a strong validation RMSE of 0.268.
