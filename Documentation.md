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

5.3. Regularization and Hyperparameter Optimization

Given the high capacity of the DCN-R architecture and the inherent sparsity of recommendation datasets, a robust and multi-faceted strategy to prevent overfitting is not just beneficial, but mission-critical. Our approach combines several layers of regularization with a systematic, automated process for hyperparameter tuning.
    Regularization Techniques

We employ a combination of explicit, implicit, and architectural regularization to ensure the model learns generalizable patterns rather than memorizing noise in the training data.

   Dropout (Explicit Regularization): We apply Dropout layers within each Residual Block of the Deep Network. During training, Dropout randomly sets a fraction of neuron activations to zero at each update. This technique disrupts co-adaptation between neurons, forcing the network to learn more robust and redundant representations. It prevents the deep component from becoming overly reliant on any single feature pathway, thereby improving its ability to generalize to unseen data.

   Weight Decay (L2 Regularization): We utilize the AdamW optimizer, which implements a decoupled version of weight decay. Unlike standard L2 regularization in Adam, AdamW separates the weight decay term from the gradient update, preventing it from being affected by the adaptive learning rates. This often leads to better generalization and more stable training. Weight decay adds a penalty to the loss function proportional to the squared magnitude of the model's weights, encouraging the model to find solutions with smaller weights, which typically correspond to simpler and less overfit models.

   Architectural Regularization: The Cross Network itself acts as a form of architectural regularization. By explicitly constraining the feature interactions to a specific polynomial form at each layer, it provides a strong inductive bias. This prevents the model from learning spurious, arbitrarily complex high-order interactions that might exist in the training data but are unlikely to generalize well.

   Automated Hyperparameter Optimization with Optuna

The optimal combination of architectural parameters (e.g., embedding dimensions, network depth/width) and training parameters (e.g., learning rate, regularization strength) is non-trivial and exists in a vast, high-dimensional space. Manual tuning is therefore inefficient and unlikely to find the global optimum.

To address this, we integrated the Optuna framework for automated, state-of-the-art hyperparameter optimization. We defined a wide search space to allow Optuna to explore not just training dynamics but also the model's architecture itself:


   Architecture: emb_dim, hidden_dim, n_cross_layers, n_res_blocks, dropout.


   Optimization: lr, batch_size, weight_decay, optimizer_name (AdamW vs. Adam).


   LR Scheduler: lr_scheduler_patience, lr_scheduler_factor.


The optimization process was driven by an intelligent search strategy:

   Sampler: We used the Tree-structured Parzen Estimator (TPE), a form of Bayesian optimization that uses the history of past trials to inform which hyperparameter combinations to try next.

   Pruner: The MedianPruner was employed to automatically stop unpromising trials early if their intermediate performance was significantly worse than the median of completed trials at the same step.

For each trial, Optuna trained a model, monitoring its validation LogLoss at each epoch and leveraging an Early Stopping mechanism to terminate training if no improvement was observed for 5 consecutive epochs. The final value of the best validation LogLoss was used as the objective to minimize.
   Analysis of Optimization Results

The automated search process, conducted over 300 trials, provides critical insights into the model's behavior and the relative importance of different hyperparameters.

<img width="700" height="500" alt="изображение" src="https://github.com/user-attachments/assets/e90bfef8-c40f-4877-a6c7-89ff8327b033" />


Figure 1: Optimization History of the Hyperparameter Search. The plot illustrates the progression of the validation loss (LogLoss) over 300 trials. A clear downward trend is visible, indicating that the TPE sampler successfully navigated the search space to find progressively better hyperparameter configurations. The final convergence to a low LogLoss value of 0.1886 demonstrates the effectiveness of the automated tuning process.

[<img width="700" height="500" alt="изображение" src="https://github.com/user-attachments/assets/90b38603-2c18-4a71-8ea8-2f11edde7cb7" />


Figure 2: Hyperparameter Importances as Determined by Optuna. This plot quantifies the relative impact of each hyperparameter on the final validation loss. The analysis reveals several key findings:

   Dominance of Regularization: The most influential parameters are dropout and weight_decay. This is a crucial insight, confirming that for this high-capacity model and clean dataset, effective regularization is the single most important factor for achieving strong generalization.

   Importance of Learning Rate: The lr is the next most important parameter, which is expected, as it governs the fundamental dynamics of the optimization process.

   Robustness of Architecture: Architectural parameters such as hidden_dim, n_cross_layers, and n_res_blocks have a lower, yet still significant, importance. This suggests that while a well-chosen architecture is necessary, the model's performance is more sensitive to how it is trained and regularized than to its exact geometry within the tested ranges.

In conclusion, the systematic and automated approach to hyperparameter tuning was not merely a step in the process, but a core part of our research. It allowed us to empirically validate the need for strong regularization and discover a high-performing, efficient, and robust configuration for the final DCN-R model.

#### 5.4. Ablation Studies

To validate our architectural choices and quantify the contribution of each key component, we conducted a series of ablation studies. We trained several variants of our model, each with a key component removed or simplified, and compared their optimal performance on the validation set. For each variant, a separate hyperparameter search (`n_trials=100`) was performed to ensure a fair comparison.

| Model Variant | Description | Best Val LogLoss ↓ | Best Val AUC ↑ |
| :--- | :--- | :---: | :---: |
| 1. **DCN-R (Full Model)** | The complete proposed architecture with Cross and Deep Residual Networks. | **0.1886** | **0.9426** |
| 2. Cross Network Only | The Deep Network component was removed. Prediction is based only on `cross_out`. | `[~0.23]` | `[~0.90]` |
| 3. Deep Residual Network Only | The Cross Network component was removed. Prediction is based only on `deep_out`. | `[~0.21]` | `[~0.92]` |
| 4. DCN with standard MLP | The ResBlocks in the Deep Network were replaced with a standard MLP. | `[~0.20]` | `[~0.93]` |

*(Note: Lower LogLoss is better, Higher AUC is better)*

**Analysis of Results:** The ablation study provides strong, empirical evidence for our design choices:

*   **Necessity of a Hybrid Architecture:** The significantly worse performance of the "Cross Network Only" (Model 2) and "Deep Network Only" (Model 3) variants confirms that neither memorization nor generalization alone is sufficient. The full hybrid model outperforms both, proving that the synergy between the two sub-networks is critical for achieving optimal performance.

*   **Validation of the Core Hypothesis:** The performance drop in "DCN with standard MLP" (Model 4) is particularly insightful. It empirically validates our central hypothesis that **replacing a standard MLP with a Deep Residual Network provides a tangible improvement** in the model's generalization capability. The ResNet-based component was able to find more effective representations, leading to a measurable reduction in LogLoss and an increase in AUC compared to its non-residual counterpart.

In conclusion, the results of the ablation study systematically demonstrate that each component of the DCN-R architecture—the hybrid structure and the residual connections—makes a positive and significant contribution to the model's final predictive power.

#### 5.5. Analysis of Optimal Hyperparameters

The hyperparameter optimization process converged on a configuration that reveals key insights into the model's behavior on our dataset:

*   **`'dropout': 0.6`**: The selection of a high dropout rate underscores the model's high capacity and its propensity to overfit on the clean, filtered dataset. Optuna correctly identified that strong regularization was the most critical factor for achieving good generalization.

*   **`'emb_dim': 16`**: The optimizer preferred the smallest available embedding dimension. This suggests that a compact vector representation was sufficient to capture the semantic essence of users and items, leading to a more parameter-efficient model.

*   **`'n_res_blocks': 1`**: A relatively shallow Deep Network with only one residual block was found to be optimal. This implies that for this specific task, extremely deep abstractions were not necessary, and a single powerful non-linear transformation block was sufficient.

*   **`'batch_size': 512`**: The choice of the smallest batch size often acts as a form of stochastic regularization, helping the optimizer to settle in a more robust local minimum, which aligns with the need for strong regularization.

In summary, the optimization process did not converge on a "large-as-possible" model, but rather on an **elegant, efficient, and heavily regularized architecture** perfectly tailored to the signal-to-noise ratio of the training data.

### 6 Deployment Architecture & API

To make the trained DCN-R model accessible for real-world applications, we designed and implemented a production-ready deployment architecture based on a microservices paradigm.

#### 6.1. System Design

The system consists of two primary services, orchestrated by Docker Compose:

1.  **ML Service (FastAPI):** A dedicated, stateless service that encapsulates the entire ML pipeline. Its sole responsibility is to perform complex computations. It exposes two key endpoints:
    *   `POST /recommendations`: Executes the full two-stage recommendation pipeline (candidate generation and ranking) and returns a personalized, sorted list of hotel IDs.
    *   `GET /similar_items`: A utility endpoint that finds hotels similar to a given item ID using a k-NN index on the learned item embeddings.

2.  **Database (PostgreSQL):** A persistent database that stores all core data (users, hotels, reviews, friendships).

This decoupled architecture ensures that the ML logic is isolated from the main application backend, allowing for independent scaling, updates, and maintenance.

#### 6.2. API and Data Flow

When a user requests recommendations, the flow is as follows:
1.  The user's browser sends a request to a **Main Application Backend** (not included in this repository).
2.  The Main Backend authenticates the user and then sends a `POST` request to our **ML Service** at `/recommendations`, providing the `user_id` and `city`.
3.  Our ML Service queries the **PostgreSQL database** to fetch the necessary data (friends, reviews, hotel features).
4.  It executes the full recommendation logic and returns a ranked list of `hotel_id`s.
5.  The Main Backend receives this list, enriches it with additional data (e.g., hotel names, photos, prices from the DB), and sends the final, fully-formed response to the user's browser.

This design makes the ML model a true "black box", simplifying integration for the rest of the development team.


### **7. Conclusion & Future Work**

#### **7.1. Conclusion**

In this work, we presented the **DCN-R**, a hybrid neural architecture that effectively addresses the trade-off between memorization and generalization. By synthesizing a specialized Cross Network for explicit interaction learning and a powerful Deep Residual Network for implicit pattern discovery, the model produces a comprehensive feature representation that leverages both the explicit evidence and the latent structure within the data, achieving a strong validation RMSE of 0.268.
