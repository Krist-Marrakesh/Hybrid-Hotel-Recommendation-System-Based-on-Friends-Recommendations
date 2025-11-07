# **DCN-R: A Hybrid Deep & Cross Architecture with Residual Connections for Feature Interaction Learning in Recommender Systems**

### **Motivation and Problem Statement**

In the travel industry, choosing a hotel is often driven not only by price or photos but also by trust in the source of recommendations. Official reviews tend to feel impersonal, and algorithmic suggestions often lack a sense of human context.

At the same time, we know that recommendations from friends and acquaintances are perceived as far more reliable and relevant. However, most booking platforms still lack a convenient way to collect and use such socially informed feedback. Travelers end up asking their friends directly, searching for advice in public forums, or relying on random viral videos on social media.
I want to provide users with socially personalized recommendations to increase engagement, trust, and booking conversion.

"Friends Recommend" is a feature concept designed to solve this gap: it enables a traveler to receive personalized hotel recommendations based on the experience and reviews of their social connections. The functionality would allow users to:
Discover hotels their friends have actually stayed in.
Read fresh, trustworthy, and personally relevant reviews.

### *Functional Goal*
We set out to build a prototype of the “Friends Recommend”  — a system that lets a user find hotels visited by their friends, followers, or colleagues and view their feedback.

### *User Stories*:
*As a traveler, I want to see hotel recommendations from my friends so that I can trust their quality and relevance.*

*As a hotel owner, I want satisfied guests to easily recommend my hotel to their friends.*

Since no real social-graph dataset was available, we decided to synthetically generate one and test different architectures for learning personalized recommendations. This led us to design and evaluate the Deep & Cross Network with Residual Blocks (DCN-R) — a hybrid model aimed at capturing both explicit friend-based interactions and deep latent patterns in user–item behavior.


### **Abstraction**

In industrial recommendation systems, data is typically huge, sparse, and tabular—and this creates a key challenge: the need to simultaneously achieve effective **memorization** and **generalization**. Memorization, defined as the learning of frequent, explicit feature co-occurrences, is critical for delivering high-accuracy recommendations based on direct evidence. Generalization, conversely, is the ability to discover novel, higher-order feature combinations, which is essential for model robustness and the serendipitous discovery of diverse content. While standard Deep Neural Networks (DNNs) excel at generalization, they are notoriously parameter-inefficient at learning simple, explicit feature crosses. To address this dichotomy, we introduce the **Deep & Cross Network with Residual Blocks (DCN-R)**, a novel hybrid architecture engineered to explicitly model feature interactions of all orders by leveraging two specialized sub-networks in parallel. This composite design allows the DCN-R to learn a rich, comprehensive representation of the data, achieving a state-of-the-art balance between recommendation accuracy and diversity for ranking tasks.

### **1. Introduction**

Inasmuch as recommender systems have become a ubiquitous and mission-critical component of the modern digital landscape, their efficacy hinges on their ability to effectively model complex user preferences and item characteristics from large-scale, sparse, and high-cardinality tabular datasets. So, at the heart of this challenge lies a fundamental design dichotomy: the trade-off between memorization and generalization.
Let's take a closer look at what's going on

Memorization is the learning of direct, frequently occurring feature interactions, critical to exploiting observed patterns in historical data, while Generalization is the ability to discover hidden, higher-order patterns and apply this understanding to new, previously unseen combinations of features, critical to diversity and randomness.
While the original Deep & Cross Network (DCN) and its successor DCN-V2 successfully addressed the need for explicit feature crossing, their deep component typically relies on a standard Multi-Layer Perceptron (MLP). But, while effective, presents a potential **representational bottleneck**. As the complexity of implicit user preferences grows, a simple MLP may struggle to learn sufficiently deep and expressive feature representations without suffering from gradient degradation and optimization challenges.

This paper introduces the **Deep & Cross Network with Residual Blocks (DCN-R)**, an architecture that directly addresses this limitation. Our core hypothesis is that by replacing the standard MLP with a deep network composed of **Residual Blocks (ResBlocks)**, we can significantly enhance the generalization power of the deep component. Thanks to the *Skip connections* inherent in ResBlocks, we can achieve stable training of much deeper networks, allowing the model to capture more complex and high-level feature abstractions than is possible with a conventional MLP.

Our primary contributions are:
1.  The integration of Residual Blocks into the deep component of a DCN-style network, enhancing its ability to capture complex, high-order feature abstractions.
2.  Develop a comprehensive, end-to-end description of the model architecture, training methodology, and inference pipeline that ensures reproducibility and transparency of experiments.
3.  An empirical validation of our architectural choices through an ablation study, demonstrating the necessity the of both the hybrid structure and residual connections in achieving optimal results.

### **2. Related Work**

The architecture of DCN-R builds upon a rich lineage of research in deep learning for recommendation systems, unifying several influential paradigms into a coherent framework.

Early progress in this direction was marked by **Wide & Deep Learning**, proposed by Google — a seminal work that introduced the idea of hybrid architectures combining a simple linear component (Wide, for memorization) with a deep neural network (Deep, for generalization). This formulation established the foundation for balancing explicit feature interactions with implicit representation learning.

Building upon research developments popularized by *Google*, Factorization Machines (FMs) introduced an elegant and computationally efficient mechanism for modeling second-order feature interactions. Their neural extension, DeepFM, combined the strengths of FMs and DNNs in parallel, enabling the model to jointly learn both low- and high-order interactions. However, these architectures were still limited in their ability to systematically model higher-order cross-feature structures.

To address these limitations, the **Deep & Cross Network (DCN)** introduced a Cross Network component — a learnable and more generalized mechanism for feature crossing, effectively replacing the manually engineered Wide component with an automated alternative. Subsequent models, such as AutoInt, further explored attention-based approaches, leveraging self-attention to learn the relative importance of different feature combinations. While highly expressive, attention mechanisms tend to model such interactions implicitly, often at the cost of interpretability and structural control.

Our work refines the DCN paradigm by enhancing its generalization capabilities through the integration of residual connections — a proven architectural technique widely adopted in computer vision. By bridging the advantages of explicit cross-feature modeling and stable gradient propagation, DCN-R achieves a more balanced and extensible representation learning framework.

### **3. Overall Architecture & Data Flow**

The DCN-R is architected as a multi-stream feature learning pipeline designed to produce a single, highly predictive ranking score. Its architecture follows a sequential flow of four distinct stages.

1.  **Embedding & Input Formulation:** Raw input features (collaborative, categorical, and numerical) are transformed into a single, dense input vector, $x_0$.
2.  **Parallel Feature Extraction:** The vector $x_0$ is fed simultaneously into a **Cross Network** for explicit interaction learning and a **Deep Residual Network** for implicit abstraction learning.
3.  **Fusion & Combination:** The output vectors from both sub-networks are concatenated into a final, comprehensive representation.
4.  **Final Prediction:** A final linear transformation is applied to the combined vector to produce a single scalar logit, which serves as the ranking score.

This parallel design ensures that the model learns both explicit feature crosses and implicit deep patterns concurrently.


<img width="561" height="1160" alt="АРХИтектура drawio" src="https://github.com/user-attachments/assets/415d265d-4871-42b3-89a6-438196c13556" />


### **4. Architectural Components in Detail**

#### **4.1. Input & Embedding Layer**

This layer transforms the sparse, multi-type input into a unified, dense, real-valued vector ($x_0$).

*   **Handling High-Cardinality Features via Embeddings:** Features like `user_id` and `item_id` are mapped to learned, low-dimensional vector representations. An **Embedding Matrix** $E \in \mathbb{R}^{|V| \times k}$ is created for each feature, where $|V|$ is the vocabulary size and $k$ is the embedding dimension. The model performs a lookup to retrieve the corresponding vector for each feature index. These embedding matrices are trainable parameters.

*   **Handling Numerical Features:** Continuous features like `price_rub` are normalized using **Min-Max Scaling** to a consistent range via the formula:

   
$$
x'_{ij} = \frac{x_{ij} - \min(X_j)}{\max(X_j) - \min(X_j)}
$$


*   **Formation of the Final Input Vector ($x_0$):** All embedding vectors and normalized numerical features are concatenated to form the final input vector:


       x₀ = [emb(user_id) ⊕ emb(item_id) ⊕ … ⊕ norm_numerical_features]

    Where $\oplus$ denotes concatenation.


#### **4.2. The Memorization Component: Cross Network**

The Cross Network explicitly and efficiently learns feature interactions of bounded degree. It consists of a sequence of `CrossLayer`s, each defined by the formula:

$$
   x_{l+1} = x_{0} \odot (x_{l}^T w_{l}) + b_{l} + x_{l}
$$

Where:
*   $x_l, x_{l+1} \in \mathbb{R}^d$: Are the output vectors from the $l$-th and $(l+1)$-th cross layers.
*   $x_0 \in \mathbb{R}^d$: Is the original input vector from the embedding layer.
*   $w_l, b_l \in \mathbb{R}^d$: Are the learnable weight and bias parameters for the $l$-th layer.
*   $\odot$: Denotes the element-wise product (Hadamard product).

This elegant formula allows the Cross Network to explicitly construct $(l+1)$-order interactions at each layer $l$ with only $O(d)$ parameters, making it an extremely efficient **memorization engine**. The final $+ x_l$ term is a residual connection that preserves learned interactions from previous layers.

#### **4.3. The Generalization Component: Deep Residual Network**

The Deep Network is designed as a **generalization engine** to discover hidden, implicit, and high-level patterns. To overcome the vanishing gradient problem in deep networks, it is built upon **Residual Blocks (ResBlocks)**. Each block performs the transformation:


$$
x_{l+1} = \text{ReLU}(F(x_l, \{W_i\}) + x_l)
$$

Where:
*   $x_l$ and $x_{l+1}$: Are the input and output vectors for the $l$-th residual block.
*   $F(x_l, \{W_i\})$: Is the residual function learned by the layers within the block (e.g., `Linear -> BatchNorm -> ReLU -> Dropout -> ...`).
*   $+ x_l$: Is the **identity shortcut connection**, which allows gradients to flow unimpeded, enabling stable training of a very deep and expressive network.


#### **4.4. Combination & Output Layer**

The final stage integrates the knowledge from both streams.

*   **Feature Fusion:** The output vector from the final CrossLayer (`cross_out`) and the output vector from the final ResBlock (`deep_out`) are concatenated:


$$
\mathbf{v}_{\text{final}} = [ \mathbf{cross}_{\text{out}} \oplus \mathbf{deep}_{\text{out}} ]
$$

$$
\text{Score} = \mathbf{w}_{\text{final}}^T \mathbf{v}_{\text{final}} + b_{\text{final}}
$$

    
*   **Score Prediction:** This combined vector is passed through a final, single-neuron linear layer to produce an unbounded scalar score (logit):


$$
\text{Score} = \mathbf{w}_{\text{final}}^T \mathbf{v}_{\text{final}} + b_{\text{final}}
$$


   No final activation function is used, as only the relative order of the scores is required for the ranking task.

### **5. Training Methodology & Experimental Validation**

#### 5.1. Objective Function

Although the final output is used for ranking, the model is trained to solve a binary classification problem: predicting the probability of a positive outcome (`was_booked = 1`). For this task, we employ the **Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`)** as the objective function. This is the canonical choice for binary classification and is both numerically stable and effective.

The loss $\mathcal{L}$ for a batch of $N$ examples is calculated as:


$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$


Where:
*   $y_i$ is the ground-truth binary label (`1` for `was_booked`, `0` otherwise).
*   $\hat{y}_i$ is the raw scalar logit output by the model.
*   $\sigma(\cdot)$ is the sigmoid function, which converts the logit into a probability.

The `BCEWithLogitsLoss` function is highly optimized, as it combines the sigmoid operation and the BCE calculation in a single, numerically stable step. Minimizing this loss function directly encourages the model to output high scores for positive examples and low scores for negative examples, which is an excellent proxy for optimizing ranking quality.


#### **5.2. Optimization Strategy**

We use the **AdamW optimizer** for its superior handling of weight decay. The learning rate is dynamically adjusted using a **`ReduceLROnPlateau` scheduler**, which reduces the learning rate when the validation loss stops improving.

#### Experimental Setup and Results

To ensure the robustness of our model and validate our architectural choices, we conducted a rigorous experimental process, including automated hyperparameter optimization and a comprehensive ablation study.

##### **Hyperparameter Optimization**

The optimal combination of architectural and training parameters was discovered using the **Optuna framework**. We defined a wide search space for key parameters such as `emb_dim`, `hidden_dim`, `n_cross_layers`, `n_res_blocks`, `dropout`, `lr`, and `weight_decay`. The optimization process, driven by a TPE sampler and a Median Pruner, was run for 300 trials with an Early Stopping mechanism.

<img width="700" height="500" alt="optimization_history" src="https://github.com/user-attachments/assets/4624e5b6-c25b-4aad-9889-953a1752e3b8" />

*Figure 1: Optimization history. The plot shows a clear convergence to a low LogLoss value of 0.1886, demonstrating the effectiveness of the automated tuning.*

<img width="700" height="500" alt="param_importances" src="https://github.com/user-attachments/assets/ee16b646-9719-4c68-bb48-8ce080e9c8a8" />

*Figure 2: Hyperparameter importances. The results reveal that regularization parameters (`dropout`, `weight_decay`) and the `learning rate` were the most critical factors for achieving strong generalization on our high-signal dataset.*

The process converged on an **elegant, efficient, and heavily regularized architecture** (`dropout=0.6`, `emb_dim=16`, `n_res_blocks=1`), rather than a "large-as-possible" model, perfectly tailoring it to the signal-to-noise ratio of the training data.

##### **5.3 Ablation Studies**

To quantify the contribution of each architectural component, we conducted a series of ablation studies. For each variant, a separate, independent hyperparameter search (`n_trials=100`) was performed.

| Model Variant | Description | Best Val LogLoss ↓ | Best Val AUC ↑ |
| :--- | :--- | :---: | :---: |
| 1. **DCN-R (Full Model)** | The complete hybrid architecture. | 0.1886 | 0.9426 |
| 2. **Cross Network Only** | The Deep Network component was removed. | 0.1886 | **0.9440** |
| 3. Deep Network Only | The Cross Network component was removed. | 0.1901 | 0.9404 |
| 4. DCN with standard MLP | The ResBlocks were replaced with a standard MLP. | 0.1901 | **0.9392** |

*(Note: Lower LogLoss is better, Higher AUC is better)*

**Analysis of Results:** The ablation study yielded several critical insights:
*   **The Primacy of Explicit Feature Interactions:** The exceptional performance of the "Cross Network Only" variant (Model 2) and the poor performance of the "Deep Network Only" variant (Model 3) unequivocally demonstrate that the predictive signal in our dataset is **dominated by explicit feature interactions**. Our data preprocessing strategy created a "clean," **memorization-dominant** problem space where a well-tuned Cross Network excels.
*   **The Detrimental Effect of an Inadequate Deep Component:** The "DCN with standard MLP" variant (Model 4) showed the worst performance in terms of AUC. This is a crucial finding, suggesting that adding a standard MLP component **actively hindered the performance** of the powerful Cross Network, likely by introducing noise.
*   **Validation of the ResNet-based Architecture:** In contrast, our full **DCN-R model** (Model 1) performed on par with the best model. This demonstrates the robustness of the ResNet-based deep component. Unlike the standard MLP, it was capable of learning without degrading the overall performance, proving its superiority as a generalization component within a hybrid architecture.

  
#### 5.4 Analysis of Optimal Hyperparameters

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

   6.3. Model Registry and Artifact Management

   To ensure robustness, versioning, and scalability, our architecture separates the storage of model metadata from the storage of large binary artifacts. We implemented a Model Registry pattern using our primary PostgreSQL database.
   A dedicated table, ml_models, serves as a central catalog for all trained models. For each model version, this table stores:
   Metadata: The model's version tag, creation timestamp, performance metrics (AUC, LogLoss), and the optimal hyperparameters discovered during training.
   
   ointers: Instead of storing the large binary files directly, the table holds paths to the model weights (.pth), item embeddings (.npy), and other serialized objects (.gz).
   The large artifacts themselves are stored in a dedicated, high-throughput Object Storage service (e.g., Amazon S3, Google Cloud Storage).
   This decoupled approach provides several key advantages:
   
   Scalability: The API service can efficiently download artifacts from the object store without overloading the transactional database.
   Versioning and Governance: The Model Registry acts as a single source of truth, providing a clear history of all experiments and production models.
   
   Atomic Deployments: Promoting a new model to production is a simple, atomic transaction in the database (updating an is_active flag), which makes the deployment process safe and easily reversible.
   
   Upon startup, the API service queries the registry for the active model version, retrieves the artifact paths, downloads them from the object store, and loads them into memory. This ensures that the service always runs with the correct, centrally-managed model version.

#### 7.1. Conclusion

In this work, we presented the **DCN-R**, a hybrid neural architecture that effectively addresses the trade-off between memorization and generalization. Through a rigorous process of hyperparameter optimization and extensive ablation studies, we demonstrated the model's ability to achieve a strong validation performance (**AUC: 0.9426**, **LogLoss: 0.1886**).

Our key finding is that for high-signal, low-noise recommendation tasks, a well-tuned **Cross Network serves as the primary engine of predictive power**. Furthermore, we empirically proved that the architectural choice of the deep component is critical, with a **Residual Network** providing a stable and robust generalization module, whereas a standard MLP can be detrimental to performance. By synthesizing these two specialized components, the DCN-R provides a powerful and adaptable foundation for building high-performance, real-world recommender systems.




  
