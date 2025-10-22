DCN-R: A Hybrid Deep & Cross Architecture with Residual Connections for Feature Interaction Learning in Recommender Systems

Abstract
Recommender systems operating on large-scale, sparse, tabular data face a fundamental dichotomy: the need for both memorization and generalization. Memorization, the learning of frequent, explicit feature co-occurrences, is crucial for providing accurate and direct recommendations. Generalization, the discovery of novel, previously unseen feature combinations, is essential for diversity and serendipity. Standard Deep Neural Networks (DNNs) excel at generalization by capturing high-order implicit interactions but are inefficient at learning explicit feature crosses. Conversely, linear models are effective at memorization but fail to generalize. We introduce the Deep & Cross Network with Residual Blocks (DCN-R), a novel hybrid architecture designed to explicitly and efficiently learn feature interactions of both types, thereby achieving a state-of-the-art balance between memorization and generalization for ranking tasks.

1. Introduction & Architectural Principles
The core innovation of the DCN-R architecture is the parallel deployment of two specialized sub-networks that operate on the same input vector: a Cross Network and a Deep Residual Network. This explicit separation of tasks allows each component to perform its function optimally, with their outputs being combined at a final stage to produce a comprehensive prediction.

### **1.1. Memory Component: Cross Network**

#### **Architectural Principle**

The primary goal of the Cross Network is to explicitly and efficiently learn feature interactions of limited degree. While standard deep neural networks (DNNs) can learn these interactions implicitly, this process is inefficient and requires a large number of parameters. The Cross Network addresses this issue by implementing explicit cross-interactions at each layer, achieving high performance with minimal overhead.

The architecture consists of a sequence of CrossLayer layers. Each layer l takes the output of the previous layer x_l and computes the next layer x_{l+1} based on its interaction with the original vector x_0.  This process is described by the following formula:

$$
x_{l+1} = x_{0} \odot (x_{l}^T w_{l}) + b_{l} + x_{l}
$$

Where:

-   $x_l, x_{l+1} \in \mathbb{R}^d$: Are the output vectors from the $l$-th and $(l+1)$-th cross layers, respectively.
-   $x_0 \in \mathbb{R}^d$: Is the **original** input vector from the embedding layer, which is consistently used at every layer.
-   $w_l, b_l \in \mathbb{R}^d$: Are the learnable weight and bias parameters for the $l$-th layer.
-   $\odot$: Denotes the element-wise product (Hadamard product).

 #### **Formula Decomposition: The Heart of the Mechanism**

To understand how this formula creates interactions, let's break it down step by step:

1. **`x_l^T w_l` (Dot Product):** In the first step, the dot product is calculated between the vector `x_l` from the previous layer and the weight vector `w_l`. The result of this operation is a **single number (scalar)**. This scalar can be interpreted as a "score" or "degree of importance" calculated based on the features obtained in layer `l`. The vector `w_l` learns which combination of features from `x_l` to consider "important".

2. **`x_0 \odot (...)` (Interaction Creation):** Next, this single scalar is multiplied element-wise by the **original input vector `x_0`**. This operation is the core of interaction creation.  It effectively "scales" or "weights" each element of the original vector x_0 according to the "importance" calculated in the previous step. This creates explicit cross-talk between all elements of x_0 and the generalized information from x_l.

3. **`+ b_l` (Bias):** The standard learnable bias vector is added.

4. **`+ x_l` (Residual Connection):** This is a critical component. The output of the previous layer, x_l, is directly added to the result. This ensures that the interactions learned in previous layers are preserved and passed on. The model doesn't relearn everything from scratch at each new layer; it simply adds a new, more complex layer of interactions to its existing knowledge.

Thanks to this elegant formula, Cross Network explicitly constructs (l+1)-order interactions at each layer l, doing so with O(d) parameters, compared to O(d²) for standard approaches. This makes it an extremely efficient "memorization engine" that is guaranteed to find and exploit the predictive power of explicit feature combinations in the data.


### **1.2. Generalization Component: Deep Residual Network**

#### **Architectural Principle**

While the Cross Network is an expert at **remembering** explicit and predictable interactions, the Deep Network is designed as a **generalization** "engine." Its task is to discover **hidden, implicit, and high-level patterns** that cannot be expressed through simple feature combinations. It models abstract concepts that allow the system to recommend new, previously unseen, but relevant hotels to the user.

Standard deep neural networks (DNNs) face the fundamental problem of **vanishing gradient** as their depth increases, making training very deep and therefore powerful models unstable and inefficient.  To overcome this limitation, we don't use a simple sequence of layers, but instead build our Deep subnetwork on Residual Blocks, a key innovation introduced in the ResNet architecture.

#### **Architectural Unit: The Residual Block**

Instead of forcing a sequence of layers to learn the desired final transformation H(x), we force them to learn a residual function F(x) := H(x) - x. The final transformation is then reconstructed as H(x) = F(x) + x. This seemingly simple trick dramatically changes the learning dynamics.

 Each residual block `l` in our network performs the following transformation:

$$
x_{l+1} = \text{ReLU}(F(x_l, \{W_i\}) + x_l)
$$

Where:
* \( x_l \) and \( x_{l+1} \): Are the input and output vectors for the \(l\)th residual block.
* \( F(x_l, \{W_i\}) \): This is the **residual function** learned by the layers within the block. In our implementation, it is represented by the following sequence of operations:

1. `Linear_1 -> BatchNorm1d -> ReLU -> Dropout`
2. `Linear_2 -> BatchNorm1d`
* \( + x_l \): This is the **identity shortcut connection**. The input of block `x_l` is directly, without modification, added to the output of transformation `F`. 

#### **Block Architecture Decomposition: Data Flow**

1. **Main Path (Transformation Path):** The input vector `x_l` passes through the block's main computational path, which learns to find the "delta" or "residual" `F(x_l)`. `BatchNorm` is used after each linear layer to stabilize training and accelerate convergence. `ReLU` introduces nonlinearity, and `Dropout` provides regularization, preventing overfitting.

2. **Shortcut Path:** Parallel to the main path, the input `x_l` passes through the "shortcut" path without any modifications.

3. **Addition and Activation:** At the block's output, the results from both paths are summed. This allows gradients from backpropagation to flow unimpeded through the shortcut path, solving the attenuation problem. The final `ReLU` is applied to the sum. 

#### **Full Architecture of the Deep Subnetwork**

1. **Initial Projection:** Since the dimensionality of the input vector `x_0` (`input_dim`) may not match the desired dimensionality of the hidden layers (`hidden_dim`), `x_0` is first passed through a single standard linear layer (`nn.Linear`), which projects it into the desired space.

2. **Sequence of Residual Blocks:** The output of the initial layer is sequentially passed through a stack of several (two in our model) residual blocks. The output of the first block is the input to the second.

3. **Final Output:** The output vector from the last residual block, `deep_out`, is the output of the entire Deep subnetwork. This vector contains the high-level, abstract features learned by the model.

Using residual blocks allows us to build a deep and, therefore, highly expressive neural network capable of effective generalization. It doesn't compete with the Cross Network, but rather **complements** it. The Deep subnetwork takes on the task of extracting implicit, complex patterns, while the Cross Network specializes in explicit, combinatorial rules. This synergy is the key to the high accuracy of the final hybrid DCN-R model.


2. Complete Network Topology & Data Flow. 

The DCN-R follows a well-defined data flow from input to final prediction.
#Input & Embedding Layer
The model accepts raw features and transforms them into a unified numerical representation.
Collaborative & Categorical Features (user_id, item_id, city, etc.): Each unique ID or category is mapped to a low-dimensional, dense embedding vector via an nn.Embedding layer. These embeddings are learned during training.
Numerical Features (price, stars, rating_..., etc.): Continuous features are normalized (e.g., via MinMaxScaler) to ensure uniformity in scale.
All resulting vectors are then concatenated to form a single input vector, x<sub>0</sub>.

2.1 Parallel Sub-Network Processing

The input vector x_0 is simultaneously fed into the two core components:
The Deep Network processes x_0 through an initial linear projection followed by a stack of ResBlocks, producing the deep_out vector.
The Cross Network processes x_0 through a stack of CrossLayers, producing the cross_out vector.

2.2 Combination & Output Layer

The outputs from the two sub-networks are combined to form a final prediction.
Concatenation: The deep_out and cross_out vectors are concatenated into a final, comprehensive feature vector that contains information from both memorization and generalization pathways.
Final Prediction: This final vector is passed through a single, final linear neuron. The output of this neuron is a single scalar value (a logit).

3. Model Output and Objective

The final scalar output is a raw, uncalibrated score. It does not represent a probability. Its sole purpose is to serve as a ranking metric. A higher score for a user-hotel pair indicates a higher predicted relevance or compatibility. The model is trained to minimize a regression loss (e.g., Mean Squared Error) on the was_booked target, effectively learning to assign higher scores to pairs that result in a positive outcome.

4. Conclusion

The DCN-R architecture provides an elegant and effective solution to the central challenge of recommender systems. By explicitly separating the tasks of memorization and generalization into dedicated sub-networks—an efficient Cross Network and a powerful Deep Residual Network—it leverages the strengths of both paradigms. This hybrid approach allows the model to learn both direct, observable feature interactions and deep, abstract patterns, resulting in highly accurate and diverse recommendations.
