Model Documentation: Hybrid Deep & Cross Network (DCN-V2) for Hotel Recommendation
1. High-Level Overview
This project utilizes a hybrid ranking model based on the Deep & Cross Network V2 (DCN-V2) architecture, further enhanced with Residual Blocks (ResNet).
The primary purpose of this model is not to generate recommendations from scratch, but to rank a pre-selected list of candidate hotels. Given a user and a list of ~200-500 candidate hotels, the model calculates a personalized relevance score for each user-hotel pair. The final recommended list is then sorted according to these scores.
2. Architectural Philosophy: Solving the Memorization vs. Generalization Dilemma
A core challenge in recommendation systems is balancing two distinct learning types:
Memorization: The ability to learn and exploit simple, direct, and frequently occurring rules. For example, "Users who book 5-star hotels in Sochi often book them again." This relies on learning explicit, low-order feature interactions.
Generalization: The ability to find deeper, hidden patterns and apply them to new, unseen situations. For example, identifying an abstract user taste for "quiet, family-friendly hotels with good breakfast" and recommending a new hotel that fits this profile, even if no similar users have booked it before.
A standard Deep Neural Network (DNN) is excellent at generalization but inefficient at memorization. Our DCN-V2 architecture is a hybrid model that solves this by using two specialized sub-networks in parallel, getting the best of both worlds.
3. Detailed Model Architecture
The model processes data in three main stages: the Input & Embedding Layer, the Core Dual-Network, and the Final Combination Layer.

Stage 1: Input & Embedding Layer
The model accepts a variety of raw features and transforms them into dense numerical vectors (embeddings) that a neural network can process.
Collaborative Features (user_id, item_id): Each unique user and hotel is mapped to a trainable embedding vector. This vector learns a latent representation of the user's tastes and the hotel's characteristics. Similar users and similar hotels will have vectors that are close to each other in the embedding space.
Categorical Features (city, hotel_type): Each category (e.g., "Sochi", "Hotel") is also mapped to a learned embedding vector.
Numerical Features (price_rub, stars, rating_location, etc.): These features are first normalized using a MinMaxScaler to be within a range. This ensures that no single feature dominates the learning process due to its large scale.
All of these resulting vectors are then concatenated into a single, wide input vector, x_0. This x_0 vector is the unified input for both core sub-networks.

Stage 2: Core Architecture - The Two Sub-Networks
The input vector x_0 is fed simultaneously into two parallel networks:
A. The Cross Network (The Memorization Expert)
Purpose: To explicitly and efficiently learn predictive feature interactions.
Architecture: It consists of a series of CrossLayers. Each layer is defined by the formula:
code
Code
x_next = x_0 * (x_current' · w) + b + x_current
How it Works: Each layer calculates the interactions between the current vector x_current and the original input vector x_0. This design allows the network to create increasingly complex feature crosses (e.g., city AND price, then city AND price AND stars) at each subsequent layer in a very parameter-efficient way. It is guaranteed to find important, explicit rules if they exist in the data.
B. The Deep Network (The Generalization Expert)
Purpose: To find hidden, high-level, and abstract patterns in the data that the Cross Network might miss.
Architecture: This is a deep feed-forward neural network (FFNN), but we have enhanced it with Residual Blocks (ResBlocks), inspired by ResNet.
How it Works: Instead of a simple stack of layers, the network is composed of blocks where the input to the block is added back to its output (out += identity). These "shortcut connections" allow gradients to flow more easily during training, which prevents the vanishing gradient problem and enables the stable training of deeper, more powerful networks. This network learns the abstract concepts, like a user's latent taste profile.

Stage 3: Final Combination Layer
Concatenation: The output vector from the Cross Network (containing explicit feature interactions) and the output vector from the Deep Network (containing implicit abstract patterns) are concatenated into a final, comprehensive vector.
Prediction: This final vector is fed into a single output neuron (Final Linear Layer) with a linear activation function.
Output: The model outputs a single floating-point number (a raw score or logit).

4. Model Output Interpretation
The single number produced by the model is not a probability or a rating. Its absolute value is meaningless on its own.
Its sole purpose is for ranking. A higher score for a user-item pair (U, A) than for (U, B) simply means that the model predicts user U is more likely to interact positively with hotel A than with hotel B. This allows us to sort the candidate list in a personalized and highly relevant order.

