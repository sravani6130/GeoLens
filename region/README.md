# Task: Predicting Scene Region (Classification)

This component of the **GeoLens** project functions as a **scene classifier**. Its purpose is to analyze a photograph and categorize it into one of several predefined environmental regions (e.g., "Library," "Stadium," "Quad").



---

### 1. The Core Challenge: Class Imbalance and Visual Similarity

This classification task presents two primary challenges:

1.  **Class Imbalance:** Real-world datasets are rarely balanced. Our training data contains many more images of certain regions (e.g., "Quad") than others (e.g., "Stadium"). A naive model would become biased towards the majority classes, achieving high accuracy by simply ignoring the rare ones.
2.  **Intra-class Variation & Inter-class Similarity:** A single region (like a "Dormitory") can have many different appearances. Conversely, two different regions (like an "Admin Building" and a "Classroom Building") might share very similar architectural features, making them difficult to distinguish.

### 2. Our Solution: A Weighted and Robust Classification Strategy


#### A. Handling Class Imbalance

We use a two-pronged approach to ensure the model gives fair attention to all regions:
1.  **`WeightedRandomSampler`:** During data loading, this sampler oversamples images from the minority classes (those with fewer examples) and undersamples from the majority classes. This presents a more balanced distribution of data to the model during each training epoch.
2.  **Weighted Loss Function:** We use a `CrossEntropyLoss` function with `class_weights`. These weights assign a higher penalty when the model misclassifies an image from a rare class. This forces the model to learn the features of minority classes just as diligently as it learns the features of majority classes.

#### B. The Model's Task

The model's final layer outputs a vector of raw scores (logits), one for each possible region. The region corresponding to the highest score is chosen as the prediction. The standard **Softmax** function is implicitly applied by the `CrossEntropyLoss` to convert these scores into probabilities during training.

### 3. Model Architecture and Implementation

- **Backbone:** The model is built on the **ConvNeXt-Base** architecture, pre-trained on ImageNet. Its powerful feature extraction capabilities are ideal for distinguishing between visually complex scenes.
- **Prediction Head:**
  1.  The `timm` model is initialized with `num_classes=0` to get the raw feature vector (1024 features for ConvNeXt-Base).
  2.  A `Dropout` layer (`p=0.5`) is applied to the features for regularization to combat overfitting.
  3.  A single `Linear` layer acts as the final classifier, mapping the 1024 features to `N` outputs, where `N` is the total number of unique regions.

