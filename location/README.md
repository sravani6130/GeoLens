# Task: Predicting Geographic Location (Latitude & Longitude)

This component of the **GeoLens** project is responsible for predicting the precise geographic coordinates—**latitude and longitude**—of a photograph, using only its visual content. This is a classic computer vision problem known as **Visual Geolocalization**.



---

### 1. The Core Challenge: Mapping Pixels to a Globe

Predicting continuous latitude and longitude values is a regression problem. The main challenges are:

1.  **Scale and Precision:** Latitude and longitude values are high-precision floating-point numbers. A tiny change in a value can correspond to a significant distance on the ground.
2.  **Visual Ambiguity:** Different locations can look very similar (e.g., two different grassy fields on campus), while the same location can look very different depending on the season, weather, or time of day.
3.  **Data Noise:** The training data may contain outliers—images taken far from the primary area of interest—which can confuse the model and degrade performance.

### 2. Our Solution: A Robust Regression Pipeline

To address these challenges, we implemented a carefully designed pipeline involving data cleaning, normalization, and a powerful regression model.

#### A. Data Pre-processing: Cleaning and Normalization

1.  **Outlier Removal:** We first analyze the distribution of GPS coordinates in the training set and filter out the top and bottom 2% of values. This removes extreme outliers, allowing the model to focus on learning the visual features of the core campus area.
2.  **Z-Score Normalization:** We then calculate the mean (μ) and standard deviation (σ) for both latitude and longitude from the filtered training set. Every coordinate is then normalized:
    `normalized_value = (original_value - μ) / σ`
    This transforms the target values to a small, zero-centered range, which is optimal for neural network training, leading to faster convergence and a more stable learning process.

#### B. The Model's Task

The model is trained to predict the **normalized** latitude and longitude. The loss is calculated on these normalized values.

#### C. The Loss Function: `MSELoss`

We use the **Mean Squared Error (MSE)** loss function. It measures the average squared difference between the predicted normalized coordinates and the true normalized coordinates. This is a standard and effective loss function for regression tasks.

#### D. Converting Predictions Back to Coordinates

For the final output, the model's normalized predictions are converted back to the original latitude/longitude scale using the means and standard deviations calculated during pre-processing:
`original_value = (normalized_value * σ) + μ`

### 3. Model Architecture and Implementation

- **Backbone:** The model uses the **ConvNeXt-Base** architecture, accessed via the `timm` library and pre-trained on ImageNet. This larger backbone provides greater capacity for learning the subtle visual details required for precise localization.

- **Prediction Head:** A custom regression head is attached to the backbone:
  1.  `AdaptiveAvgPool2d` and `Flatten`: To create a compact feature vector from the backbone's output.
  2.  A multi-layer perceptron with `ReLU` activations and a `Dropout` layer (`p=0.5`) for regularization, preventing overfitting.
  3.  **Final `Linear` layer with 2 output neurons**, corresponding to the predicted normalized latitude and longitude.

