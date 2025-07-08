# Task: Predicting Camera Direction (Orientation)

This component of the **GeoLens** project is dedicated to a single, challenging task: **predicting the camera's compass direction (from 0 to 360 degrees) using only the visual information from a photograph.**


---

### 1. The Core Challenge: The Circular Nature of Angles

Predicting a directional angle is fundamentally different from standard regression tasks (like predicting house prices). The core challenge is the **periodicity of angles**.

Consider these two scenarios:
- **Prediction:** 1°, **True Value:** 359°
- **Prediction:** 170°, **True Value:** 180°

A standard loss function like Mean Squared Error (MSE) would calculate the errors as:
- `(1 - 359)² = (-358)² = 128,164`
- `(170 - 180)² = (-10)² = 100`

The model would be punished astronomically for being off by only 2 degrees, while the 10-degree error is considered much smaller. This flawed feedback prevents the model from learning effectively. It fails to understand that 359° and 1° are adjacent.

### 2. Our Solution: A Geometrically-Aware Approach

To overcome this, we re-frame the problem from predicting a 1D angle to predicting a 2D point on a unit circle. This aligns the model's objective with the true geometric nature of angles.

#### A. Representing Angles as Points

Any angle `θ` can be uniquely represented by its `(x, y)` coordinates on a circle of radius 1 using cosine and sine:
- `x = cos(θ)`
- `y = sin(θ)`

Now, the angles 1° and 359° are represented by points that are very close to each other on the circle's circumference, accurately reflecting their proximity.

#### B. The Model's Task

Instead of predicting one value `θ`, the model's final layer is designed to output **two continuous values**:
1.  The predicted `cosine` of the angle.
2.  The predicted `sine` of the angle.

#### C. The Loss Function: `CosineSineLoss`

We use a custom loss function that calculates the **Euclidean distance** between the predicted point and the true point on the circle.

**Loss = (cos_pred - cos_true)² + (sin_pred - sin_true)²**

This loss is minimized when the predicted point is exactly the same as the true point, providing a smooth and accurate gradient for the model to learn from.

#### D. Converting Predictions Back to Angles

After the model outputs its `[cos_pred, sin_pred]` vector, we convert it back into a human-readable angle in degrees using the **two-argument arctangent (`atan2`) function**:

**Angle (in radians) = `atan2(sin_pred, cos_pred)`**
**Angle (in degrees) = Angle (in radians) * 180 / π**

The `atan2` function is essential as it correctly handles all four quadrants, giving a result between -180° and 180°, which we can then map to the 0-360° range.

### 3. Model Architecture and Implementation

- **Backbone:** The model leverages the powerful **ConvNeXt-Tiny** architecture, pre-trained on ImageNet. This provides a strong foundation of general visual feature recognition, a technique known as Transfer Learning.

- **Prediction Head:** The original classifier of ConvNeXt is replaced with a custom head:
  1.  `AdaptiveAvgPool2d`: Summarizes the spatial feature maps into a fixed-size vector.
  2.  `Flatten`: Converts the 2D feature map into a 1D vector.
  3.  `Linear` -> `ReLU` -> `Linear`: A small Multi-Layer Perceptron (MLP) to process the features.
  4.  **Final `Linear` layer with 2 output neurons**, corresponding to the `[cos(θ), sin(θ)]` predictions.