# es335-24-fall-assignment-4
ES 335: Machine Learning Fall 2024, IIT Gandhinagar - Assignment 4

Here is the table for task 1:

| Model                                             | Training Time | Training Loss | Training Accuracy | Testing Accuracy | Number of Parameters |
|---------------------------------------------------|---------------|---------------|-------------------|-------------------|-----------------------|
| VGG (1 block)                                     | 24.44 seconds | 0.7406        | 0.7000            | 70.000%           | 67,110,017           |
| VGG (3 blocks)                                    | 24.80 seconds | 0.4614        | 0.8000            | 80.000%           | 16,870,721           |
| VGG (3 blocks) (with data augmentation)           | 23.65 seconds | 0.5235        | 0.7000            | 70.000%           | 16,870,721           |
| Transfer learning using VGG16 (all layers tuned)  | 52.78 seconds | 0.1236        | 0.9750            | 97.500%           | 165,721,921          |
| Transfer learning using VGG16 (MLP layers only)   | 41.44 seconds | 0.2596        | 0.9500            | 95.000%           | 165,721,921          |
