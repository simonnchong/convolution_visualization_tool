# DeepLab: CNN Workbench 🧠

[](https://www.google.com/search?q=https://simonnchong.github.io/convolution_visualization_tool/)
[](https://opensource.org/licenses/MIT)

**DeepLab** is an interactive educational tool designed to demystify the **Convolutional Neural Network (CNN)**. It provides a hands-on environment for students and AI enthusiasts to visualize exactly how mathematical operations transform raw images into abstract **feature maps**.

## 🌟 Key Features

  * **Interactive Input Layer:** Draw digits or shapes directly on a **14x14 pixel grid** or upload your own image.
  * **Real-Time Convolution:** Visualizes the entire pipeline with customizable parameters:
      * **Kernels:** Experiment with Edge Detection, Sharpen, and Emboss filters.
      * **Activation Functions:** Observe the effects of **ReLU**, Sigmoid, and Tanh activation on feature values.
  * **Mathematical Transparency:**
      * **Microscope Mode:** Hover over any pixel in the output feature map to reveal the **exact calculation** (`Input × Weight + Bias`) that created that single value.
      * **Animation Mode:** Watch the kernel physically slide across the input grid step-by-step to understand the convolution process.
  * **Dynamic Parameters:** Adjust **Kernel Size** (3x3, 5x5) and **Stride** to see how dimensionality changes instantly.

## 🚀 Live Demo

Try the tool directly in your browser:
**[https://simonnchong.github.io/convolution\_visualization\_tool/](https://www.google.com/search?q=https://simonnchong.github.io/convolution_visualization_tool/)**

## 🛠️ Tech Stack

  * **Framework:** React (v18)
  * **Build Tool:** Vite
  * **Styling:** Tailwind CSS
  * **Icons:** Lucide React
  * **Deployment:** GitHub Pages (via GitHub Actions)

## 💻 Running Locally

If you want to run this project on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/simonnchong/convolution_visualization_tool.git
    ```
2.  **Navigate to the directory:**
    ```bash
    cd convolution_visualization_tool
    ```
3.  **Install dependencies (requires Node.js):**
    ```bash
    npm install
    ```
4.  **Start the development server:**
    ```bash
    npm run dev
    ```

## 📄 License

This project is open source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
