# 🧠 Pattern Recognition Systems (PRS)

This repository contains C++ implementations of key algorithms and concepts from the **Pattern Recognition Systems** course at **TUCN Cluj**.
The code is based on laboratory exercises provided by [Prof. Ion Giosan](https://users.utcluj.ro/~igiosan/Resources/PRS/PRS_labs.pdf).

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Labs](#labs)

   * [Lab 3: Hough Transform](#lab-3-hough-transform)
   * [Lab 4: Distance Transform](#lab-4-distance-transform)
   * [Lab 5: Statistical Data Analysis](#lab-5-statistical-data-analysis)
 
---

## 🧩 Overview

This project explores classic computer vision and pattern recognition techniques implemented using **OpenCV** and **C++**.

The repository includes:

* Implementation of **Hough Transform** for line detection
* **Chamfer Distance Transform** for shape matching
* **Statistical analysis** (mean, covariance, correlation matrices) on facial image datasets

 
```
opencv_core
opencv_imgproc
opencv_highgui
```

---

## 🔬 Labs

### 🧮 Lab 3 — Hough Transform

**Goal:** Detect straight lines in edge images using the Hough Transform accumulator.

**Key Functions:**

* `hough(Mat& img, int w, int k)`
* `drawHoughLines(Mat& originalEdgeImg, const vector<peak>& peaks)`

**Workflow:**

1. Compute the Hough accumulator.
2. Identify peaks corresponding to dominant lines.
3. Draw detected lines on the image.

**Example:**

```cpp
Mat img = imread("Images/edge_simple.bmp", IMREAD_GRAYSCALE);
hough(img, 3, 6);
```

---

### 🧭 Lab 4 — Distance Transform

**Goal:** Compute the **Chamfer Distance Transform** and use it for template matching.

**Key Functions:**

* `chamfer_dt(Mat& img)`
* `matching_score(const Mat& img, const Mat& randomImg)`

**Workflow:**

1. Compute distance transform on a binary template.
2. Compare template against unknown objects.
3. Lower scores → better matches.

**Example:**

```cpp
Mat img = imread("Images/template.bmp", IMREAD_GRAYSCALE);
Mat obj = imread("Images/unknown_object1.bmp", IMREAD_GRAYSCALE);
Mat dt = chamfer_dt(img);
double score = matching_score(img, obj);
```

---

### 📊 Lab 5 — Statistical Data Analysis

**Goal:** Perform statistical analysis on face images (mean, covariance, correlation).

**Key Functions:**

* `load_img()`
* `compute_and_save_mean_vector(const Mat& I)`
* `compute_and_save_covariance_matrix(const Mat& I)`
* `compute_and_save_correlation_matrix(const Mat& I)`
* `task5_analyze_all_pairs(const Mat& I)`

**Workflow:**

1. Load all 400 face images.
2. Flatten them into feature vectors.
3. Compute mean, covariance, and correlation matrices.
4. Analyze correlations between image pixels (features).

**Example:**

```cpp
Mat I = load_img();
compute_and_save_mean_vector(I);
compute_and_save_covariance_matrix(I);
compute_and_save_correlation_matrix(I);
task5_analyze_all_pairs(I);
```

---
 
