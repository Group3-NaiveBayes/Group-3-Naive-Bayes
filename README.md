# Group-3-Naive-Bayes
COMP 3009 - Final Project - Naive Bayes Classification - Weather Classification

# Naive Bayes — Weather Classification (Rain? Yes/No)

A tiny, math-first implementation of a **Naive Bayes** classifier using **Python built-ins only**.  
We classify whether it will **rain** based on four categorical features: `Outlook`, `Temperature`, `Humidity`, and `Windy`.  
An optional check with `scikit-learn`’s `MultinomialNB` is included for comparison.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Math in One Minute](#math-in-one-minute)
- [Files in This Repo](#files-in-this-repo)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [What You’ll See (Sample Output)](#what-youll-see-sample-output)
- [Tuning & Experiments](#tuning--experiments)
- [How It Works (Implementation Notes)](#how-it-works-implementation-notes)
- [Team](#team)

---

## Project Overview

- **Goal:** Show how Bayes’ Theorem + the conditional independence assumption directly produce a working classifier.
- **Scope:** A small, fully transparent demo (14 rows of categorical weather data).
- **Why:** Reinforce the connection between **probability math** and **computational problem-solving**.

This project was completed for **COMP 3009 — Applied Math for Data Science & AI (University of Denver)**.  
It connects theory to practice by hand-coding a classifier and verifying results using scikit-learn.

---

## Math in One Minute

Bayes’ Theorem provides a way to update beliefs based on evidence:

\[
P(Y|X) = \frac{P(X|Y) \times P(Y)}{P(X)}
\]

- **P(Y)** → Prior probability of a class (e.g., Rain = Yes or No)  
- **P(X|Y)** → Likelihood of seeing certain weather features given the class  
- **P(Y|X)** → Posterior probability that it rains, given current conditions  

The **Naive** assumption simplifies this by treating all features as independent:

\[
P(X|Y) = \prod_i P(x_i | Y)
\]

Even though this assumption isn’t always true (e.g., temperature and humidity are often correlated),  
it makes computation tractable — and surprisingly effective.

Laplace Smoothing (α = 1) is applied to prevent zero probabilities:

\[
P(x_i|Y) = \frac{\text{count}(x_i, Y) + \alpha}{\text{count}(Y) + \alpha k}
\]

---

## Files in This Repo

| File | Description |
|------|--------------|
| **`Group3_NaiveBayes_Code.py`** | Full Python implementation — computes priors, likelihoods, Laplace smoothing, and predictions; includes sklearn comparison. |
| **`Code_Walkthrough.pdf`** | Step-by-step explanation of each function and how math is implemented in code. |
| **`Final_Project_Summary.pdf`** | Formal written summary connecting mathematical foundations to data science and AI concepts. |
| **`Group3_NaiveBayes_Presentation.pptx`** | 10-minute presentation slides summarizing the project — includes real-world use cases, course connections, results, and visuals. |
---

## Requirements

You only need Python’s standard library — but you can optionally install scikit-learn for verification.

```bash
pip install scikit-learn
```
---

## Quickstart

# Clone the repo
git clone https://github.com/Group3-NaiveBayes/Group-3-Naive-Bayes.git

cd Group-3-Naive-Bayes

# Run the classifier
python Group3_NaiveBayes_Code.py

---

## What You’ll See (Sample Output)

Empirical class priors (unsmoothed):
  P(Yes) ≈ 0.6429 (9/14)
  P(No)  ≈ 0.3571 (5/14)

Smoothed priors used by the model:
  P(Yes) = 0.6250
  P(No)  = 0.3750

Example likelihoods (with smoothing):
  P(Outlook=Sunny | Y=Yes) = 0.25
  P(Humidity=High | Y=No)  = 0.71

New day: {'Outlook':'Sunny','Temperature':'Mild','Humidity':'High','Windy':False}
Predicted: No
Posterior probabilities: {'Yes': 0.4117, 'No': 0.5883}

Leave-One-Out Accuracy: 0.50
sklearn LOO accuracy: 0.64

---

## Tuning & Experiments

- Try changing the **Laplace smoothing α** parameter.
- Replace `Windy` Boolean with a categorical feature (e.g., `"Calm"`, `"Breezy"`, `"Gusty"`).
- Add more observations to improve model reliability.
- Compare performance using different validation strategies (e.g., k-fold cross-validation).
- Test the model with slightly modified data to observe sensitivity to feature distribution.
---

## How It Works (Implementation Notes)

---

## Team

**Group 3 – Naive Bayes Classification Project**

| Name | Role |
|------|------|
| **Jordon Abrams** | Math & Implementation |
| **Jonny Tahai** | Research & Documentation |
| **Sam Kippur** | Testing & Presentation |

University of Denver • COMP 3009 – Applied Math for Data Science & AI   
Date: November 2025
