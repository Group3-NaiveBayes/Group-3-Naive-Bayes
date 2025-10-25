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

---

## Math in One Minute

We classify by maximizing the posterior:

\[
P(Y\mid X) \propto P(Y)\prod_i P(x_i \mid Y)
\]

- **Priors** \(P(Y)\): class frequencies (smoothed).
- **Likelihoods** \(P(x_i \mid Y)\): per-feature categorical frequencies (smoothed).
- **Laplace smoothing** (\(\alpha=1\)):  
  \[
  P(x_i\mid Y)=\frac{\text{count}(x_i,Y)+\alpha}{\text{count}(Y)+\alpha\cdot k}
  \]
  where \(k\) is the number of possible values of that feature.
- Computation uses **log-probabilities** to avoid underflow; final posteriors are normalized.

---

## Files in This Repo

## Requirements

## Quickstart

## What You’ll See (Sample Output)

## Tuning & Experiments

## How It Works (Implementation Notes)

## Team

