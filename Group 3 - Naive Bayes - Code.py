"""
Naive Bayes — Weather - Rain? (Yes/No)
+ Optional sklearn check

- Small, neat, fully commented
- Computes priors & likelihoods with Laplace smoothing (alpha)
- Classifies a new point via log-posteriors
- Leave-One-Out evaluation on tiny data
"""

import math
from collections import Counter, defaultdict

# 1) Tiny categorical dataset

DATA = [
    {"Outlook":"Sunny","Temperature":"Hot","Humidity":"High","Windy":False,"Rain":"No"},
    {"Outlook":"Sunny","Temperature":"Hot","Humidity":"High","Windy":True, "Rain":"No"},
    {"Outlook":"Overcast","Temperature":"Hot","Humidity":"High","Windy":False,"Rain":"Yes"},
    {"Outlook":"Rain","Temperature":"Mild","Humidity":"High","Windy":False,"Rain":"Yes"},
    {"Outlook":"Rain","Temperature":"Cool","Humidity":"Normal","Windy":False,"Rain":"Yes"},
    {"Outlook":"Rain","Temperature":"Cool","Humidity":"Normal","Windy":True, "Rain":"No"},
    {"Outlook":"Overcast","Temperature":"Cool","Humidity":"Normal","Windy":True,"Rain":"Yes"},
    {"Outlook":"Sunny","Temperature":"Mild","Humidity":"High","Windy":False,"Rain":"No"},
    {"Outlook":"Sunny","Temperature":"Cool","Humidity":"Normal","Windy":False,"Rain":"Yes"},
    {"Outlook":"Rain","Temperature":"Mild","Humidity":"Normal","Windy":False,"Rain":"Yes"},
    {"Outlook":"Sunny","Temperature":"Mild","Humidity":"Normal","Windy":True,"Rain":"Yes"},
    {"Outlook":"Overcast","Temperature":"Mild","Humidity":"High","Windy":True,"Rain":"Yes"},
    {"Outlook":"Overcast","Temperature":"Hot","Humidity":"Normal","Windy":False,"Rain":"Yes"},
    {"Outlook":"Rain","Temperature":"Mild","Humidity":"High","Windy":True,"Rain":"No"},
]

FEATURES = ["Outlook","Temperature","Humidity","Windy"]
TARGET = "Rain"
CLASSES = ["Yes","No"]  # fixed order for printing/prediction display

# 2) Domains (for smoothing k)

def feature_domains(rows):
    dom = {f:set() for f in FEATURES}
    for r in rows:
        for f in FEATURES:
            dom[f].add(r[f])
    return {f: sorted(list(v)) for f,v in dom.items()}

# 3) Train: priors & likelihoods (Laplace smoothing)

def fit_naive_bayes(rows, alpha=1.0):
    """
    Returns:
      priors[y] = P(Y=y) (smoothed)
      cond[(f, val, y)] = P(f=val | Y=y) (smoothed)
      domains[f] = list of possible values
    """
    n = len(rows)
    domains = feature_domains(rows)

    # class counts
    y_counts = Counter(r[TARGET] for r in rows)
    num_classes = len(CLASSES)

    # smoothed priors: (count + alpha) / (n + alpha * |classes|)
    priors = {y: (y_counts[y] + alpha) / (n + alpha * num_classes) for y in CLASSES}

    # conditional counts per (feature, value, class)
    fv_counts = defaultdict(int)
    y_total_for_feature = defaultdict(int)
    for r in rows:
        y = r[TARGET]
        for f in FEATURES:
            v = r[f]
            fv_counts[(f, v, y)] += 1
            y_total_for_feature[(f, y)] += 1

    # smoothed likelihoods: (count + alpha) / (count_y_for_feature + alpha*k)
    cond = {}
    for f in FEATURES:
        k = len(domains[f])
        for y in CLASSES:
            denom = y_total_for_feature[(f, y)] + alpha * k
            for v in domains[f]:
                num = fv_counts[(f, v, y)] + alpha
                cond[(f, v, y)] = num / denom

    return priors, cond, domains

# 4) Predict: argmax_y P(Y)*pi P(f_i|Y) - use logs

def predict_proba(sample, priors, cond):
    log_post = {}
    for y in CLASSES:
        s = math.log(priors[y])
        for f in FEATURES:
            v = sample[f]
            p = cond.get((f, v, y), 1e-12)  # unseen fallback
            s += math.log(p)
        log_post[y] = s
    # softmax
    m = max(log_post.values())
    exps = {y: math.exp(log_post[y] - m) for y in CLASSES}
    Z = sum(exps.values())
    return {y: exps[y]/Z for y in CLASSES}

def predict(sample, priors, cond):
    probs = predict_proba(sample, priors, cond)
    y_hat = max(probs.items(), key=lambda kv: kv[1])[0]
    return y_hat, probs

# 5) Leave-One-Out accuracy

def leave_one_out_accuracy(rows, alpha=1.0):
    correct = 0
    for i in range(len(rows)):
        train = rows[:i] + rows[i+1:]
        test = rows[i]
        priors, cond, _ = fit_naive_bayes(train, alpha=alpha)
        y_hat, _ = predict({f:test[f] for f in FEATURES}, priors, cond)
        correct += int(y_hat == test[TARGET])
    return correct / len(rows)

# 6) Run demo

if __name__ == "__main__":
    ALPHA = 1.0
    priors, cond, domains = fit_naive_bayes(DATA, alpha=ALPHA)

    # Empirical (unsmoothed) priors for display
    y_counts = Counter(r[TARGET] for r in DATA)
    total = len(DATA)
    print("Empirical class priors (unsmoothed):")
    for y in CLASSES:
        print(f"  P({y}) ≈ {y_counts[y]/total:.4f}  ({y_counts[-----------------------------------------y]}/{total})")

    print("\nSmoothed priors used by the model:")
    for y in CLASSES:
        print(f"  P({y}) = {priors[y]:.4f}")

    # Example likelihoods to show the math link
    print("\nExample likelihoods (with smoothing):")
    for (f, v) in [("Outlook","Sunny"), ("Humidity","High")]:
        for y in CLASSES:
            print(f"  P({f}={v} | Y={y}) = {cond[(f, v, y)]:.4f}")

    # Predict a new day
    new_day = {"Outlook":"Sunny","Temperature":"Mild","Humidity":"High","Windy":False}
    y_hat, probs = predict(new_day, priors, cond)
    print("\nNew day:", new_day)
    print("Predicted:", y_hat)
    print("Posterior probabilities:", {k: round(v,4) for k,v in probs.items()})

    # Leave-One-Out evaluation
    acc = leave_one_out_accuracy(DATA, alpha=ALPHA)
    print(f"\nLeave-One-Out Accuracy on tiny dataset: {acc:.3f}")

    # sklearn comparison

    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import accuracy_score
        import numpy as np

        # Stringify Windy so False is treated as a category (not "absence")
        X_dicts = []
        for row in DATA:
            d = {}
            for k, v in row.items():
                if k == "Windy":
                    d[k] = str(v)  # "True" or "False" as category
                elif k != "Rain":
                    d[k] = v
            X_dicts.append(d)
        y = [row["Rain"] for row in DATA]

        vec = DictVectorizer(sparse=True)
        X = vec.fit_transform(X_dicts)

        # set class priors to match our smoothed priors (order must match clf.classes_)
        desired_priors = {"No": (y_counts["No"] + ALPHA) / (total + ALPHA*2),
                          "Yes": (y_counts["Yes"] + ALPHA) / (total + ALPHA*2)}

        loo = LeaveOneOut()
        preds, truth = [], []
        for train_idx, test_idx in loo.split(X):
            clf = MultinomialNB(alpha=ALPHA)
            # Fit once to learn class order, then refit with fixed class_prior aligned to that order
            clf.fit(X[train_idx], np.array(y)[train_idx])
            order = list(clf.classes_)  # usually ["No","Yes"]
            priors_ordered = [desired_priors[c] for c in order]
            clf = MultinomialNB(alpha=ALPHA, class_prior=priors_ordered, fit_prior=False)
            clf.fit(X[train_idx], np.array(y)[train_idx])

            preds.append(clf.predict(X[test_idx])[0])
            truth.append(y[test_idx[0]])

        print("sklearn LOO accuracy:", accuracy_score(truth, preds))

    except ImportError:
        print("\n[sklearn check skipped — scikit-learn not installed]")
