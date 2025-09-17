# Heart Disease Prediction Model - Council of Education and Development Programmes Pvt. Ltd., Thane

[![GitHub stars](https://img.shields.io/github/stars/abubakarpungiwale/heart-disease-prediction?style=social)](https://github.com/abubakarpungiwale/heart-disease-prediction/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/abubakarpungiwale/heart-disease-prediction?style=social)](https://github.com/abubakarpungiwale/heart-disease-prediction/network)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This repository showcases machine learning projects developed during a NSDC-certified AI/ML & Data Science Training at **Council of Education and Development Programmes Pvt. Ltd., Thane.**. It demonstrates an end-to-end pipeline including data exploration, preprocessing, model training, and evaluation to classify patients as having heart disease.

This repository showcases machine learning projects developed during a NSDC-certified AI/ML & Data Science Training at Council of Education and Development Programmes Pvt. Ltd., Thane. It demonstrates expertise in predictive modeling, data analysis, and visualization across domains like finance, text classification, and real estate, using advanced algorithms and ensemble techniques.

**Key Highlights**:
- **Ensemble Modeling**: Random Forest as the top-performing ensemble method for robust predictions.
- **Multiple Algorithms**: Comparative analysis of Logistic Regression, Naive Bayes, SVM (linear/rbf), K-Nearest Neighbors, Decision Tree, and Random Forest.
- **Future Extensions**: Potential integration of neural networks (e.g., via TensorFlow/Keras) for enhanced deep learning-based predictions.

This project showcases proficiency in classification tasks, feature scaling, and model optimization, ideal for data science and ML engineering roles.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Technologies](#key-technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Key Technologies

- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn (for preprocessing, models, and evaluation).
- **Models**: Logistic Regression, Gaussian Naive Bayes, SVM (linear/rbf kernels), K-Nearest Neighbors (k=5), Decision Tree, Random Forest (n_estimators=500, ensemble).
- **Techniques**: StandardScaler for feature scaling, train_test_split (75/25), accuracy scoring.
- **Metric**: Accuracy score for model comparison.
- **Visualization**: Correlation heatmaps, distributions, barplots for algorithm performance.

## Installation

```bash
git clone https://github.com/abubakarpungiwale/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
```

## Methodology

- **Preprocessing**: EDA (descriptions, samples, correlations), no missing values, StandardScaler for normalization.
- **Feature Engineering**: Analyzed 13 features (e.g., age, chol, thalach) against binary target.
- **Training**: Split data (75% train, 25% test), trained 7 classifiers with default/hyperparameter-tuned settings.
- **Ensemble**: Random Forest as bagging ensemble for reduced variance and improved accuracy.

## Performance Metrics

- **Model Comparison** (Accuracy on Test Set):
  - Random Forest: ~85% (highest, ensemble strength).
  - SVM (rbf): ~82%.
  - Logistic Regression: ~80%.
  - Others: 75-80%.
- **Insights**: Barplot visualization shows Random Forest outperforming others by 5-10%. Neural networks could further boost performance in extensions.
- Detailed logs and plots in the notebook for interpretability.

## Contributing

Fork and submit pull requests for enhancements like neural network integration.

## License

MIT License - see [LICENSE](LICENSE).

## Contact

- **Author**: Abubakar Maulani Pungiwale
- **Email**: abubakarp496@email.com
- **LinkedIn**: [linkedin.com/in/abubakarpungiwale](https://linkedin.com/in/abubakarpungiwale)
- **Contact**: +91 9321782858

Connect for ML discussions or data science opportunities!

---

*Generated on September 18, 2025*
