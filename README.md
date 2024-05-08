# Predicting Shellfish Toxicity in the Adriatic Sea

## Overview
This project leverages explainable machine learning techniques to predict shellfish toxicity in the Gulf of Trieste, focusing on mussels affected by harmful algal blooms (HABs). Using a comprehensive dataset spanning 28 years, the study examines the occurrence of diarrhetic shellfish poisoning (DSP) events, enhancing the predictability and management of shellfish safety.

## Objective
The goal is to develop a machine learning model that can predict DSP events accurately, helping to establish effective early warning systems for aquaculture safety.

## Models Used
- **Random Forest**: Best overall performance for predicting DSP events.
- **SVM (Support Vector Machines)**: Included in model evaluation for its robustness in classification tasks.
- **Decision Tree Classifier**: Utilized for its interpretability and ease of representation.
- **MLPClassifier (Neural Networks)**: Explored for its capability in capturing complex patterns.

## Key Findings
- **Predictive Features**: Certain species of phytoplankton, specifically *Dinophysis fortii* and *D. caudata*, along with environmental factors like salinity, river discharge, and precipitation, were identified as significant predictors.
- **Explainability**: Used methods like permutation importance and SHAP to interpret the feature influence on model predictions, ensuring transparency and understandability of the model decisions.

## Dataset
This project uses a unique dataset that records the presence of toxic phytoplankton and environmental conditions over a 28-year period in the mussel farming areas of the Gulf of Trieste.

## Requirements
Make sure to install the following packages:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
shap
umap-learn
dtreeviz
```

## Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Execute the notebooks to train models and view the analysis.

## Contributing
We welcome contributions that:
- Improve prediction accuracy.
- Enhance model explainability.
- Update or expand the dataset.

## License
This project is released under the MIT license.

## Citation
Please cite this work in your publications if it helps your research:
```
@article{marzidovsek2024explainable,
  title={Explainable machine learning for predicting shellfish toxicity in the Adriatic Sea using long-term monitoring data of HABs},
  author={Marzidovšek, Martin and others},
  journal={arXiv preprint arXiv:2405.04372},
  year={2024}
}
```

## Contact
For questions or support, please contact Martin Marzidovšek at [email protected].


## Read the Full Paper
For more detailed insights and methodologies, read the full paper here: [Explainable machine learning for predicting shellfish toxicity](http://arxiv.org/abs/2405.04372)