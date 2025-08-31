from setuptools import setup, find_packages

setup(
    name="TransLLaMAX",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1",
        "transformers>=4.35",
        "peft>=0.5",
        "datasets>=2.14",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "shap",
        "lime",
        "pyyaml",
        "bleurt",
        "bert_score",
        "nltk",
        "rouge-score",
        "scipy"
    ],
    entry_points={
        "console_scripts": [
            "train=scripts.run_training:main",
            "infer=scripts.run_inference:main",
            "evaluate=scripts.run_evaluation:main",
            "explain=scripts.run_explainability:main",
        ],
    },
)
