import os
import re
import string
import warnings

from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import spacy

from collections import Counter
from joblib import load, dump
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback
from unidecode import unidecode
from yellowbrick.classifier import classification_report, confusion_matrix
from yellowbrick.text import FreqDistVisualizer

# Ferramentas Modelagem
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from optuna.distributions import LogUniformDistribution, IntUniformDistribution, UniformDistribution
from optuna.integration import OptunaSearchCV

# Modelos Classificação Supervisionada
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Redes neurais
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#import tensorflow as tf
#from tensorflow import keras
# Configs Tensorflow
#tf.get_logger().setLevel("ERROR")
#print(tf.__version__)
#print(tf.config.list_physical_devices())

# Configs Optuna
from optuna.exceptions import ExperimentalWarning
from optuna.logging import set_verbosity, WARNING

set_verbosity(WARNING)
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# Configs Outras Libs
pd.set_option("display.max_colwidth", 200)
shap.initjs()
sns.set_style("darkgrid")
tqdm.pandas()
warnings.filterwarnings("ignore")
