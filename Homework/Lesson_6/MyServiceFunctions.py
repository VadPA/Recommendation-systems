import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
# working with text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import SpaceTokenizer
from nltk.corpus import stopwords
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import itertools

# Модель второго уровня
from lightgbm import LGBMClassifier

# Написанные нами функции
from metrics import precision_at_k, recall_at_k
from utils import prefilter_items
from recommenders import MainRecommender

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')


# соберем pipeline

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.key]


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


class OHEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        self.columns = []

    def fit(self, X, y=None):
        self.columns = [col for col in pd.get_dummies(X, prefix=self.key).columns]
        return self

    def transform(self, X):
        X = pd.get_dummies(X, prefix=self.key)
        test_columns = [col for col in X.columns]
        for col_ in self.columns:
            if col_ not in test_columns:
                X[col_] = 0
        return X[self.columns]


class OHEEncoderBin(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        self.columns = []

    def fit(self, X, y=None):
        B = [col for col in pd.get_dummies(X, prefix=self.key).columns]
        self.columns = B[:1]
        return self

    def transform(self, X):
        X = pd.get_dummies(X, prefix=self.key)

        return X[self.columns]


class NumericPower(BaseEstimator, TransformerMixin):
    def __init__(self, key, p=2):
        self.key = key
        self.columns = []
        self.p = p + 1

    def fit(self, X, y=None):
        B = [self.key + str(i) for i in range(1, self.p)]
        self.columns = B + ['log']
        return self

    def transform(self, X):
        Xp = X.values.reshape(-1, 1)
        for i in range(2, self.p):
            Xp = np.hstack([Xp, (X.values.reshape(-1, 1) ** i).astype(float)])

        Xp = np.hstack([Xp, np.log(X.values.reshape(-1, 1) + 1).astype(float)])
        B = pd.DataFrame(data=Xp, index=X.index, columns=[self.columns])
        return B[self.columns]


class TextImputer(BaseEstimator, TransformerMixin):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.key] = X[self.key].fillna(self.value)
        return X


def replace_commodity_desc(text):
    '''
    перевод текста в нижний регистр и удаление ненужных символов в названии товара
    '''
    if not isinstance(text, str):
        text = str(text)

    if text in My_Dict:
        text = My_Dict[text]

    return text


def replace_sub_commodity_desc(text):
    '''
    перевод текста в нижний регистр и удаление ненужных символов в названии товара
    '''
    if not isinstance(text, str):
        text = str(text)

    new_str = []
    for el in text.split():
        if el in My_dict_sub_desc:
            new_str.append(My_dict_sub_desc[el])
        else:
            new_str.append(el)

    return ' '.join(new_str)


def lower_text(text):
    '''
    перевод текста в нижний регистр и удаление ненужных символов в названии товара
    '''
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = text.strip('\n').strip('\r').strip('\t')

    text = re.sub("[0-9]|[%©«»?*!@#№$^•·&]", '', text)
    text = re.sub("[-—.,:;_()]", ' ', text)
    text = re.sub("[+=]|[/-]", ' ', text)

    text = ' '.join(text.split())

    return text


def text_without_stopword(text):
    '''
    избавляемся от стоп-слов
    '''
    if not isinstance(text, str):
        text = str(text)

    words = SpaceTokenizer().tokenize(text)

    words_lem_without_stopwords = [i for i in words if not i in stopwords.words('english')]

    return " ".join(words_lem_without_stopwords)


My_Dict = {'FRZN ICE': 'frozen ice',
           'PNT BTR/JELLY/JAMS': 'peanut butter jelly jams',
           'ICE CREAM/MILK/SHERBTS': 'ice cream milk sherbets',
           'DINNER MXS:DRY': 'dinner mix dry',
           'FRZN VEGETABLE/VEG DSH': 'frozen vegetable vegetable dish',
           'FRZN FRUITS': 'frozen fruits',
           'HOUSEHOLD CLEANG NEEDS': 'household cleaning needs',
           'FD WRAPS/BAGS/TRSH BG': 'food wraps bags trash bags',
           'PICKLE/RELISH/PKLD VEG': 'pickle/relish/pickled veg',
           'DRY BN/VEG/POTATO/RICE': 'dry fruits veg potato rice',
           'FACIAL TISS/DNR NAPKIN': 'facial tissue dinner napkin',
           'REFRGRATD DOUGH PRODUCTS': 'refrigerated dough products',
           'SUGARS/SWEETNERS': 'sugars sweeteners',
           'BABY HBC': 'baby health and beauty care',
           'BEANS - CANNED GLASS & MW': 'canned glass and microwave',
           'SOAP - LIQUID & BAR': 'soap liquid and soap bar',
           'CRACKERS/MISC BKD FD': 'crackers miscellaneous baked food',
           'CONVENIENT BRKFST/WHLSM SNACKS': 'convenient breakfast wholesome snacks',
           'CANDY - CHECKLANE': 'candy - checklane',
           'FRZN MEAT/MEAT DINNERS': 'frozen meat meat dinners',
           'WATER - CARBONATED/FLVRD DRINK': 'water - carbonated flavored drink',
           'FRZN BREAKFAST FOODS': 'frozen breakfast foods',
           'ELECTRICAL SUPPPLIES': 'electrical supplies',
           'FRZN JCE CONC/DRNKS': 'frozen ice concentrated drinks',
           'MOLASSES/SYRUP/PANCAKE MIXS': 'molasses syrup pancake mix',
           'MEAT - MISC': 'meat miscellaneous',
           'LUNCHMEAT': 'lunch meat',
           'SALD DRSNG/SNDWCH SPRD': 'salad dressing sandwich spread',
           'REFRGRATD JUICES/DRNKS': 'refrigerated juices refrigerated drinks',
           'FRZN NOVELTIES/WTR ICE': 'frozen novelties water ice',
           'GREETING CARDS/WRAP/PARTY SPLY': 'greeting cards wrap party supply',
           'PWDR/CRYSTL DRNK MX': 'power drink crystal drink mix',
           'MISC. DAIRY': 'miscellaneous dairy',
           'FRZN POTATOES': 'frozen potatoes',
           'SEAFOOD - MISC': 'seafood miscellaneous',
           'DISPOSIBLE FOILWARE': 'disposable foil ware',
           'SNKS/CKYS/CRKR/CNDY': 'snacks cakes crackers candy',
           'MISC WINE': 'miscellaneous wine',
           'COUPON/MISC ITEMS': 'coupon miscellaneous items',
           'DELI SPECIALTIES (RETAIL PK)': 'deli specialties retail packaging',
           'PREPARED/PKGD FOODS': 'prepared packaged foods',
           'COUPONS/STORE & MFG': 'coupons store and manufacturing',
           'NATURAL HBC': 'natural health and beauty care',
           'BABYFOOD': 'baby food',
           'FRZN SEAFOOD': 'frozen seafood',
           'RW FRESH PROCESSED MEAT': 'relative weight fresh processed meat',
           'NDAIRY/TEAS/JUICE/SOD': 'dairy teas juice sod',
           'MISCELLANEOUS HBC': 'miscellaneous health and beauty care',
           'SPORTS MEMORABLILIA': 'sports memorabilia',
           'PKG.SEAFOOD MISC': 'package seafood miscellaneous'
           }

My_dict_sub_desc = {'sft': 'soft',
                    'frz': 'frozen',
                    'sw': 'sweet',
                    'drnk': 'drink',
                    'btl': 'bottle',
                    'carb': 'carbohydrates',
                    'incl': 'including',
                    'cke': 'cake',
                    'fds': 'food',
                    'iqf': 'individually quick frozen',
                    'mlt': 'multipack',
                    'cntrl': 'control',
                    'liqs': 'liquors',
                    'rw': 'random weight',
                    'ltr': 'litre',
                    'bbq': 'barbecue',
                    'iws': 'individually wrapped slice',
                    'exc': '',
                    'sgl': 'single',
                    'sv': 'serve',
                    'wra': 'wraps',
                    'srv': 'serve',
                    'gds': 'goods',
                    'pk': 'package',
                    'sup': 'supper',
                    'pkg': 'package',
                    'jui': 'juice',
                    'frzn': 'frozen',
                    'shldr': 'shoulder',
                    'stk': 'steak',
                    'mustar': 'mustard',
                    'prem': 'premium',
                    'flvr': 'flavor',
                    'pnt': 'peanut',
                    'mxs': 'mix',
                    'fd': 'food',
                    'dnrs': 'dinner',
                    'refrgratd': 'refrigerated',
                    'sweetners': 'sweeteners',
                    'dnr': 'dinner',
                    'bg': 'bags',
                    'hbc': 'health and beauty care',
                    'mw': 'microwave',
                    'bkd': 'baked',
                    'misc': 'miscellaneous',
                    'ndairy': 'dairy',
                    'mfg': 'manufacturing',
                    'pkgd': 'packaged',
                    'cndy': 'candy',
                    'crkr': 'crackers',
                    'ckys': 'cakes',
                    'snks': 'snacks',
                    'dishwash': 'dishwasher',
                    'pwdr': 'power',
                    'crystl': 'crystal',
                    'mx': 'mix',
                    'sply': 'supply',
                    'wtr': 'water',
                    'drnks': 'drink',
                    'sald': 'salad',
                    'drsng': 'dressing',
                    'sndwch': 'sandwich',
                    'sprd': 'spread',
                    'jce': 'juice',
                    'conc': 'concentrated',
                    'flvrd': 'flavored',
                    'brkfst': 'breakfast',
                    'whlsm': 'wholesome',
                    'bulkbag': 'bulk bag',
                    'ric': 'rich',
                    'pse': 'pseudoephedrine',
                    'br': 'bread',
                    'gol': 'gold',
                    'clu': 'club',
                    'ss': 'single serve',
                    'crbntd': 'carbonated',
                    'drnking': 'drinking',
                    'mneral': 'mineral',
                    'wate': 'water',
                    'covergirl': 'cover girl',
                    'juicecombinations': 'juice combinations',
                    'saucessalsapicantee': 'sauces salsa picante',
                    'vac': 'vacuum',
                    'pac': 'packed',
                    'disp': 'dispenser',
                    'beeralemalt': 'beer ale malt',
                    'handke': 'handkerchief',
                    'hygn': 'hygienic',
                    'refrig': 'refrigeration',
                    'marinad': 'marinades',
                    'mlk': 'milk',
                    'pwdrs': 'powder',
                    'sweetyams': 'sweet yams',
                    'bulkba': 'bulk bag',
                    'lipcare': 'lip care',
                    'syrp': 'syrup',
                    'xpctrnt': 'expectorant',
                    'lozng': 'long',
                    'drp': 'dry',
                    'exce': 'except',
                    'cnd': 'canned',
                    'sngl': 'single',
                    'loave': 'loaves',
                    'reg': 'regular',
                    'ex': 'exempt',
                    'hashbrown': 'hash browns',
                    'friskie': 'friskies',
                    'swt': 'sweet',
                    'refigerated': 'refrigerated',
                    'chocol': 'chocolate',
                    'chix': 'chicken',
                    'brd': 'bread',
                    'porton': 'portion',
                    'can': 'canned',
                    'sandwic': 'sandwiches',
                    'dogfd': 'dog food',
                    'specilaty': 'specialty',
                    'dxm': 'dextromethorphan',
                    'sa': 'sauce',
                    'chococlate': 'chocolate',
                    'sal': 'salad',
                    'prpck': 'prepackaged',
                    'dps': 'dips',
                    'fre': 'fresh',
                    'pnch': 'punch',
                    'swe': 'sweet',
                    'cranapple': 'cran apple',
                    'breaders': 'breeders',
                    'containr': 'container',
                    'cr': 'creams',
                    'sandwicheshandhelds': 'sandwiches handhelds',
                    'pizzaingred': 'pizza ingredients',
                    'spf': 'sun protection factor',
                    'pican': 'picante',
                    'juic': 'juice',
                    'retl': 'retail',
                    'accss': 'access',
                    'cndles': 'candles',
                    'smi': 'semi',
                    'pickld': 'pickled',
                    'supplemen': 'supplement',
                    'otyher': 'other',
                    'oj': 'orange juice',
                    'saus': 'sausage',
                    'wholecrowns': 'whole crowns',
                    'shlfsh': 'shellfish',
                    'supp': 'supplement',
                    'bev': 'beverage',
                    'fitnessdiet': 'fitness diet',
                    'cooki': 'cookie',
                    'sociallifestyle': 'social lifestyle',
                    'occas': 'occasion',
                    'jhook': 'J-hook',
                    'protien': 'protein',
                    'deod': 'deodorant',
                    'mashedspec': 'mashed spec',
                    'catfd': 'cat food',
                    'ppk': 'prepackaged',
                    'slc': 'sliced',
                    'decaffinated': 'decaffeinated',
                    'chp': 'chop',
                    'cookng': 'cooking',
                    'bkngwares': 'baking wares',
                    'holdiay': 'holiday',
                    'merch': 'merchandise',
                    'crabetc': 'crab etc',
                    'rbr': 'rubber',
                    'nailcare': 'nail care',
                    'wt': '',
                    'whl': ''
                    }
