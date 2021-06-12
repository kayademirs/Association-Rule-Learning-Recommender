import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
from helpers.helpers import check_df, retail_data_prep

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()

check_df(df)

############################################
# Görev 1: Veri Ön İşleme İşlemlerini Gerçekleştiriniz
############################################
df = retail_data_prep(df)

############################################
# ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################

df_ger = df[df['Country'] == "Germany"]


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

ger_inv_pro_df = create_invoice_product_df(df_ger, True)
ger_inv_pro_df.head()

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    return product_name


############################################
# Görev 2: Germany müşterileri üzerinden birliktelik kuralları
# üretiniz.
# Birliktelik Kurallarının Çıkarılması
############################################

# Tüm olası ürün birlikteliklerinin olasılıkları
frequent_itemsets = apriori(ger_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

# Birliktelik kurallarının çıkarılması:
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()
rules.sort_values("lift", ascending=False).head(500)


############################################
# Görev 3:
# Sepetteki kullanıcılar için ürün önerisi yapınız.
############################################

# Örnek:
# Kullanıcı 1 ürün id'si: 21987
# Kullanıcı 2 ürün id'si: 23235
# Kullanıcı 3 ürün id'si: 22747

productsIDs = [21987, 23235, 22747]
for product_id in productsIDs:
    print(product_id, check_id(df, product_id))


############################################
# Görev 4:
# Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules, 21987, 5)
arl_recommender(rules, 23235, 5)
arl_recommender(rules, 22747, 5)

############################################
# Görev 5:
# Önerilen ürünlerin isimleri nelerdir?
############################################

for product_id in arl_recommender(rules, 21987, 5):
    print(product_id, check_id(df, product_id))

for product_id in arl_recommender(rules, 23235, 5):
    print(product_id, check_id(df, product_id))

for product_id in arl_recommender(rules, 22747, 5):
    print(product_id, check_id(df, product_id))