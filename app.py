from flask import Flask, render_template
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

def run_market_basket_analysis(file_path):
    # Bagian 1 & 2: Membaca dan memproses data (Tidak ada perubahan di sini)
    df = pd.read_csv(file_path, encoding='unicode_escape')
    df.columns = df.columns.str.strip()
    df.dropna(axis=0, subset=['InvoiceNo', 'CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype('int64')
    df = df[~df['InvoiceNo'].astype(str).str.contains('C')]
    df.dropna(axis=0, subset=['Description'], inplace=True)
    df['Description'] = df['Description'].str.strip()

    basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
              
    transaction_count = len(basket)
    basket_sets = (basket >= 1)
    
    if 'POSTAGE' in basket_sets.columns:
        basket_sets.drop('POSTAGE', inplace=True, axis=1)

    # Bagian 3 & 4: Menjalankan Algoritma (Tidak ada perubahan signifikan di sini)
    frequent_itemsets = apriori(basket_sets, min_support=0.04, use_colnames=True)
    
    if frequent_itemsets.empty:
        return transaction_count, "No frequent itemsets found."

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
    
    if rules.empty:
        return transaction_count, "No association rules found."

    # ===================== PENYESUAIAN TAMPILAN TABEL =====================
    
    # Memilih hanya kolom yang paling penting untuk ditampilkan
    rules_simplified = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    
    # Mengurutkan aturan untuk menemukan yang terbaik
    rules_simplified = rules_simplified.sort_values(['confidence', 'lift'], ascending=[False, False])
    
    # Mengembalikan DataFrame yang sudah disederhanakan
    return transaction_count, rules_simplified
    # =====================================================================

# Rute Flask (Tidak ada perubahan di sini)
@app.route('/')
def index():
    try:
        transaction_count, analysis_result = run_market_basket_analysis('online_retail_20k.csv')
        
        if isinstance(analysis_result, str):
            message = analysis_result
            return render_template('index.html', message=message, transaction_count=transaction_count)

        return render_template('index.html', 
                               tables=[analysis_result.to_html(classes='data', header="true", index=False)], 
                               transaction_count=transaction_count)

    except FileNotFoundError:
        return "Error: File 'online_retail_20k.csv' tidak ditemukan."
    except Exception as e:
        return f"Terjadi kesalahan saat pemrosesan: {e}"

if __name__ == '__main__':
    app.run(debug=True)