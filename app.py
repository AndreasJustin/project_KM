from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import plotly
import plotly.express as px
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

def run_market_basket_analysis(file_path, min_support, min_confidence):
    """
    Fungsi inti untuk menjalankan analisis keranjang belanja.
    """
    try:
        df = pd.read_csv(file_path, encoding='unicode_escape')
    except Exception as e:
        return None, None, f"Gagal membaca file CSV. Pastikan formatnya benar. Error: {e}"

    # --- Pembersihan Data ---
    df.columns = df.columns.str.strip()
    df.dropna(axis=0, subset=['InvoiceNo', 'CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype('int64')
    df = df[~df['InvoiceNo'].astype(str).str.contains('C')]
    df.dropna(axis=0, subset=['Description'], inplace=True)
    df['Description'] = df['Description'].str.strip()

    # --- Transformasi Data ---
    basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
              
    transaction_count = len(basket)
    
    # Encode ke 0 dan 1
    def encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1
            
    basket_sets = basket.applymap(encode_units)
    
    # Hapus kolom POSTAGE jika ada
    if 'POSTAGE' in basket_sets.columns:
        basket_sets.drop('POSTAGE', inplace=True, axis=1)

    # --- Algoritma Apriori ---
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        return transaction_count, None, "Tidak ada itemset yang sering muncul (frequent itemsets) dengan nilai minimum support yang diberikan. Coba turunkan nilai minimum support."

    # --- Aturan Asosiasi ---
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    if rules.empty:
        return transaction_count, None, "Tidak ada aturan asosiasi yang ditemukan dengan nilai minimum confidence yang diberikan. Coba turunkan nilai minimum confidence."

    # --- Pembersihan dan Penyortiran Hasil ---
    rules['antecedents'] = rules['antecedents'].apply(lambda a: ', '.join(list(a)))
    rules['consequents'] = rules['consequents'].apply(lambda c: ', '.join(list(c)))
    
    rules_simplified = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules_simplified = rules_simplified.sort_values(['confidence', 'lift'], ascending=[False, False])
    
    return transaction_count, rules_simplified, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return redirect(url_for('index'))
        
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = "uploaded_data.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            min_support = float(request.form['min_support'])
            min_confidence = float(request.form['min_confidence'])
        except ValueError:
            return render_template('index.html', error="Nilai support dan confidence harus berupa angka.")

        transaction_count, rules_df, message = run_market_basket_analysis(filepath, min_support, min_confidence)
        
        if message:
            return render_template('index.html', message=message)

        # --- Membuat Visualisasi dengan Plotly ---
        fig = px.scatter(rules_df, 
                         x="support", 
                         y="confidence",
                         size="lift", 
                         color="lift",
                         hover_name="antecedents",
                         hover_data={"consequents": True, "lift": ':.2f', "support": ':.4f', "confidence": ':.2f'},
                         labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
                         title=f"Visualisasi Aturan Asosiasi (Support vs. Confidence)",
                         color_continuous_scale=px.colors.sequential.Viridis)
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('results.html', 
                               tables=[rules_df.to_html(classes='data', header="true", index=False)], 
                               transaction_count=transaction_count,
                               min_support=min_support,
                               min_confidence=min_confidence,
                               graphJSON=graphJSON)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)