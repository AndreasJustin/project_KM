# 1. Import library pandas
import pandas as pd

# 2. Tentukan nama file input dan output
file_input = 'online_retail.csv'
file_output = 'online_retail_10k.csv'
jumlah_baris = 10000

# Pesan untuk pengguna
print(f"Membaca {jumlah_baris} baris pertama dari file '{file_input}'...")

try:
    # 3. Baca hanya N baris pertama dari file CSV
    # Parameter 'nrows' sangat efisien karena tidak memuat seluruh file ke memori.
    # 'encoding='unicode_escape'' seringkali dibutuhkan untuk dataset retail online.
    df_trimmed = pd.read_csv(file_input, encoding='unicode_escape', nrows=jumlah_baris)

    # 4. Simpan DataFrame yang sudah dipotong ke file CSV baru
    # 'index=False' digunakan agar tidak ada kolom indeks tambahan yang tidak perlu di file baru.
    # 'encoding='utf-8'' adalah standar encoding yang baik untuk file output.
    print(f"Menyimpan data ke file baru '{file_output}'...")
    df_trimmed.to_csv(file_output, index=False, encoding='utf-8')

    # 5. Berikan konfirmasi keberhasilan kepada pengguna
    print("\n=====================")
    print("      BERHASIL!      ")
    print("=====================")
    print(f"Sebanyak {len(df_trimmed)} baris data telah diambil dari '{file_input}'.")
    print(f"Data tersebut telah disimpan dalam file baru bernama '{file_output}'.")

except FileNotFoundError:
    print(f"\nERROR: File '{file_input}' tidak ditemukan.")
    print("Pastikan file tersebut berada di folder yang sama dengan skrip Python ini.")
except Exception as e:
    print(f"\nTerjadi sebuah error: {e}")