import os
import pandas as pd
import statistics

# --- Configurazione ---
# Assicurati che questo percorso punti alla directory principale del tuo dataset
DATASET_DIR = "/projects/0/prjs1482/UvA/AA_DATASET/dataset"

def analizza_lunghezza_descrizioni(directory_dataset):
    """
    Analizza tutti i file .csv in un dataset per calcolare la lunghezza media
    e la deviazione standard delle prime 4 descrizioni in ogni file.
    """
    
    lunghezze_descrizioni = []
    cartelle_analizzate = 0
    
    print(f"Avvio dell'analisi nella directory: {directory_dataset}\n")

    # Itera su ogni elemento nella directory del dataset
    for nome_cartella in sorted(os.listdir(directory_dataset)):
        percorso_cartella = os.path.join(directory_dataset, nome_cartella)
        
        # Assicurati che sia una cartella
        if not os.path.isdir(percorso_cartella):
            continue
            
        # Trova il file CSV corrispondente nella cartella
        percorso_csv = None
        for file in os.listdir(percorso_cartella):
            if file.startswith("trials_for_") and file.endswith(".csv"):
                percorso_csv = os.path.join(percorso_cartella, file)
                break
        
        if not percorso_csv:
            print(f"ATTENZIONE: Nessun file .csv trovato nella cartella {nome_cartella}. Cartella saltata.")
            continue
            
        try:
            # Leggi il file CSV con pandas
            df = pd.read_csv(percorso_csv, encoding="utf-8-sig")
            
            # Controlla che ci siano le colonne e le righe necessarie
            if 'msg' not in df.columns:
                print(f"ATTENZIONE: La colonna 'msg' non è presente in {percorso_csv}. Cartella saltata.")
                continue
            
            if len(df) < 4:
                print(f"ATTENZIONE: Il file {percorso_csv} ha meno di 4 righe. Cartella saltata.")
                continue

            # Prendi le prime 4 descrizioni
            prime_4_descrizioni = df['msg'].head(4)
            
            # Calcola la lunghezza (numero di parole) di ogni descrizione e aggiungila alla lista
            for descrizione in prime_4_descrizioni:
                # str(descrizione) gestisce eventuali valori non-stringa (es. NaN)
                lunghezza = len(str(descrizione).split())
                lunghezze_descrizioni.append(lunghezza)
            
            cartelle_analizzate += 1

        except Exception as e:
            print(f"ERRORE: Impossibile processare il file {percorso_csv}. Dettagli: {e}")

    # --- Calcoli finali e stampa dei risultati ---
    if not lunghezze_descrizioni:
        print("\nAnalisi completata, ma non sono state trovate descrizioni valide da analizzare.")
        return

    descrizioni_totali = len(lunghezze_descrizioni)
    lunghezza_media = statistics.mean(lunghezze_descrizioni)
    
    # La deviazione standard richiede almeno 2 valori
    if descrizioni_totali > 1:
        dev_std = statistics.stdev(lunghezze_descrizioni)
    else:
        dev_std = 0

    print("\n--- RISULTATI DELL'ANALISI ---\n")
    print(f"Numero di cartelle analizzate con successo: {cartelle_analizzate}")
    print(f"Numero totale di descrizioni analizzate:   {descrizioni_totali}")
    print("\n")
    print(f"Lunghezza media delle descrizioni: {lunghezza_media:.2f} parole")
    print(f"Deviazione standard:               {dev_std:.2f} parole")
    print("\n----------------------------------\n")


# Esegui la funzione di analisi
if __name__ == "__main__":
    analizza_lunghezza_descrizioni(DATASET_DIR)