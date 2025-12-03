import json
import numpy as np
import pandas as pd
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

def regenerate_split():
    """Regenera split.json basado en train.csv actual"""
    
    # Leer CSV
    train_csv = file_path / "data" / "train.csv"
    df = pd.read_csv(train_csv)
    n_samples = len(df)
    
    print(f"📊 Total de muestras en train.csv: {n_samples}")
    
    # Dividir 80/20
    val_size = int(n_samples * 0.2)
    train_size = n_samples - val_size
    
    # Generar índices aleatorios sin reemplazo
    all_indices = np.arange(n_samples)
    np.random.seed(42)  # Para reproducibilidad
    np.random.shuffle(all_indices)
    
    train_indices = all_indices[:train_size].tolist()
    val_indices = all_indices[train_size:].tolist()
    
    # Crear diccionario
    split_dict = {
        "train": train_indices,
        "val": val_indices
    }
    
    # Guardar JSON
    split_json = file_path / "data" / "split.json"
    with open(split_json, "w") as f:
        json.dump(split_dict, f, indent=2)
    
    print(f"✅ Nuevo split.json guardado:")
    print(f"   Train: {len(train_indices)} muestras")
    print(f"   Val: {len(val_indices)} muestras")
    print(f"   Guardado en: {split_json}")
    
    # Verificación
    print(f"\n🔍 Verificación:")
    print(f"   Max train index: {max(train_indices)} (debe ser < {n_samples})")
    print(f"   Max val index: {max(val_indices)} (debe ser < {n_samples})")

if __name__ == "__main__":
    regenerate_split()