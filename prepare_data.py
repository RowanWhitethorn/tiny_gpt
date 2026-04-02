import torch

# 1. Cargar el texto
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Reutilizar la lógica de tu Tokenizador
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]

# 3. Convertir todo el libro a un gran tensor de enteros
data = torch.tensor(encode(text), dtype=torch.long)

# 4. Split: 90% para entrenar, 10% para validar
# (Es como estudiar para un examen y luego hacer un simulacro con preguntas nuevas)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Total de caracteres: {len(data)}")
print(f"Dataset de entrenamiento listo: {len(train_data)} caracteres")
