# Definimos un texto de ejemplo para extraer nuestro vocabulario inicial
# En un proyecto real, esto sería tu archivo 'input.txt' de entrenamiento
data = "Hola, estoy aprendiendo a entrenar mi propia IA en mi RTX 4060."

# 1. Creamos el vocabulario: todos los caracteres únicos que existen en nuestro texto
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Caracteres únicos: {''.join(chars)}")
print(f"Tamaño del vocabulario: {vocab_size}")

# 2. Mapeo: Carácter -> Entero e Entero -> Carácter
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# 3. Funciones de codificación y decodificación
encode = lambda s: [stoi[c] for c in s] # string -> lista de enteros
decode = lambda l: ''.join([itos[i] for i in l]) # lista de enteros -> string

# --- TEST ---
test_string = "Hola IA"
encoded = encode(test_string)
decoded = decode(encoded)

print(f"Original: {test_string}")
print(f"Tokenizado (IDs): {encoded}")
print(f"Decodificado: {decoded}")
