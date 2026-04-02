import torch
from model_v2 import GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Necesitamos reconstruir el vocabulario exactamente igual que en el entrenamiento
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 2. Instanciar el modelo PASANDO el vocab_size
# Aquí estaba el error: ahora el modelo recibe el argumento
model = GPTLanguageModel(vocab_size).to(device)

# 3. Cargar los pesos que tu 4060 calculó
model.load_state_dict(torch.load('shakespeare_gpt_v1.pth'))
model.eval()

print("\n--- GPT Shakespeare Edition (Escribe 'salir' para terminar) ---")

while True:
    user_input = input("\nTu Prompt > ")
    
    if user_input.lower() in ['salir', 'exit', 'quit']:
        break
        
    if not user_input: # Si el usuario solo da Enter, le damos un espacio inicial
        user_input = " "
        
    # Convertimos tu texto a tensor y lo movemos a la GPU
    try:
        context = torch.tensor(encode(user_input), dtype=torch.long, device=device).unsqueeze(0)
        
        print("\n[Generando respuesta...]")
        # Generamos 300 tokens (puedes subirlo si quieres respuestas más largas)
        generated_ids = model.generate(context, max_new_tokens=300)[0].tolist()
        
        print("-" * 40)
        print(decode(generated_ids))
        print("-" * 40)
    except KeyError as e:
        print(f"\nError: El carácter {e} no existe en el vocabulario de Shakespeare.")