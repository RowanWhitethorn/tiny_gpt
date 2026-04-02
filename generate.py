import torch
from model_v2 import GPTLanguageModel # Asumiendo que moviste la clase a un archivo aparte

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Reconstruir la estructura (debe tener los mismos hiperparámetros)
model = GPTLanguageModel().to(device)

# 2. Cargar el conocimiento
model.load_state_dict(torch.load('shakespeare_gpt_v1.pth'))
model.eval()

# 3. Usar la IA
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = model.generate(context, max_new_tokens=1000)[0].tolist()
print(decode(generated_chars))