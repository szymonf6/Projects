import torch

# Tworzenie tensora
tensor_a = torch.tensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

tensor_b = torch.tensor([[9, 8, 7],
                        [6, 5, 4],
                        [3, 2, 1]])

# Dodawanie tensorów
tensor_sum = tensor_a + tensor_b

# Mnożenie tensorów
tensor_product = tensor_a * tensor_b

# Transpozycja tensora
tensor_transpose = tensor_a.t()

# Wyświetlenie wyników
print("Tensor A:")
print(tensor_a)

print("Tensor B:")
print(tensor_b)

print("Dodawanie tensorów:")
print(tensor_sum)

print("Mnożenie tensorów:")
print(tensor_product)

print("Transpozycja tensora A:")
print(tensor_transpose)
