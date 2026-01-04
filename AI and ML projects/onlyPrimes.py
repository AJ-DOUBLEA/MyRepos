#This code is to see if pytorch can identify prime numbers using its weigths and bias stuff (Still new lol)

import torch

x = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

#1 is a prime and 0 is not a prime
y_true = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])  
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.01
training_complete = False
step = 0
while not training_complete:
    step += 1

    y = w * x + b

    loss = ((y - y_true)**2).mean()


    loss.backward()
    

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    w.grad.zero_()
    b.grad.zero_()
    print(f"\n\nStep {step}: y = {y}, w = {w.item():.4f}, b = {b.item():.4f}, loss = {loss.item():.4f}")
    loss_tolerance = 0.01

    if loss.item() <= loss_tolerance:
        print("Model has learned to identify primes! 🎉")
        training_complete = True


#Lol, guess it cant