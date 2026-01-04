import torch


x = torch.tensor(4.0)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
y_true = torch.tensor(10.0)

learning_rate = 0.01
traning_complete = False
step = 0
while not traning_complete: 
  
    step += 1

    y = w * x + b

    loss = (y - y_true)**2


    loss.backward()
    

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    w.grad.zero_()
    b.grad.zero_()

    print(f"\n\nStep {step}: y = {y.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}, loss = {loss.item():.4f}")
 
    distance_from_y_tolerance = 0.01
    if abs(y.item() - y_true.item()) <= distance_from_y_tolerance:
        print("AI gets a point! 🎉")
        traning_complete = True
    else:
        print("Keep training... 🚀")

