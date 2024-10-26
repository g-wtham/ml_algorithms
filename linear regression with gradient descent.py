def update_w_and_b(spendings, sales, weights, bias, alpha_lr):
    dw = 0.0 #initialize the weights and bias with 0 before gradient descent
    db = 0.0
    for i in range(len(spendings)):
        # MSE = (actual_y - predicted_y)^2 -> partial derivative of this => 2 * (actual_y - predicted_y) * dl/dw(actual_y - predicted_y) => 2 * (actual_y - predicted_y) * -x
        # dl/dw = -2 * x * [actual(y) - predicted(wx+b)] - where w = slope, x = input_val(spendings), b = bias, y = actual_output (sales); predicted_y = ax+b 
        dw += (-2/float(len(spendings))) * spendings[i] * (sales[i] - (weights*spendings[i] + bias))
        db += (-2/float(len(spendings))) * (sales[i] - (weights*spendings[i] + bias))

        #new_weight = (old_weight - dw * alpha_lr)/total_data_points
    w = (weights - dw * alpha_lr) #dont use sales as n points, as its not our input, rather it is the output that we need to predict
    b = (bias - db * alpha_lr)

    return w, b
    

def train(spendings, sales, weights, bias, alpha_lr, epochs):
    for e in range(epochs):
        w, b = update_w_and_b(spendings, sales, weights, bias, alpha_lr)
        if e % 500 == 0:
            print("At Epoch :", e, ", the loss is ", avg_loss(spendings, sales, weights, bias))
        
    return w, b


def avg_loss(spendings, sales, weights, bias):
    mean_square_error = 0.0
    for i in range(len(spendings)):
        mean_square_error += (sales[i] - (weights*spendings[i]+bias)) ** 2
    return mean_square_error/float(len(spendings))

def predict(weights, bias, spendings):
    return weights*spendings + bias

spendings = [100, 150, 200, 250, 300, 350, 400, 450, 500]
sales = [10, 15, 20, 25, 30, 35, 40, 45, 50]

w, b = train(spendings, sales, 0.0, 0.0, 0.001, 15000)
spendings_new = 23.0
y_new = predict(w, b, spendings_new)
print(y_new)