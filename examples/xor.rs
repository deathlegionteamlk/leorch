use leorch::tensor::Tensor;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn main() {
    println!("=== Leorch XOR Example ===\n");

    let x_train = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y_train = vec![0.0, 1.0, 1.0, 0.0];

    println!("Training data:");
    println!("X shape: [4, 2]");
    println!("Y shape: [4, 1]\n");

    println!("Creating neural network...");
    println!("Architecture: 2 -> 4 -> 1\n");

    let mut w1 = vec![
        vec![0.5, -0.2, 0.1, 0.8],
        vec![-0.3, 0.7, -0.5, 0.2],
    ];
    let mut b1 = vec![0.0, 0.0, 0.0, 0.0];
    
    let mut w2 = vec![0.4, -0.6, 0.3, 0.9];
    let mut b2 = 0.0;

    let lr = 0.5;
    let num_epochs = 10000;
    let print_interval = 1000;

    println!("Training parameters:");
    println!("  Learning rate: {}", lr);
    println!("  Epochs: {}", num_epochs);
    println!("  Loss function: MSE\n");
    println!("Starting training...\n");

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        
        let mut dw1 = vec![vec![0.0; 4]; 2];
        let mut db1 = vec![0.0; 4];
        let mut dw2 = vec![0.0; 4];
        let mut db2 = 0.0;

        for i in 0..4 {
            let x1 = x_train[i][0];
            let x2 = x_train[i][1];
            let target = y_train[i];

            let z1_0 = x1 * w1[0][0] + x2 * w1[1][0] + b1[0];
            let z1_1 = x1 * w1[0][1] + x2 * w1[1][1] + b1[1];
            let z1_2 = x1 * w1[0][2] + x2 * w1[1][2] + b1[2];
            let z1_3 = x1 * w1[0][3] + x2 * w1[1][3] + b1[3];

            let a1_0 = relu(z1_0);
            let a1_1 = relu(z1_1);
            let a1_2 = relu(z1_2);
            let a1_3 = relu(z1_3);

            let z2 = a1_0 * w2[0] + a1_1 * w2[1] + a1_2 * w2[2] + a1_3 * w2[3] + b2;
            let a2 = sigmoid(z2);

            let error = a2 - target;
            total_loss += error * error;

            let dz2 = error * sigmoid_derivative(z2);
            
            dw2[0] += dz2 * a1_0;
            dw2[1] += dz2 * a1_1;
            dw2[2] += dz2 * a1_2;
            dw2[3] += dz2 * a1_3;
            db2 += dz2;

            let da1_0 = dz2 * w2[0];
            let da1_1 = dz2 * w2[1];
            let da1_2 = dz2 * w2[2];
            let da1_3 = dz2 * w2[3];

            let dz1_0 = da1_0 * relu_derivative(z1_0);
            let dz1_1 = da1_1 * relu_derivative(z1_1);
            let dz1_2 = da1_2 * relu_derivative(z1_2);
            let dz1_3 = da1_3 * relu_derivative(z1_3);

            dw1[0][0] += dz1_0 * x1;
            dw1[1][0] += dz1_0 * x2;
            dw1[0][1] += dz1_1 * x1;
            dw1[1][1] += dz1_1 * x2;
            dw1[0][2] += dz1_2 * x1;
            dw1[1][2] += dz1_2 * x2;
            dw1[0][3] += dz1_3 * x1;
            dw1[1][3] += dz1_3 * x2;
            
            db1[0] += dz1_0;
            db1[1] += dz1_1;
            db1[2] += dz1_2;
            db1[3] += dz1_3;
        }

        for i in 0..2 {
            for j in 0..4 {
                w1[i][j] -= lr * dw1[i][j] / 4.0;
            }
        }
        for i in 0..4 {
            b1[i] -= lr * db1[i] / 4.0;
        }
        for i in 0..4 {
            w2[i] -= lr * dw2[i] / 4.0;
        }
        b2 -= lr * db2 / 4.0;

        let avg_loss = total_loss / 4.0;

        if epoch % print_interval == 0 || epoch == num_epochs - 1 {
            println!("Epoch [{}/{}], Loss: {:.6}", epoch + 1, num_epochs, avg_loss);
        }

        if avg_loss < 0.01 {
            println!("\nConverged at epoch {}!", epoch + 1);
            break;
        }
    }

    println!("\n=== Final Evaluation ===\n");

    println!("Predictions:");
    println!("Input    | Target | Prediction | Correct?");
    println!("---------|--------|------------|----------");

    let mut correct_count = 0;

    for i in 0..4 {
        let x1 = x_train[i][0];
        let x2 = x_train[i][1];
        let target = y_train[i];

        let z1_0 = x1 * w1[0][0] + x2 * w1[1][0] + b1[0];
        let z1_1 = x1 * w1[0][1] + x2 * w1[1][1] + b1[1];
        let z1_2 = x1 * w1[0][2] + x2 * w1[1][2] + b1[2];
        let z1_3 = x1 * w1[0][3] + x2 * w1[1][3] + b1[3];

        let a1_0 = relu(z1_0);
        let a1_1 = relu(z1_1);
        let a1_2 = relu(z1_2);
        let a1_3 = relu(z1_3);

        let z2 = a1_0 * w2[0] + a1_1 * w2[1] + a1_2 * w2[2] + a1_3 * w2[3] + b2;
        let pred = sigmoid(z2);

        let pred_binary = if pred > 0.5 { 1.0 } else { 0.0 };
        let correct = (pred_binary - target).abs() < 0.5;
        if correct { correct_count += 1; }

        println!("[{:.0}, {:.0}]   | {:.0}      | {:.4}     | {}",
            x1, x2, target, pred, if correct { "Yes" } else { "No" });
    }

    let accuracy = (correct_count as f32 / 4.0) * 100.0;

    println!("\nAccuracy: {:.1}%", accuracy);

    if accuracy >= 75.0 {
        println!("\nXOR problem solved successfully!");
    } else {
        println!("\nXOR problem not fully solved.");
    }

    println!("\n=== Demonstrating Leorch Tensor Operations ===\n");

    let x_tensor = Tensor::from_slice(
        &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        &[4, 2],
    ).expect("Failed to create tensor");
    
    println!("Created input tensor:");
    println!("  Shape: {:?}", x_tensor.shape());
    println!("  Device: {:?}", x_tensor.device);
    println!("  Requires grad: {}", x_tensor.requires_grad);

    let ones = Tensor::ones(&[2, 3]);
    println!("\nOnes tensor [2, 3]:");
    println!("  Shape: {:?}", ones.shape());

    let eye = Tensor::eye(3);
    println!("\nEye tensor [3, 3]:");
    println!("  Shape: {:?}", eye.shape());

    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let c = a.matmul(&b).expect("Matmul failed");
    println!("\nMatrix multiplication:");
    println!("  A @ B result shape: {:?}", c.shape());

    println!("\n=== Example Complete ===");
}
