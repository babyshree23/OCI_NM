let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let clearBtn = document.getElementById("clear");
let predictBtn = document.getElementById("predict");
let resultDisplay = document.getElementById("result");

let model;
let isDrawing = false;

// Initialize canvas
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 10;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

// Drawing logic
canvas.addEventListener("mousedown", () => { isDrawing = true });
canvas.addEventListener("mouseup", () => { isDrawing = false; ctx.beginPath(); });
canvas.addEventListener("mousemove", draw);

function draw(event) {
    if (!isDrawing) return;
    ctx.lineTo(event.offsetX, event.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.offsetX, event.offsetY);
}

// Clear canvas
clearBtn.addEventListener("click", () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultDisplay.textContent = "None";
});

// Load model
async function loadModel() {
    model = await tf.loadLayersModel("model/model.json");
    console.log("Model loaded successfully");
}
loadModel();

// Preprocess image and predict digit
predictBtn.addEventListener("click", async () => {
    let imgData = ctx.getImageData(0, 0, 280, 280);
    let inputTensor = preprocessImage(imgData);
    
    let prediction = model.predict(inputTensor);
    let digit = prediction.argMax(1).dataSync()[0];
    
    resultDisplay.textContent = digit;
});

// Preprocess the image before feeding into model
function preprocessImage(imgData) {
    let tensor = tf.browser.fromPixels(imgData, 1) // Convert to grayscale
        .resizeNearestNeighbor([28, 28]) // Resize to 28x28
        .toFloat()
        .div(tf.scalar(255)) // Normalize
        .expandDims(0); // Add batch dimension
    return tensor;
}
