// ============================================================
// AI SOCIAL MEDIA CRISIS DETECTION
// Website JavaScript
// ============================================================

// ------------------------------------------------------------
// DOM Elements
// ------------------------------------------------------------

const postInput = document.getElementById("postInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultSection = document.getElementById("resultSection");
const resultLabel = document.getElementById("resultLabel");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const resultMessage = document.getElementById("resultMessage");

const crisisExampleBtn = document.getElementById("crisisExample");
const normalExampleBtn = document.getElementById("normalExample");


// ------------------------------------------------------------
// Example Posts
// ------------------------------------------------------------

const crisisText =
    "Flood water has entered several houses and people need immediate help.";

const normalText =
    "I watched a movie with my friends yesterday.";


// ------------------------------------------------------------
// Crisis Example Button
// ------------------------------------------------------------

if (crisisExampleBtn) {
    crisisExampleBtn.addEventListener("click", function () {
        postInput.value = crisisText;
        postInput.focus();
    });
}


// ------------------------------------------------------------
// Normal Example Button
// ------------------------------------------------------------

if (normalExampleBtn) {
    normalExampleBtn.addEventListener("click", function () {
        postInput.value = normalText;
        postInput.focus();
    });
}


// ------------------------------------------------------------
// Analyze Button
// ------------------------------------------------------------

if (analyzeBtn) {

    analyzeBtn.addEventListener("click", async function () {

        const text = postInput.value.trim();

        // Empty input check
        if (text === "") {

            showMessage(
                "Please enter a social media post first.",
                "warning"
            );

            return;
        }


        // Loading state
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = "⏳ Analyzing...";


        try {

            /*
             * The website sends the text to the Python backend.
             *
             * Backend endpoint:
             * POST /predict
             *
             * The backend will use your trained
             * DistilBERT model.
             */

            const response = await fetch(
                "http://127.0.0.1:8000/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        text: text
                    })
                }
            );


            // Check response
            if (!response.ok) {

                throw new Error(
                    "Backend server returned an error."
                );

            }


            const data = await response.json();


            // Display prediction
            displayResult(data);


        } catch (error) {

            console.error(error);

            /*
             * For now, if the Python backend is not running,
             * show a helpful message.
             */

            showMessage(
                "Backend is not connected. Start the Python API server and try again.",
                "error"
            );

        } finally {

            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = "🔍 Analyze Post";

        }

    });

}


// ------------------------------------------------------------
// Display Prediction Result
// ------------------------------------------------------------

function displayResult(data) {

    if (!resultSection) {
        return;
    }


    resultSection.classList.remove("hidden");


    // Get prediction label
    const prediction =
        data.prediction ||
        data.label ||
        data.result ||
        "UNKNOWN";


    // Get confidence
    let confidence =
        data.confidence ||
        0;


    /*
     * Backend may return confidence as:
     *
     * 0.89
     *
     * or
     *
     * 89
     */

    if (confidence <= 1) {
        confidence = confidence * 100;
    }


    confidence = Math.max(
        0,
        Math.min(100, confidence)
    );


    // Display label
    resultLabel.textContent =
        prediction.toUpperCase();


    // Display confidence
    confidenceValue.textContent =
        confidence.toFixed(2) + "%";


    // Animate confidence bar
    setTimeout(function () {

        confidenceBar.style.width =
            confidence + "%";

    }, 100);


    // Result message
    if (
        prediction.toLowerCase().includes("informative") &&
        !prediction.toLowerCase().includes("not")
    ) {

        resultMessage.textContent =
            "⚠️ This post appears to contain crisis-related or informative information.";

        resultMessage.className =
            "result-message crisis";

    } else {

        resultMessage.textContent =
            "✅ This post appears to be non-crisis or not informative.";

        resultMessage.className =
            "result-message normal";

    }

}


// ------------------------------------------------------------
// Message Helper
// ------------------------------------------------------------

function showMessage(message, type) {

    if (!resultSection) {
        return;
    }


    resultSection.classList.remove("hidden");


    resultLabel.textContent =
        type === "error"
            ? "ERROR"
            : "WARNING";


    confidenceValue.textContent =
        "--";


    confidenceBar.style.width =
        "0%";


    resultMessage.textContent =
        message;


    resultMessage.className =
        "result-message " + type;

}


// ------------------------------------------------------------
// Smooth Scroll to Result
// ------------------------------------------------------------

function scrollToResult() {

    if (!resultSection) {
        return;
    }


    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });

}


// ------------------------------------------------------------
// Character Counter
// ------------------------------------------------------------

if (postInput) {

    const counter =
        document.getElementById("characterCount");


    if (counter) {

        postInput.addEventListener(
            "input",
            function () {

                counter.textContent =
                    postInput.value.length + " characters";

            }
        );

    }

}


// ------------------------------------------------------------
// Enter Key Support
// ------------------------------------------------------------

if (postInput) {

    postInput.addEventListener(
        "keydown",
        function (event) {

            if (
                event.ctrlKey &&
                event.key === "Enter"
            ) {

                analyzeBtn.click();

            }

        }
    );

}


// ============================================================
// PAGE LOADED
// ============================================================

console.log(
    "🚨 AI Social Media Crisis Detection website loaded."
);

console.log(
    "Model: DistilBERT"
);

console.log(
    "Task: Crisis Detection"
);