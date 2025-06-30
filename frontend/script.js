document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict_btn');
    const predictionOutput = document.getElementById('prediction_output');

    // IMPORTANT: Replace this with the actual URL of your deployed FastAPI backend
    // For local development, it might be http://127.0.0.1:8000 or http://localhost:8000
    // On Render, it will be something like https://your-fastapi-app-name.onrender.com
const BACKEND_API_URL = 'https://food-delivery-time-estimation-qnri.onrender.com';

    predictBtn.addEventListener('click', async () => {
        // Disable the button and show loading
        predictBtn.disabled = true;
        predictionOutput.innerHTML = `
            <div class="prediction-box">
                <div class="prediction-label">Calculating your estimated delivery time...</div>
                <div class="prediction-result flex justify-center items-center">
                    <svg class="animate-spin h-10 w-10 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                </div>
            </div>
        `;
        predictionOutput.classList.remove('hidden');

        try {
            // Collect input values
            const ratings = parseFloat(document.getElementById('ratings').value);
            const distance = parseFloat(document.getElementById('distance').value);
            const is_festival = parseInt(document.getElementById('is_festival').value);
            const is_weekend = parseInt(document.getElementById('is_weekend').value);
            const day_of_week = document.getElementById('day_of_week').value;
            const distance_type = document.getElementById('distance_type').value;
            const weather = document.getElementById('weather').value;
            const order_type = document.getElementById('order_type').value;
            const city_category = document.getElementById('city_category').value;
            const order_time_of_day = document.getElementById('order_time_of_day').value;

            // Prepare data for FastAPI - ensure keys match Pydantic model in main.py
            const inputData = {
                ratings: ratings,
                distance: distance,
                is_festival: is_festival,
                is_weekend: is_weekend,
                day_of_week: day_of_week,
                distance_type: distance_type,
                weather: weather,
                order_type: order_type,
                city_category: city_category,
                order_time_of_day: order_time_of_day
            };

            // Make the actual API call to your FastAPI backend
            const response = await fetch(`${BACKEND_API_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(inputData),
            });

            if (!response.ok) {
                // If the response is not OK (e.g., 400, 500 status)
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const predictedTime = data.predicted_delivery_time_minutes;

            let insightMessage = "";
            let insightColorClass = "";
            if (predictedTime > 45) {
                insightMessage = `This delivery might take longer than usual. Consider ordering from a closer restaurant or during off-peak hours.`;
                insightColorClass = "text-yellow-400";
            } else if (predictedTime < 25) {
                insightMessage = `Great! This should be a super quick delivery. Your food will be there in a flash!`;
                insightColorClass = "text-green-400";
            } else {
                insightMessage = `Normal delivery time expected. Enjoy your meal!`;
                insightColorClass = "text-blue-400";
            }

            predictionOutput.innerHTML = `
                <div class="prediction-box">
                    <div class="prediction-label">⏱️ Estimated Delivery Time</div>
                    <div class="prediction-result">${predictedTime.toFixed(1)} <span class="text-3xl">minutes</span></div>
                    <div class="text-gray-300 text-sm mt-4">
                        Approximately ${Math.round(predictedTime)} minutes
                    </div>
                    <div class="${insightColorClass} mt-6 font-semibold text-center text-lg">
                        ${insightMessage}
                    </div>
                </div>
            `;

        } catch (error) {
            console.error('Error fetching prediction:', error);
            predictionOutput.innerHTML = `
                <div class="prediction-box bg-red-800" style="background: linear-gradient(135deg, #cc2e5d 0%, #ff6f61 100%);">
                    <div class="prediction-label">Oops! Something went wrong...</div>
                    <div class="prediction-result text-white">Failed to get prediction.</div>
                    <div class="text-red-100 text-sm mt-4">
                        Please check your inputs and try again. Error: ${error.message}
                    </div>
                </div>
            `;
        } finally {
            // Re-enable the button after prediction (or error)
            predictBtn.disabled = false;
        }
    });
});

