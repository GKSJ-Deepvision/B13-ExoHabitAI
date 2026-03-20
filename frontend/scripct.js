let chart;

console.log("JS loaded ✅");

// theme toggle
document.getElementById("toggleTheme").onclick = () => {
    document.body.classList.toggle("light");
};

function predict() {

    const loader = document.getElementById("loader");
    const result = document.getElementById("result");
    const chartCanvas = document.getElementById("chart");
    const explain = document.getElementById("explain");
    const history = document.getElementById("history");

    const data = {
        mass: mass.value,
        radius: radius.value,
        gravity: gravity.value,
        atmosphere: atmosphere.value,
        temperature: temperature.value,
        luminosity: luminosity.value,
        distance: distance.value,
        pressure: pressure.value,
        water: water.value
    };

    // validation
    for (let key in data) {
        if (data[key] === "") {
            alert("Fill all fields!");
            return;
        }
    }

    loader.classList.remove("hidden");
    result.classList.add("hidden");

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(res => {

        loader.classList.add("hidden");

        result.classList.remove("hidden");
        result.innerHTML = `
            <h3>${res.prediction}</h3>
            <p>Confidence: ${res.confidence}%</p>
        `;

        // chart
        chartCanvas.classList.remove("hidden");

        if (chart) chart.destroy();

        chart = new Chart(chartCanvas, {
            type: 'bar',
            data: {
                labels: ['Confidence'],
                datasets: [{
                    data: [res.confidence]
                }]
            }
        });

        // explanation
        explain.classList.remove("hidden");
        explain.innerHTML = `
            <h3>🤖 AI Explanation</h3>
            <p>This planet is ${res.prediction} based on multiple environmental and stellar factors.</p>
        `;

        // history
        const li = document.createElement("li");
        li.innerText = `${res.prediction} (${res.confidence}%)`;
        history.appendChild(li);

        // sound
        new Audio("https://www.soundjay.com/buttons/sounds/button-09.mp3").play();

    })
    .catch(() => {
        loader.innerHTML = "❌ Backend Error";
    });
}
