document.getElementById("planetForm").addEventListener("submit", function(e) {
    e.preventDefault(); // prevent reload

    // Collect input values
    const data = {
        pl_rade: parseFloat(document.getElementById("pl_rade").value),
        st_lum: parseFloat(document.getElementById("st_lum").value),
        pl_orbper: parseFloat(document.getElementById("pl_orbper").value),
        st_teff: parseFloat(document.getElementById("st_teff").value),
        pl_eqt: parseFloat(document.getElementById("pl_eqt").value),
        pl_orbsmax: parseFloat(document.getElementById("pl_orbsmax").value)
    };

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        const outputDiv = document.getElementById("result");

        if (result.status === "success") {
            outputDiv.innerHTML = `
                <div class="alert alert-success">
                    <h4>🌟 Prediction Result</h4>
                    <p><strong>Status:</strong> ${result.prediction === 1 ? "Habitable" : "Not Habitable"}</p>
                    <p><strong>Score:</strong> ${result.habitability_score.toFixed(2)}</p>
                </div>
            `;
        } else {
            outputDiv.innerHTML = `
                <div class="alert alert-danger">
                    ${result.message}
                </div>
            `;
        }
    })
    .catch(error => {
        document.getElementById("result").innerHTML = `
            <div class="alert alert-danger">
                Error connecting to backend
            </div>
        `;
    });
});