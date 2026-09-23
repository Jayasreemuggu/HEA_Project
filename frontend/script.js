const elements = [
    "Ag","Al","B","C","Ca","Co","Cr","Cu","Fe","Ga","Hf","I",
    "Li","Mg","Mn","Mo","Nb","Nd","Ni","O","Pd","Re","Ru","S",
    "Sc","Si","Sn","T","Ta","Ti","V","W","Y","Zn","Zr"
];

const compositionDiv = document.getElementById("composition");

elements.forEach(element => {
    const wrapper = document.createElement("div");

    wrapper.innerHTML = `
        <label>${element} (at.%)</label>
        <input
            id="element_${element}"
            type="number"
            min="0"
            max="100"
            step="0.1"
            value="0"
        >
    `;

    compositionDiv.appendChild(wrapper);
});

async function predictYS() {

    const composition = {};

    elements.forEach(element => {
        composition[element] = Number(
            document.getElementById(`element_${element}`).value
        );
    });

    const total = Object.values(composition).reduce(
        (sum, value) => sum + value,
        0
    );

    if (Math.abs(total - 100) > 0.001) {
        document.getElementById("result").innerText =
            `Composition must sum to 100 at.%. Current total: ${total.toFixed(3)} at.%`;
        return;
    }

    const body = {
        composition: composition,
        Test_Temperature_C:
            Number(document.getElementById("temperature").value),
        Phase:
            document.getElementById("phase").value,
        Test_Type:
            document.getElementById("testType").value
    };

    document.getElementById("result").innerText =
        "Running prediction...";

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(body)
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed");
        }

        document.getElementById("result").innerText =
            `Predicted Yield Strength: ${data.predicted_ys_mpa.toFixed(2)} MPa`;

    } catch (error) {

        document.getElementById("result").innerText =
            `Error: ${error.message}`;

    }
}
